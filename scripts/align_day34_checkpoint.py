from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from src.models.diffusion import DiffusionModel


PREFIXES = [
    "module.",
    "_orig_mod.",
    "model.",
    "diffusion.",
    "net.",
]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
            out[str(k)] = v
    return out


def get_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ("model_state", "model_state_dict", "state_dict"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items() if isinstance(v, torch.Tensor)}

    # Sometimes checkpoint itself is a raw state_dict.
    tensor_items = {str(k): v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if tensor_items:
        return tensor_items

    raise ValueError("Could not find a tensor state_dict inside checkpoint.")


def strip_known_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def find_embedding_shape(state: dict[str, torch.Tensor]) -> tuple[int | None, int | None, str | None]:
    candidates = []
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        rows, cols = int(v.shape[0]), int(v.shape[1])
        name = k.lower()
        if rows > 1000 and cols <= 1024 and any(x in name for x in ("embed", "embedding", "token")):
            candidates.append((rows, cols, k))
    if not candidates:
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                rows, cols = int(v.shape[0]), int(v.shape[1])
                if rows > 1000 and cols <= 1024:
                    candidates.append((rows, cols, k))
    if not candidates:
        return None, None, None
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    return candidates[0]


def infer_n_layers(state: dict[str, torch.Tensor]) -> int | None:
    layer_ids = set()
    for k in state:
        parts = k.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_ids.add(int(parts[i + 1]))
            if part == "encoder" and i + 2 < len(parts) and parts[i + 1] == "layers" and parts[i + 2].isdigit():
                layer_ids.add(int(parts[i + 2]))
    if layer_ids:
        return max(layer_ids) + 1
    return None


def infer_ff_dim(state: dict[str, torch.Tensor], d_model: int | None) -> int | None:
    if d_model is None:
        return None
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        rows, cols = int(v.shape[0]), int(v.shape[1])
        name = k.lower()
        if cols == d_model and rows > d_model and any(x in name for x in ("linear1", "ff", "feed", "mlp")):
            return rows
    return None


def build_constructor_kwargs(
    ckpt: Any,
    state: dict[str, torch.Tensor],
    vocab_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_config, dict):
        ckpt_config = {}

    flat = flatten_dict(ckpt_config)

    vocab_size_state, d_model_state, emb_key = find_embedding_shape(state)

    vocab_size = (
        flat.get("vocab_size")
        or flat.get("num_tokens")
        or flat.get("model.vocab_size")
        or flat.get("data.vocab_size")
        or summary_payload.get("vocab_size")
        or len(vocab_payload)
        or vocab_size_state
    )

    max_len = (
        flat.get("max_len")
        or flat.get("seq_len")
        or flat.get("sequence_length")
        or flat.get("data.max_len")
        or summary_payload.get("max_len")
        or 256
    )

    d_model = (
        flat.get("d_model")
        or flat.get("embed_dim")
        or flat.get("model.d_model")
        or flat.get("model.embed_dim")
        or d_model_state
        or 128
    )

    n_layers = (
        flat.get("n_layers")
        or flat.get("num_layers")
        or flat.get("model.n_layers")
        or flat.get("model.num_layers")
        or infer_n_layers(state)
        or 4
    )

    ff_dim = (
        flat.get("ff_dim")
        or flat.get("dim_feedforward")
        or flat.get("model.ff_dim")
        or flat.get("model.dim_feedforward")
        or infer_ff_dim(state, int(d_model))
        or 512
    )

    kwargs_all = {
        "vocab_size": int(vocab_size),
        "num_tokens": int(vocab_size),
        "max_len": int(max_len),
        "seq_len": int(max_len),
        "sequence_length": int(max_len),
        "pad_idx": int(flat.get("pad_idx") or flat.get("data.pad_idx") or 0),
        "unk_idx": int(flat.get("unk_idx") or flat.get("data.unk_idx") or 1),
        "diffusion_steps": int(flat.get("diffusion_steps") or flat.get("steps") or flat.get("diffusion.steps") or 64),
        "steps": int(flat.get("diffusion_steps") or flat.get("steps") or flat.get("diffusion.steps") or 64),
        "beta_schedule": flat.get("beta_schedule") or flat.get("diffusion.beta_schedule") or "cosine",
        "d_model": int(d_model),
        "embed_dim": int(d_model),
        "n_heads": int(flat.get("n_heads") or flat.get("num_heads") or flat.get("model.n_heads") or flat.get("model.num_heads") or 4),
        "num_heads": int(flat.get("n_heads") or flat.get("num_heads") or flat.get("model.n_heads") or flat.get("model.num_heads") or 4),
        "n_layers": int(n_layers),
        "num_layers": int(n_layers),
        "ff_dim": int(ff_dim),
        "dim_feedforward": int(ff_dim),
        "dropout": float(flat.get("dropout") or flat.get("model.dropout") or 0.10),
    }

    signature = inspect.signature(DiffusionModel)
    kwargs = {name: value for name, value in kwargs_all.items() if name in signature.parameters}

    print("\n[constructor inference]")
    print("embedding_key:", emb_key)
    print("inferred/full kwargs:", json.dumps(kwargs_all, indent=2))
    print("filtered kwargs:", json.dumps(kwargs, indent=2))

    return kwargs


def make_state_variants(raw_state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    variants: dict[str, dict[str, torch.Tensor]] = {}

    variants["raw"] = raw_state
    variants["strip_known_prefixes"] = {strip_known_prefixes(k): v for k, v in raw_state.items()}

    for prefix in PREFIXES:
        variants[f"strip_{prefix.rstrip('.')}"] = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in raw_state.items()
        }

    return variants


def score_exact_variant(model_state: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    matched = {}
    unexpected = []
    shape_mismatch = []

    for k, v in candidate.items():
        if k not in model_state:
            unexpected.append(k)
            continue
        if tuple(model_state[k].shape) != tuple(v.shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        matched[k] = v

    missing = [k for k in model_state if k not in matched]
    return {
        "matched": matched,
        "matched_count": len(matched),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


def suffix_match_state(model_state: dict[str, torch.Tensor], raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    aligned: dict[str, torch.Tensor] = {}
    used_ckpt = set()

    # First exact/stripped exact.
    for ck, cv in raw_state.items():
        stripped = strip_known_prefixes(ck)
        if stripped in model_state and tuple(model_state[stripped].shape) == tuple(cv.shape):
            aligned[stripped] = cv
            used_ckpt.add(ck)

    # Then suffix + shape match.
    model_items = list(model_state.items())
    for ck, cv in raw_state.items():
        if ck in used_ckpt:
            continue

        ck_clean = strip_known_prefixes(ck)
        ck_parts = ck_clean.split(".")

        candidates = []
        for mk, mv in model_items:
            if mk in aligned:
                continue
            if tuple(mv.shape) != tuple(cv.shape):
                continue

            mk_parts = mk.split(".")
            score = 0
            if mk == ck_clean:
                score += 100
            if mk.endswith(ck_clean) or ck_clean.endswith(mk):
                score += 50
            if len(mk_parts) >= 2 and len(ck_parts) >= 2 and mk_parts[-2:] == ck_parts[-2:]:
                score += 25
            if len(mk_parts) >= 1 and len(ck_parts) >= 1 and mk_parts[-1] == ck_parts[-1]:
                score += 5

            if score > 0:
                candidates.append((score, mk))

        if candidates:
            candidates.sort(reverse=True)
            aligned[candidates[0][1]] = cv
            used_ckpt.add(ck)

    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--out_checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min_match_ratio", type=float, default=0.98)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = get_state_dict(ckpt)

    vocab_payload = load_json(args.vocab_path)
    summary_payload = load_json(args.summary_path)

    kwargs = build_constructor_kwargs(
        ckpt=ckpt,
        state=raw_state,
        vocab_payload=vocab_payload,
        summary_payload=summary_payload,
    )

    model = DiffusionModel(**kwargs)
    model_state = model.state_dict()

    print("\n[model state]")
    print("model keys:", len(model_state))
    print("checkpoint keys:", len(raw_state))

    variant_reports = {}
    best_name = None
    best_result = None

    for name, candidate in make_state_variants(raw_state).items():
        result = score_exact_variant(model_state, candidate)
        variant_reports[name] = {
            "matched_count": result["matched_count"],
            "missing_count": len(result["missing"]),
            "unexpected_count": len(result["unexpected"]),
            "shape_mismatch_count": len(result["shape_mismatch"]),
            "first_missing": result["missing"][:20],
            "first_unexpected": result["unexpected"][:20],
            "first_shape_mismatch": result["shape_mismatch"][:10],
        }

        if best_result is None or result["matched_count"] > best_result["matched_count"]:
            best_name = name
            best_result = result

    suffix_state = suffix_match_state(model_state, raw_state)
    suffix_result = score_exact_variant(model_state, suffix_state)
    variant_reports["suffix_shape_match"] = {
        "matched_count": suffix_result["matched_count"],
        "missing_count": len(suffix_result["missing"]),
        "unexpected_count": len(suffix_result["unexpected"]),
        "shape_mismatch_count": len(suffix_result["shape_mismatch"]),
        "first_missing": suffix_result["missing"][:20],
        "first_unexpected": suffix_result["unexpected"][:20],
        "first_shape_mismatch": suffix_result["shape_mismatch"][:10],
    }

    if best_result is None or suffix_result["matched_count"] > best_result["matched_count"]:
        best_name = "suffix_shape_match"
        best_result = suffix_result

    assert best_result is not None
    match_ratio = best_result["matched_count"] / max(len(model_state), 1)

    print("\n[alignment candidates]")
    print(json.dumps(variant_reports, indent=2, default=str))
    print("\nBEST:", best_name, "match_ratio:", match_ratio)

    report = {
        "checkpoint": str(ckpt_path),
        "out_checkpoint": args.out_checkpoint,
        "constructor_kwargs": kwargs,
        "model_state_key_count": len(model_state),
        "checkpoint_state_key_count": len(raw_state),
        "best_variant": best_name,
        "match_ratio": match_ratio,
        "variants": variant_reports,
        "passed": match_ratio >= args.min_match_ratio,
    }

    save_json(args.report, report)

    if match_ratio < args.min_match_ratio:
        print("\nFAILED ALIGNMENT")
        print(f"match_ratio={match_ratio:.4f}, required={args.min_match_ratio:.4f}")
        print(f"Report written to: {args.report}")
        raise SystemExit(2)

    aligned_state = model.state_dict()
    aligned_state.update(best_result["matched"])

    missing, unexpected = model.load_state_dict(aligned_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Strict load after alignment failed: missing={missing}, unexpected={unexpected}")

    out_checkpoint = Path(args.out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    aligned_ckpt = {
        "model_state": model.state_dict(),
        "config": kwargs,
        "source_checkpoint": str(ckpt_path),
        "alignment_report": report,
    }

    torch.save(aligned_ckpt, out_checkpoint)

    print("\nALIGNED CHECKPOINT SAVED:")
    print(out_checkpoint)
    print("Strict load verified: missing=0 unexpected=0")


if __name__ == "__main__":
    main()
