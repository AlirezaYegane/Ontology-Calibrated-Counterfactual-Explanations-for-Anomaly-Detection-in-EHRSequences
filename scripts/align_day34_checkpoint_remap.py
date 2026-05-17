from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from src.models.diffusion import DiffusionModel


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def get_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ("model_state", "model_state_dict", "state_dict"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items() if isinstance(v, torch.Tensor)}

    tensor_items = {str(k): v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if tensor_items:
        return tensor_items

    raise ValueError("No tensor state_dict found.")


def remap_key(k: str) -> str:
    # Remove common wrappers
    for prefix in ("module.", "_orig_mod.", "model.", "diffusion.", "net."):
        if k.startswith(prefix):
            k = k[len(prefix):]

    # Day 33 checkpoint naming -> current DiffusionModel naming
    if k == "pos_embedding.weight":
        return "position_embedding.weight"

    if k.startswith("encoder.layers."):
        return k.replace("encoder.layers.", "denoiser.layers.", 1)

    if k.startswith("time_embedding.1."):
        return k.replace("time_embedding.1.", "time_embedding.proj.0.", 1)

    if k.startswith("time_embedding.3."):
        return k.replace("time_embedding.3.", "time_embedding.proj.2.", 1)

    return k


def build_model_kwargs(vocab_path: str, summary_path: str, ckpt: Any) -> dict[str, Any]:
    vocab = load_json(vocab_path)
    summary = load_json(summary_path)

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    kwargs_all = {
        "vocab_size": int(cfg.get("vocab_size", summary.get("vocab_size", len(vocab)))),
        "max_len": int(cfg.get("max_len", summary.get("max_len", 256))),
        "pad_idx": int(cfg.get("pad_idx", 0)),
        "d_model": int(cfg.get("d_model", cfg.get("embed_dim", 128))),
        "n_heads": int(cfg.get("n_heads", cfg.get("num_heads", 4))),
        "n_layers": int(cfg.get("n_layers", cfg.get("num_layers", 4))),
        "ff_dim": int(cfg.get("ff_dim", cfg.get("dim_feedforward", 512))),
        "dropout": float(cfg.get("dropout", 0.10)),
    }

    sig = inspect.signature(DiffusionModel)
    return {k: v for k, v in kwargs_all.items() if k in sig.parameters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--out_checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    raw_state = get_state_dict(ckpt)

    kwargs = build_model_kwargs(args.vocab_path, args.summary_path, ckpt)
    model = DiffusionModel(**kwargs)
    model_state = model.state_dict()

    aligned_updates: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []

    for old_key, tensor in raw_state.items():
        new_key = remap_key(old_key)

        if new_key not in model_state:
            unexpected.append(old_key)
            continue

        if tuple(model_state[new_key].shape) != tuple(tensor.shape):
            shape_mismatch.append(
                {
                    "old_key": old_key,
                    "new_key": new_key,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(model_state[new_key].shape),
                }
            )
            continue

        aligned_updates[new_key] = tensor

    buffer_keys = {
        "betas",
        "alphas",
        "alpha_cumprod",
        "sqrt_alpha_cumprod",
        "sqrt_one_minus_alpha_cumprod",
        "sqrt_recip_alphas",
    }

    trainable_model_keys = {
        k for k, v in model.named_parameters()
    }

    missing_trainable = sorted(k for k in trainable_model_keys if k not in aligned_updates)
    missing_buffers = sorted(k for k in buffer_keys if k in model_state and k not in aligned_updates)

    trainable_match_ratio = (
        (len(trainable_model_keys) - len(missing_trainable)) / max(len(trainable_model_keys), 1)
    )

    # Keep constructor-generated buffers, update only checkpoint trainable tensors.
    new_state = model.state_dict()
    new_state.update(aligned_updates)

    model.load_state_dict(new_state, strict=True)

    report = {
        "source_checkpoint": args.checkpoint,
        "out_checkpoint": args.out_checkpoint,
        "constructor_kwargs": kwargs,
        "checkpoint_state_keys": len(raw_state),
        "model_state_keys": len(model_state),
        "model_trainable_keys": len(trainable_model_keys),
        "matched_keys": len(aligned_updates),
        "unexpected_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatch),
        "missing_trainable_count": len(missing_trainable),
        "missing_buffer_count": len(missing_buffers),
        "trainable_match_ratio": trainable_match_ratio,
        "missing_trainable": missing_trainable,
        "missing_buffers_reinitialized_from_constructor": missing_buffers,
        "unexpected_first_30": unexpected[:30],
        "shape_mismatch_first_30": shape_mismatch[:30],
        "passed": (
            trainable_match_ratio == 1.0
            and len(shape_mismatch) == 0
            and len(missing_trainable) == 0
        ),
        "note": "Diffusion schedule buffers are re-created by the current constructor. Trainable parameters must match exactly.",
    }

    save_json(args.report, report)

    if not report["passed"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": kwargs,
            "source_checkpoint": args.checkpoint,
            "alignment_report": report,
        },
        out_path,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nALIGNED CHECKPOINT SAVED:", out_path)
    print("Trainable alignment: 100%")
    print("Strict load verified with constructor-generated buffers.")


if __name__ == "__main__":
    main()
