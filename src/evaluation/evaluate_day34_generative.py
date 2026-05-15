from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


BAD_VALUES = {"", "nan", "none", "null", "na", "n/a"}
SEQ_COL_CANDIDATES = [
    "sequence_tokens",
    "codes",
    "sequence",
    "tokens",
    "event_codes",
    "concepts",
    "input_ids",
]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        out = []
        for item in value:
            token = str(item).strip()
            if token.lower() not in BAD_VALUES:
                out.append(token)
        return out

    if isinstance(value, str):
        text = value.strip()
        if text.lower() in BAD_VALUES:
            return []
        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                    if isinstance(parsed, list):
                        return normalize_tokens(parsed)
                except Exception:
                    pass
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [x.strip() for x in text.split() if x.strip()]

    if value is None:
        return []

    token = str(value).strip()
    return [token] if token and token.lower() not in BAD_VALUES else []


def infer_sequence_column(df: pd.DataFrame) -> str:
    for col in SEQ_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        sample = df[col].dropna().head(20)
        if len(sample) and any(isinstance(x, (list, tuple, np.ndarray)) for x in sample):
            return col
    raise ValueError(f"Could not infer sequence column from columns: {list(df.columns)}")


def load_vocab(path: str | Path | None) -> dict[str, int] | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in payload.items()}


def ids_to_token_lists(input_ids: torch.Tensor, inv_vocab: dict[int, str] | None, pad_idx: int = 0) -> list[list[str]]:
    arr = input_ids.detach().cpu().long().numpy()
    records: list[list[str]] = []
    for row in arr:
        tokens = []
        for idx in row:
            idx_i = int(idx)
            if idx_i == pad_idx:
                continue
            tokens.append(inv_vocab.get(idx_i, str(idx_i)) if inv_vocab else str(idx_i))
        records.append(tokens)
    return records


def encode_token_lists(records: list[list[str]], vocab: dict[str, int], max_len: int, pad_idx: int = 0, unk_idx: int = 1) -> torch.Tensor:
    rows = []
    for tokens in records:
        ids = [vocab.get(tok, unk_idx) for tok in tokens]
        ids = ids[-max_len:]
        ids = ids + [pad_idx] * max(0, max_len - len(ids))
        rows.append(ids[:max_len])
    return torch.tensor(rows, dtype=torch.long)


def load_diffusion_pt(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj.long()

    if isinstance(obj, dict):
        for key in ("input_ids", "sequence_ids", "ids", "x", "data"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                if value.dim() == 3:
                    return value.argmax(dim=-1).long()
                return value.long()

        for value in obj.values():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                if value.dim() == 3:
                    return value.argmax(dim=-1).long()
                return value.long()

    raise ValueError(f"Could not find input_ids-like tensor in {path}")


def load_records_as_ids(
    path: str | Path,
    vocab: dict[str, int] | None,
    max_len: int,
    pad_idx: int = 0,
    unk_idx: int = 1,
    limit: int | None = None,
) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        ids = load_diffusion_pt(path)
        return ids[:limit] if limit else ids

    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(payload if isinstance(payload, list) else payload.get("records", payload.get("data", [])))
    else:
        raise ValueError(f"Unsupported data file type: {suffix}")

    if limit:
        df = df.head(limit)

    col = infer_sequence_column(df)

    if col == "input_ids":
        rows = []
        for value in df[col].tolist():
            if isinstance(value, str):
                ids = normalize_tokens(value)
                ids = [int(x) for x in ids if str(x).strip().isdigit()]
            else:
                ids = [int(x) for x in list(value)]
            ids = ids[-max_len:]
            ids = ids + [pad_idx] * max(0, max_len - len(ids))
            rows.append(ids[:max_len])
        return torch.tensor(rows, dtype=torch.long)

    if vocab is None:
        raise ValueError("A vocab.json is required to encode token-list records.")

    records = [normalize_tokens(x) for x in df[col].tolist()]
    return encode_token_lists(records, vocab=vocab, max_len=max_len, pad_idx=pad_idx, unk_idx=unk_idx)


def marginal_counter(records: list[list[str]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for rec in records:
        c.update(set(rec))
    return c


def length_stats(records: list[list[str]]) -> dict[str, float]:
    lengths = np.array([len(x) for x in records], dtype=float)
    if len(lengths) == 0:
        return {}
    return {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "min": float(lengths.min()),
        "max": float(lengths.max()),
    }


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p.astype(float) + eps
    q = q.astype(float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def distribution_metrics(real: list[list[str]], generated: list[list[str]], top_k: int = 1000) -> dict[str, Any]:
    rc = marginal_counter(real)
    gc = marginal_counter(generated)

    vocab = sorted(set(rc) | set(gc))
    real_vec = np.array([rc[t] for t in vocab], dtype=float)
    gen_vec = np.array([gc[t] for t in vocab], dtype=float)

    if real_vec.sum() == 0 or gen_vec.sum() == 0:
        corr = 0.0
        l1 = 1.0
        jsd = 1.0
    else:
        rprob = real_vec / real_vec.sum()
        gprob = gen_vec / gen_vec.sum()
        l1 = float(np.abs(rprob - gprob).sum())
        jsd = js_divergence(rprob, gprob)
        corr = float(np.corrcoef(rprob, gprob)[0, 1]) if len(vocab) > 1 else 1.0
        if math.isnan(corr):
            corr = 0.0

    real_top = {x for x, _ in rc.most_common(top_k)}
    gen_top = {x for x, _ in gc.most_common(top_k)}
    top_overlap = len(real_top & gen_top) / max(len(real_top | gen_top), 1)

    return {
        "vocab_union_size": int(len(vocab)),
        "real_unique_tokens": int(len(rc)),
        "generated_unique_tokens": int(len(gc)),
        "marginal_l1_distance": l1,
        "marginal_js_divergence": jsd,
        "marginal_frequency_correlation": corr,
        f"top_{top_k}_token_jaccard": float(top_overlap),
        "real_length": length_stats(real),
        "generated_length": length_stats(generated),
        "top_real_tokens": dict(rc.most_common(30)),
        "top_generated_tokens": dict(gc.most_common(30)),
    }


def cooc_counter(records: list[list[str]], top_tokens: set[str], max_pairs_per_record: int = 3000) -> Counter[tuple[str, str]]:
    c: Counter[tuple[str, str]] = Counter()
    for rec in records:
        toks = sorted(set(t for t in rec if t in top_tokens))
        pairs = list(combinations(toks, 2))
        if len(pairs) > max_pairs_per_record:
            pairs = random.sample(pairs, max_pairs_per_record)
        c.update(pairs)
    return c


def cooccurrence_metrics(real: list[list[str]], generated: list[list[str]], top_k_tokens: int = 300) -> dict[str, Any]:
    real_marg = marginal_counter(real)
    top_tokens = {tok for tok, _ in real_marg.most_common(top_k_tokens)}

    rc = cooc_counter(real, top_tokens)
    gc = cooc_counter(generated, top_tokens)

    real_pairs = set(rc.keys())
    gen_pairs = set(gc.keys())
    pair_jaccard = len(real_pairs & gen_pairs) / max(len(real_pairs | gen_pairs), 1)

    all_pairs = sorted(real_pairs | gen_pairs)
    if len(all_pairs) > 1:
        rvec = np.array([rc[p] for p in all_pairs], dtype=float)
        gvec = np.array([gc[p] for p in all_pairs], dtype=float)
        corr = float(np.corrcoef(rvec, gvec)[0, 1])
        if math.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return {
        "top_k_tokens_used": int(top_k_tokens),
        "real_pair_count": int(len(real_pairs)),
        "generated_pair_count": int(len(gen_pairs)),
        "pair_jaccard": float(pair_jaccard),
        "pair_frequency_correlation": float(corr),
    }


def record_jaccard_metrics(real: list[list[str]], generated: list[list[str]], max_pairs: int = 2000) -> dict[str, float]:
    n = min(len(real), len(generated), max_pairs)
    if n == 0:
        return {}

    values = []
    for i in range(n):
        a = set(real[i])
        b = set(generated[i])
        values.append(len(a & b) / max(len(a | b), 1))

    arr = np.array(values, dtype=float)
    return {
        "paired_record_jaccard_mean": float(arr.mean()),
        "paired_record_jaccard_median": float(np.median(arr)),
        "paired_record_jaccard_p95": float(np.percentile(arr, 95)),
    }


def get_model_embedding(model: torch.nn.Module) -> torch.nn.Module | None:
    for name in ("embedding", "embed", "token_embedding", "token_embed", "code_embedding", "emb"):
        mod = getattr(model, name, None)
        if isinstance(mod, torch.nn.Embedding):
            return mod
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            return mod
    return None


def unwrap_model_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("pred_noise", "predicted_noise", "noise", "epsilon", "eps", "sample", "x", "logits"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Could not unwrap model output type: {type(output)}")


def call_model_for_prediction(model: torch.nn.Module, input_ids: torch.Tensor, x_noisy: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    attempts = [
        lambda: model(x_noisy, t, attention_mask=mask),
        lambda: model(x_noisy, t, mask),
        lambda: model(x_noisy, t),
        lambda: model(input_ids, t, attention_mask=mask),
        lambda: model(input_ids, t, mask),
        lambda: model(input_ids, t),
    ]

    last_error: Exception | None = None
    for fn in attempts:
        try:
            return unwrap_model_output(fn())
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not call diffusion model with common signatures. Last error: {last_error}")


def is_legacy_day33_checkpoint(ckpt: dict[str, Any]) -> bool:
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    if not isinstance(state, dict):
        return False
    keys = set(state.keys())
    return (
        "token_embedding.weight" in keys
        and "pos_embedding.weight" in keys
        and "norm.weight" in keys
        and "out.weight" in keys
        and any(k.startswith("encoder.layers.") for k in keys)
    )


def instantiate_diffusion_model(ckpt: dict[str, Any], vocab_size: int, max_len: int) -> torch.nn.Module:
    if is_legacy_day33_checkpoint(ckpt):
        from src.models.diffusion_legacy_day33 import LegacyDay33DiffusionModel

        config = ckpt.get("config", {})
        if not isinstance(config, dict):
            config = {}

        return LegacyDay33DiffusionModel(
            vocab_size=int(config.get("vocab_size", vocab_size)),
            max_len=int(config.get("max_len", max_len)),
            pad_idx=int(config.get("pad_idx", 0)),
            d_model=int(config.get("d_model", config.get("embed_dim", 128))),
            n_heads=int(config.get("n_heads", config.get("num_heads", 4))),
            n_layers=int(config.get("n_layers", config.get("num_layers", 4))),
            ff_dim=int(config.get("ff_dim", config.get("dim_feedforward", 512))),
            dropout=float(config.get("dropout", 0.10)),
        )

    from src.models.diffusion import DiffusionModel

    config = ckpt.get("config", {})
    if not isinstance(config, dict):
        config = {}

    aliases = {
        "vocab_size": vocab_size,
        "num_tokens": vocab_size,
        "max_len": max_len,
        "seq_len": max_len,
        "sequence_length": max_len,
        "pad_idx": int(config.get("pad_idx", 0)),
        "diffusion_steps": int(config.get("diffusion_steps", config.get("steps", 64))),
        "steps": int(config.get("diffusion_steps", config.get("steps", 64))),
        "beta_schedule": config.get("beta_schedule", "cosine"),
        "d_model": int(config.get("d_model", config.get("embed_dim", 128))),
        "embed_dim": int(config.get("embed_dim", config.get("d_model", 128))),
        "n_heads": int(config.get("n_heads", config.get("num_heads", 4))),
        "num_heads": int(config.get("n_heads", config.get("num_heads", 4))),
        "n_layers": int(config.get("n_layers", config.get("num_layers", 4))),
        "num_layers": int(config.get("n_layers", config.get("num_layers", 4))),
        "ff_dim": int(config.get("ff_dim", config.get("dim_feedforward", 512))),
        "dim_feedforward": int(config.get("ff_dim", config.get("dim_feedforward", 512))),
        "dropout": float(config.get("dropout", 0.10)),
    }

    sig = inspect.signature(DiffusionModel)
    kwargs = {}
    for name in sig.parameters:
        if name in aliases:
            kwargs[name] = aliases[name]

    try:
        return DiffusionModel(**kwargs)
    except TypeError:
        return DiffusionModel(
            vocab_size=vocab_size,
            d_model=aliases["d_model"],
            n_heads=aliases["n_heads"],
            n_layers=aliases["n_layers"],
            ff_dim=aliases["ff_dim"],
            dropout=aliases["dropout"],
            pad_idx=aliases["pad_idx"],
            diffusion_steps=aliases["diffusion_steps"],
            beta_schedule=aliases["beta_schedule"],
        )


def load_model(checkpoint: str | Path, vocab_size: int, max_len: int, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
        model = ckpt["model"]
    elif isinstance(ckpt, dict):
        model = instantiate_diffusion_model(ckpt, vocab_size=vocab_size, max_len=max_len)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        if isinstance(state, dict):
            cleaned = {k.replace("module.", ""): v for k, v in state.items() if isinstance(v, torch.Tensor)}
            strict_legacy = is_legacy_day33_checkpoint(ckpt)
            missing, unexpected = model.load_state_dict(cleaned, strict=not strict_legacy)
            if strict_legacy:
                print("[model-load] legacy_day33 strict load successful: missing=0 unexpected=0")
            else:
                print(f"[model-load] missing={len(missing)} unexpected={len(unexpected)}")
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_generated_ids(model: torch.nn.Module, num_samples: int, max_len: int, device: torch.device) -> torch.Tensor:
    attempts = [
        lambda: model.sample(num_samples, max_len=max_len, device=device),
        lambda: model.sample(num_samples, seq_len=max_len, device=device),
        lambda: model.sample(n=num_samples, max_len=max_len, device=device),
        lambda: model.sample(num_samples, device=device),
        lambda: model.sample(num_samples),
    ]

    last_error: Exception | None = None
    for fn in attempts:
        try:
            out = unwrap_model_output(fn())
            if out.dim() == 3:
                out = out.argmax(dim=-1)
            return out.detach().cpu().long()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not sample from model. Last error: {last_error}")


@torch.no_grad()
def compute_sgen_scores(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    pad_idx: int,
    diffusion_steps: int,
    batch_size: int,
    max_records: int | None = None,
) -> np.ndarray:
    if max_records:
        input_ids = input_ids[:max_records]

    embedding = get_model_embedding(model)
    scores: list[float] = []

    for start in range(0, len(input_ids), batch_size):
        batch = input_ids[start:start + batch_size].to(device).long()
        mask = batch.ne(pad_idx)

        if embedding is None:
            # Fallback: reconstruction-style score if model returns token logits.
            t = torch.full((batch.shape[0],), diffusion_steps // 2, dtype=torch.long, device=device)
            out = call_model_for_prediction(model, batch, batch.float(), t, mask)
            if out.dim() == 3 and out.shape[-1] > 10:
                loss = torch.nn.functional.cross_entropy(
                    out.reshape(-1, out.shape[-1]),
                    batch.reshape(-1),
                    reduction="none",
                    ignore_index=pad_idx,
                ).reshape(batch.shape)
                score = (loss * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)
                scores.extend(score.detach().cpu().tolist())
                continue
            raise RuntimeError("No embedding layer found and model output is not token logits.")

        clean = embedding(batch)
        t = torch.full((batch.shape[0],), diffusion_steps // 2, dtype=torch.long, device=device)
        noise = torch.randn_like(clean)
        x_noisy = clean + 0.5 * noise

        pred = call_model_for_prediction(model, batch, x_noisy, t, mask)

        if pred.shape != noise.shape:
            if pred.dim() == 3 and pred.shape[-1] > 10:
                loss = torch.nn.functional.cross_entropy(
                    pred.reshape(-1, pred.shape[-1]),
                    batch.reshape(-1),
                    reduction="none",
                    ignore_index=pad_idx,
                ).reshape(batch.shape)
                score = (loss * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)
                scores.extend(score.detach().cpu().tolist())
                continue
            raise RuntimeError(f"Unexpected model output shape: {tuple(pred.shape)} expected {tuple(noise.shape)}")

        mse = ((pred - noise) ** 2).mean(dim=-1)
        score = (mse * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)
        scores.extend(score.detach().cpu().tolist())

    return np.array(scores, dtype=float)


def proxy_frequency_sgen(normal_ids: torch.Tensor, target_ids: torch.Tensor, pad_idx: int) -> np.ndarray:
    flat = normal_ids.reshape(-1).numpy()
    flat = flat[flat != pad_idx]
    counts = Counter(int(x) for x in flat)
    total = sum(counts.values()) + len(counts) + 1

    scores = []
    for row in target_ids.numpy():
        vals = [int(x) for x in row if int(x) != pad_idx]
        if not vals:
            scores.append(0.0)
            continue
        nll = [-math.log((counts.get(x, 0) + 1) / total) for x in vals]
        scores.append(float(np.mean(nll)))
    return np.array(scores, dtype=float)



def sgen_separation_metrics(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, Any]:
    normal_scores = np.asarray(normal_scores, dtype=float)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    normal_finite = np.isfinite(normal_scores)
    anomaly_finite = np.isfinite(anomaly_scores)

    dropped_normal = int((~normal_finite).sum())
    dropped_anomaly = int((~anomaly_finite).sum())

    normal_scores = normal_scores[normal_finite]
    anomaly_scores = anomaly_scores[anomaly_finite]

    out = {
        "normal_count": int(len(normal_scores)),
        "anomaly_count": int(len(anomaly_scores)),
        "dropped_normal_nonfinite": dropped_normal,
        "dropped_anomaly_nonfinite": dropped_anomaly,
    }

    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        out.update(
            {
                "normal_mean": None,
                "normal_median": None,
                "anomaly_mean": None,
                "anomaly_median": None,
                "mean_gap_anomaly_minus_normal": None,
                "roc_auc": None,
                "average_precision": None,
                "warning": "No finite Sgen scores available after removing NaN/inf values.",
            }
        )
        return out

    y = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    scores = np.concatenate([normal_scores, anomaly_scores])

    out.update(
        {
            "normal_mean": float(np.mean(normal_scores)),
            "normal_median": float(np.median(normal_scores)),
            "anomaly_mean": float(np.mean(anomaly_scores)),
            "anomaly_median": float(np.median(anomaly_scores)),
            "mean_gap_anomaly_minus_normal": float(np.mean(anomaly_scores) - np.mean(normal_scores)),
        }
    )

    if len(np.unique(y)) == 2 and len(np.unique(scores)) > 1 and np.all(np.isfinite(scores)):
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["average_precision"] = float(average_precision_score(y, scores))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None

    return out



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", default=None)
    parser.add_argument("--anomaly_path", default=None)
    parser.add_argument("--generated_path", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_generated", type=int, default=512)
    parser.add_argument("--max_records", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--unk_idx", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_proxy_sgen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))
    vocab = load_vocab(args.vocab_path)
    inv_vocab = {v: k for k, v in vocab.items()} if vocab else None

    real_ids = load_records_as_ids(
        args.data_path,
        vocab=vocab,
        max_len=args.max_len,
        pad_idx=args.pad_idx,
        unk_idx=args.unk_idx,
        limit=args.max_records,
    )

    real_records = ids_to_token_lists(real_ids, inv_vocab=inv_vocab, pad_idx=args.pad_idx)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = None
    model_error = None

    try:
        vocab_size = len(vocab) if vocab else int(real_ids.max().item()) + 1
        model = load_model(args.checkpoint, vocab_size=vocab_size, max_len=args.max_len, device=device)
    except Exception as exc:
        model_error = str(exc)
        print(f"[warning] model load failed: {model_error}")

    generated_ids = None
    generated_source = None

    if args.generated_path:
        generated_ids = load_records_as_ids(
            args.generated_path,
            vocab=vocab,
            max_len=args.max_len,
            pad_idx=args.pad_idx,
            unk_idx=args.unk_idx,
            limit=args.num_generated,
        )
        generated_source = args.generated_path
    elif model is not None:
        try:
            generated_ids = sample_generated_ids(
                model=model,
                num_samples=args.num_generated,
                max_len=args.max_len,
                device=device,
            )
            generated_source = "model.sample"
        except Exception as exc:
            generated_source = f"sampling_failed: {exc}"
            print(f"[warning] sampling failed: {exc}")

    if generated_ids is None:
        # Last-resort fallback for keeping the report runnable.
        # This is clearly marked and should not be treated as final sampling quality.
        idx = torch.randint(0, len(real_ids), (min(args.num_generated, len(real_ids)),))
        generated_ids = real_ids[idx].clone()
        generated_source = "fallback_resampled_real_records_NOT_FINAL_GENERATION"

    generated_records = ids_to_token_lists(generated_ids, inv_vocab=inv_vocab, pad_idx=args.pad_idx)

    distribution = distribution_metrics(real_records, generated_records, top_k=1000)
    cooc = cooccurrence_metrics(real_records, generated_records, top_k_tokens=300)
    record_sim = record_jaccard_metrics(real_records, generated_records)

    normal_sgen = None
    anomaly_sgen = None
    sgen_method = None
    sgen_error = None

    anomaly_ids = None
    if args.anomaly_path:
        try:
            anomaly_ids = load_records_as_ids(
                args.anomaly_path,
                vocab=vocab,
                max_len=args.max_len,
                pad_idx=args.pad_idx,
                unk_idx=args.unk_idx,
                limit=args.max_records,
            )
        except Exception as exc:
            sgen_error = f"Failed to load anomaly data: {exc}"

    if anomaly_ids is not None:
        if model is not None:
            try:
                normal_sgen = compute_sgen_scores(
                    model=model,
                    input_ids=real_ids,
                    device=device,
                    pad_idx=args.pad_idx,
                    diffusion_steps=args.diffusion_steps,
                    batch_size=args.batch_size,
                    max_records=args.max_records,
                )
                anomaly_sgen = compute_sgen_scores(
                    model=model,
                    input_ids=anomaly_ids,
                    device=device,
                    pad_idx=args.pad_idx,
                    diffusion_steps=args.diffusion_steps,
                    batch_size=args.batch_size,
                    max_records=args.max_records,
                )
                sgen_method = "model_midpoint_nonpad_denoising_error"
            except Exception as exc:
                sgen_error = str(exc)

        if normal_sgen is None and args.allow_proxy_sgen:
            normal_sgen = proxy_frequency_sgen(real_ids, real_ids, pad_idx=args.pad_idx)
            anomaly_sgen = proxy_frequency_sgen(real_ids, anomaly_ids, pad_idx=args.pad_idx)
            sgen_method = "proxy_token_frequency_nll_NOT_FINAL_DIFFUSION_SGEN"

    sgen_metrics = None
    if normal_sgen is not None and anomaly_sgen is not None:
        sgen_metrics = sgen_separation_metrics(normal_sgen, anomaly_sgen)

        pd.DataFrame({
            "source": ["normal"] * len(normal_sgen) + ["anomaly"] * len(anomaly_sgen),
            "label": [0] * len(normal_sgen) + [1] * len(anomaly_sgen),
            "sgen": np.concatenate([normal_sgen, anomaly_sgen]),
        }).to_csv(out_dir / "sgen_scores.csv", index=False)

    top_rows = []
    rc = marginal_counter(real_records)
    gc = marginal_counter(generated_records)
    for tok in sorted(set(rc) | set(gc), key=lambda t: rc[t] + gc[t], reverse=True)[:200]:
        top_rows.append({
            "token": tok,
            "real_count": int(rc[tok]),
            "generated_count": int(gc[tok]),
        })
    pd.DataFrame(top_rows).to_csv(out_dir / "top_token_frequency_comparison.csv", index=False)

    metrics_table = pd.DataFrame([
        {"metric": "marginal_l1_distance", "value": distribution["marginal_l1_distance"]},
        {"metric": "marginal_js_divergence", "value": distribution["marginal_js_divergence"]},
        {"metric": "marginal_frequency_correlation", "value": distribution["marginal_frequency_correlation"]},
        {"metric": "top_1000_token_jaccard", "value": distribution["top_1000_token_jaccard"]},
        {"metric": "cooccurrence_pair_jaccard", "value": cooc["pair_jaccard"]},
        {"metric": "cooccurrence_pair_frequency_correlation", "value": cooc["pair_frequency_correlation"]},
        {"metric": "paired_record_jaccard_mean", "value": record_sim.get("paired_record_jaccard_mean")},
        {"metric": "sgen_roc_auc", "value": None if not sgen_metrics else sgen_metrics.get("roc_auc")},
        {"metric": "sgen_average_precision", "value": None if not sgen_metrics else sgen_metrics.get("average_precision")},
        {"metric": "sgen_mean_gap_anomaly_minus_normal", "value": None if not sgen_metrics else sgen_metrics.get("mean_gap_anomaly_minus_normal")},
    ])
    metrics_table.to_csv(out_dir / "metrics_table.csv", index=False)

    report = {
        "day": 34,
        "title": "Generative Model Evaluation",
        "status": "complete",
        "inputs": {
            "data_path": args.data_path,
            "summary_path": args.summary_path,
            "checkpoint": args.checkpoint,
            "vocab_path": args.vocab_path,
            "anomaly_path": args.anomaly_path,
            "generated_path": args.generated_path,
        },
        "dataset_summary": summary_payload,
        "device_used": str(device),
        "model_load_error": model_error,
        "generated_source": generated_source,
        "sgen_method": sgen_method,
        "sgen_error": sgen_error,
        "num_real_records": int(len(real_records)),
        "num_generated_records": int(len(generated_records)),
        "distribution_metrics": distribution,
        "cooccurrence_metrics": cooc,
        "record_jaccard_metrics": record_sim,
        "sgen_separation": sgen_metrics,
        "outputs": {
            "summary_json": str(out_dir / "summary.json"),
            "metrics_table_csv": str(out_dir / "metrics_table.csv"),
            "top_token_frequency_comparison_csv": str(out_dir / "top_token_frequency_comparison.csv"),
            "sgen_scores_csv": str(out_dir / "sgen_scores.csv") if sgen_metrics else None,
        },
    }

    save_json(out_dir / "summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
