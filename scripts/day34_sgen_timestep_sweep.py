from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.evaluate_day34_generative import (
    call_model_for_prediction,
    get_model_embedding,
    load_model,
    load_records_as_ids,
    load_vocab,
)


def cosine_alpha_cumprod(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:].clamp(min=1e-5, max=0.9999)


@torch.no_grad()
def compute_scores_at_t(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    t_value: int,
    alpha_cumprod: torch.Tensor,
    device: torch.device,
    batch_size: int,
    pad_idx: int,
    repeats: int,
) -> np.ndarray:
    embedding = get_model_embedding(model)
    if embedding is None:
        raise RuntimeError("No embedding layer found.")

    all_repeat_scores = []

    for _ in range(repeats):
        scores = []

        for start in range(0, len(input_ids), batch_size):
            batch = input_ids[start:start + batch_size].to(device).long()
            mask = batch.ne(pad_idx)

            clean = embedding(batch)

            t = torch.full((batch.shape[0],), t_value, dtype=torch.long, device=device)
            noise = torch.randn_like(clean)

            ac = alpha_cumprod[t_value].to(device)
            x_noisy = torch.sqrt(ac) * clean + torch.sqrt(1.0 - ac) * noise

            pred = call_model_for_prediction(model, batch, x_noisy, t, mask)

            if pred.shape != noise.shape:
                raise RuntimeError(
                    f"Unexpected pred shape {tuple(pred.shape)} vs noise {tuple(noise.shape)}"
                )

            mse = ((pred - noise) ** 2).mean(dim=-1)
            score = (mse * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)
            scores.extend(score.detach().cpu().numpy().tolist())

        all_repeat_scores.append(np.asarray(scores, dtype=float))

    stacked = np.stack(all_repeat_scores, axis=0)
    return np.nanmean(stacked, axis=0)


def finite_metrics(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, Any]:
    normal_scores = np.asarray(normal_scores, dtype=float)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    normal_scores = normal_scores[np.isfinite(normal_scores)]
    anomaly_scores = anomaly_scores[np.isfinite(anomaly_scores)]

    y = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    s = np.concatenate([normal_scores, anomaly_scores])

    row: dict[str, Any] = {
        "normal_count": int(len(normal_scores)),
        "anomaly_count": int(len(anomaly_scores)),
        "normal_mean": float(np.mean(normal_scores)),
        "normal_median": float(np.median(normal_scores)),
        "anomaly_mean": float(np.mean(anomaly_scores)),
        "anomaly_median": float(np.median(anomaly_scores)),
        "mean_gap_anomaly_minus_normal": float(np.mean(anomaly_scores) - np.mean(normal_scores)),
    }

    if len(np.unique(s)) > 1:
        row["roc_auc"] = float(roc_auc_score(y, s))
        row["average_precision"] = float(average_precision_score(y, s))
        row["roc_auc_if_reversed"] = float(roc_auc_score(y, -s))
        row["average_precision_if_reversed"] = float(average_precision_score(y, -s))
    else:
        row["roc_auc"] = None
        row["average_precision"] = None
        row["roc_auc_if_reversed"] = None
        row["average_precision_if_reversed"] = None

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--anomaly_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--unk_idx", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.vocab_path)
    assert vocab is not None

    normal_ids = load_records_as_ids(
        args.data_path,
        vocab=vocab,
        max_len=args.max_len,
        pad_idx=args.pad_idx,
        unk_idx=args.unk_idx,
        limit=args.max_records,
    )

    anomaly_ids = load_records_as_ids(
        args.anomaly_path,
        vocab=vocab,
        max_len=args.max_len,
        pad_idx=args.pad_idx,
        unk_idx=args.unk_idx,
        limit=args.max_records,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(
        args.checkpoint,
        vocab_size=len(vocab),
        max_len=args.max_len,
        device=device,
    )

    alpha_cumprod = cosine_alpha_cumprod(args.diffusion_steps)

    # Avoid t=0 only; evaluate low/mid/high noise.
    t_values = sorted(set([
        1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, args.diffusion_steps - 1
    ]))
    t_values = [t for t in t_values if 0 <= t < args.diffusion_steps]

    rows = []
    best = None

    for t_value in t_values:
        print(f"[sweep] t={t_value}")

        normal_scores = compute_scores_at_t(
            model=model,
            input_ids=normal_ids,
            t_value=t_value,
            alpha_cumprod=alpha_cumprod,
            device=device,
            batch_size=args.batch_size,
            pad_idx=args.pad_idx,
            repeats=args.repeats,
        )

        anomaly_scores = compute_scores_at_t(
            model=model,
            input_ids=anomaly_ids,
            t_value=t_value,
            alpha_cumprod=alpha_cumprod,
            device=device,
            batch_size=args.batch_size,
            pad_idx=args.pad_idx,
            repeats=args.repeats,
        )

        row = {"timestep": t_value, **finite_metrics(normal_scores, anomaly_scores)}
        rows.append(row)

        if best is None or (row.get("roc_auc") or 0.0) > (best.get("roc_auc") or 0.0):
            best = row

        pd.DataFrame({
            "source": ["normal"] * len(normal_scores) + ["anomaly"] * len(anomaly_scores),
            "label": [0] * len(normal_scores) + [1] * len(anomaly_scores),
            "timestep": [t_value] * (len(normal_scores) + len(anomaly_scores)),
            "sgen": np.concatenate([normal_scores, anomaly_scores]),
        }).to_csv(out_dir / f"sgen_scores_t{t_value}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sgen_timestep_sweep.csv", index=False)

    report = {
        "status": "complete",
        "checkpoint": args.checkpoint,
        "max_records": args.max_records,
        "repeats": args.repeats,
        "best_by_roc_auc": best,
        "all_rows": rows,
        "interpretation": "If all ROC-AUC values remain around 0.5, the issue is not checkpoint loading; the current diffusion Sgen proxy does not separate injected anomalies.",
    }

    (out_dir / "sgen_timestep_sweep_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
