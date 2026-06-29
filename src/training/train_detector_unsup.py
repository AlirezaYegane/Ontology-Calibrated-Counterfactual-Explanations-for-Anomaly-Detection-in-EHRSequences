"""
src/training/train_detector_unsup.py
=====================================
Phase 3 -- Training scaffold for the unsupervised next-token detector.

Trains ONLY on normal sequences (label==0 when labels exist), so synthetic
anomaly labels never leak into the detector. Provides a `run_training` function
usable from tests on a tiny in-memory dataset (smoke path), plus an argparse CLI.

This is a scaffold: it does NOT launch full-scale training unless invoked
explicitly with real data and epochs. No H200, no large runs here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.models.detector_unsup import (
    UnsupDetectorConfig,
    UnsupervisedSequenceDetector,
    build_vocab,
    set_seed,
)

log = logging.getLogger(__name__)


def filter_normal_sequences(
    rows: list[dict[str, Any]],
    seq_key: str = "codes",
    label_key: str = "is_synthetic_anomaly",
) -> list[list[str]]:
    """Keep only normal (label==0 / missing) sequences for unsupervised training."""
    out: list[list[str]] = []
    for r in rows:
        label = r.get(label_key, 0)
        try:
            is_anom = int(label) == 1
        except (TypeError, ValueError):
            is_anom = False
        if is_anom:
            continue
        seq = r.get(seq_key)
        if isinstance(seq, (list, tuple)) and len(seq) > 0:
            out.append([str(t) for t in seq])
    return out


def run_training(
    normal_sequences: list[list[str]],
    out_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 1,
    max_len: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train the unsupervised detector on normal sequences. Returns a summary."""
    if not normal_sequences:
        raise ValueError("No normal sequences provided for unsupervised training.")

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(normal_sequences)
    cfg = UnsupDetectorConfig(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_len=max_len,
    )
    det = UnsupervisedSequenceDetector(cfg, vocab, device=device)
    optimizer = torch.optim.Adam(det.model.parameters(), lr=lr)

    metrics_path = out_dir / "metrics.jsonl"
    n = len(normal_sequences)
    rows_logged = []
    with metrics_path.open("w", encoding="utf-8") as fh:
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                batch = normal_sequences[start : start + batch_size]
                loss = det.train_step(batch, optimizer)
                epoch_loss += loss
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            rec = {"epoch": epoch, "train_loss": round(avg, 6), "n_sequences": n}
            rows_logged.append(rec)
            fh.write(json.dumps(rec) + "\n")
            log.info("epoch %d train_loss %.4f", epoch, avg)

    det.save(out_dir)
    summary = {
        "detector": "unsupervised_next_token_gru",
        "trained_on": "normal_sequences_only",
        "n_normal_sequences": n,
        "vocab_size": len(vocab),
        "epochs": epochs,
        "seed": seed,
        "config": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "max_len": max_len,
        },
        "final_train_loss": rows_logged[-1]["train_loss"] if rows_logged else None,
        "checkpoint": str((out_dir / "detector_unsup.pt")),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _val_metrics(
    detector: UnsupervisedSequenceDetector,
    val_seqs: list[list[str]],
    val_labels: list[int],
    batch_size: int,
) -> dict[str, float]:
    """Validation metrics for model selection (val-only; never test)."""
    from src.evaluation.stats import average_precision, roc_auc

    scores = detector.anomaly_scores(val_seqs, batch_size=batch_size)
    normal_losses = [s for s, y in zip(scores, val_labels) if int(y) == 0]
    out = {
        "val_normal_mean_nll": float(sum(normal_losses) / max(len(normal_losses), 1)),
    }
    n_pos = sum(1 for y in val_labels if int(y) == 1)
    if n_pos and n_pos < len(val_labels):
        import numpy as np

        y = np.asarray([int(v) for v in val_labels])
        s = np.asarray(scores, dtype=float)
        out["val_roc_auc"] = float(roc_auc(y, s))
        out["val_average_precision"] = float(average_precision(y, s))
    else:
        out["val_roc_auc"] = float("nan")
        out["val_average_precision"] = float("nan")
    return out


def train_detector_full(
    train_normals: list[list[str]],
    val_seqs: list[list[str]],
    val_labels: list[int],
    *,
    out_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
    max_len: int = 256,
    device: str = "cpu",
    early_stopping_patience: int = 4,
    early_stopping_metric: str = "val_roc_auc",
    resume: bool = True,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full-scale unsupervised detector training with val-based early stopping +
    resume. Trains ONLY on normal sequences; model selection / thresholds use the
    validation split only (test is never touched here)."""
    if not train_normals:
        raise ValueError("No normal training sequences provided.")

    import random as _random

    set_seed(seed)
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    vocab_dir = out_dir / "vocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(train_normals)
    cfg = UnsupDetectorConfig(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len,
    )
    det = UnsupervisedSequenceDetector(cfg, vocab, device=device)
    optimizer = torch.optim.Adam(
        det.model.parameters(), lr=lr, weight_decay=weight_decay
    )

    start_epoch = 0
    best_metric = -float("inf")
    best_epoch = -1
    last_path = ckpt_dir / "last.pt"
    higher_is_better = early_stopping_metric != "val_normal_mean_nll"
    if not higher_is_better:
        best_metric = float("inf")

    if resume and last_path.exists():
        state = torch.load(last_path, map_location=det.device)
        det.model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        log.info("resumed from epoch %d (best=%s)", start_epoch, best_metric)

    metrics_path = out_dir / "train_metrics.jsonl"
    n = len(train_normals)
    rng = _random.Random(seed)
    history: list[dict[str, Any]] = []
    mode = "a" if (resume and metrics_path.exists()) else "w"

    def _is_better(cur: float, best: float) -> bool:
        return cur > best if higher_is_better else cur < best

    with metrics_path.open(mode, encoding="utf-8") as fh:
        for epoch in range(start_epoch, epochs):
            order = list(range(n))
            rng.shuffle(order)
            shuffled = [train_normals[i] for i in order]
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                batch = shuffled[start : start + batch_size]
                epoch_loss += det.train_step(batch, optimizer)
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)

            vm = _val_metrics(det, val_seqs, val_labels, batch_size)
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "n_train": n,
                **{k: (round(v, 6) if v == v else None) for k, v in vm.items()},
            }
            history.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            log.info("epoch %d train_loss %.4f %s", epoch, train_loss, vm)

            cur = vm.get(early_stopping_metric, float("nan"))
            improved = cur == cur and _is_better(cur, best_metric)
            if improved:
                best_metric, best_epoch = float(cur), epoch
                det.save(ckpt_dir)  # best checkpoint (model + vocab)
            torch.save(
                {
                    "model_state": det.model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                },
                last_path,
            )
            if best_epoch >= 0 and (epoch - best_epoch) >= early_stopping_patience:
                log.info(
                    "early stopping at epoch %d (best epoch %d)", epoch, best_epoch
                )
                break

    # vocab metadata (size only; the vocab.json itself lives under the ignored
    # checkpoints/ dir as it is MIMIC-derived).
    (vocab_dir / "vocab_meta.json").write_text(
        json.dumps({"vocab_size": len(vocab)}, indent=2), encoding="utf-8"
    )

    summary = {
        "detector": "unsupervised_next_token_gru",
        "trained_on": "normal_sequences_only",
        "n_train_normal": n,
        "n_val": len(val_seqs),
        "vocab_size": len(vocab),
        "epochs_planned": epochs,
        "epochs_run": (history[-1]["epoch"] + 1) if history else start_epoch,
        "best_epoch": best_epoch,
        "best_metric_name": early_stopping_metric,
        "best_metric_value": (
            None
            if best_metric in (float("inf"), -float("inf"))
            else round(best_metric, 6)
        ),
        "final_val_metrics": history[-1] if history else None,
        "seed": seed,
        "device": str(det.device),
        "weight_decay": weight_decay,
        "lr": lr,
        "best_checkpoint": str(ckpt_dir),
        "config_snapshot": config_snapshot or {},
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if config_snapshot is not None:
        (out_dir / "config.json").write_text(
            json.dumps(config_snapshot, indent=2), encoding="utf-8"
        )
    return summary


def _load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
    if limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3 unsupervised detector training (scaffold)."
    )
    ap.add_argument(
        "--data", required=True, help="Processed sequences file (.pkl/.parquet)."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Cap rows (smoke runs).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true", help="Tiny run for sanity only.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    rows = _load_rows(Path(args.data), args.limit or (200 if args.smoke else None))
    normal = filter_normal_sequences(rows)
    log.info("normal sequences for training: %d", len(normal))
    summary = run_training(
        normal,
        out_dir=args.out_dir,
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
