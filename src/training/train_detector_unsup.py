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
