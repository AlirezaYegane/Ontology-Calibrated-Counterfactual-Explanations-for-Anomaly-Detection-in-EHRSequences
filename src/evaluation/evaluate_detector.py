"""
src/evaluation/evaluate_detector.py
====================================
Day 19 -- Evaluate the trained GRU anomaly detector.

Loads a trained checkpoint, builds a labeled anomaly test set from the
test split, computes anomaly scores, and reports ROC AUC, Average
Precision, and P/R/F1 at the 80th-percentile threshold.

CLI::

    python -m src.evaluation.evaluate_detector --config config/detector.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import yaml

from src.models.detector import AnomalyDetectorGRU
from src.preprocessing.anomaly_injection import build_anomaly_test_set
from src.training.train_detector import collate_fn
from src.utils.vocab import BOS, EOS, PAD, encode_sequence, load_vocab

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _encode_codes(
    codes: list[str],
    vocab: dict[str, int],
    max_len: int,
) -> list[int]:
    """Encode a code list with BOS/EOS, matching training format."""
    encoded = encode_sequence(codes, vocab, max_len=max_len - 2)
    return [BOS] + encoded + [EOS]


def score_dataset(
    model: AnomalyDetectorGRU,
    codes_list: list[list[str]],
    vocab: dict[str, int],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Compute anomaly scores for a list of code sequences.

    Returns a 1-D numpy array of scores aligned with *codes_list*.
    """
    model.eval()
    all_scores: list[float] = []

    for start in range(0, len(codes_list), batch_size):
        batch_codes = codes_list[start : start + batch_size]
        encoded = [
            torch.tensor(_encode_codes(c, vocab, max_len), dtype=torch.long)
            for c in batch_codes
        ]
        padded = collate_fn(encoded).to(device)
        scores = model.anomaly_score(padded, pad_idx=PAD)
        all_scores.extend(scores.cpu().tolist())

    return np.array(all_scores)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Compute evaluation metrics.

    Parameters
    ----------
    labels:
        Binary labels (0=normal, 1=anomalous).
    scores:
        Anomaly scores (higher = more anomalous).

    Returns
    -------
    Dict with ROC AUC, AP, threshold, P/R/F1, and mean scores.
    """
    roc_auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))

    # Threshold at the 80th percentile of all scores
    threshold = float(np.percentile(scores, 80))
    preds = (scores >= threshold).astype(int)

    return {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap, 4),
        "threshold_p80": round(threshold, 6),
        "precision_at_p80": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall_at_p80": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1_at_p80": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "mean_score_normal": round(float(scores[labels == 0].mean()), 6),
        "mean_score_anomalous": round(float(scores[labels == 1].mean()), 6),
        "n_normal": int((labels == 0).sum()),
        "n_anomalous": int((labels == 1).sum()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 19 -- Evaluate trained GRU anomaly detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True, type=Path,
        help="Path to YAML config (same as training)",
    )
    p.add_argument(
        "--n-per-type", type=int, default=500,
        help="Number of records per anomaly type in the test set",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    d_cfg = cfg["data"]
    m_cfg = cfg["model"]
    o_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- load vocab ---
    vocab = load_vocab(Path(d_cfg["vocab_path"]))
    vocab_size = len(vocab)

    # --- load model ---
    ckpt_path = Path(o_cfg["checkpoint_dir"]) / "detector_best.pt"
    log.info("Loading checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = AnomalyDetectorGRU(
        vocab_size=vocab_size,
        embed_dim=m_cfg["embed_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        dropout=m_cfg["dropout"],
        pad_idx=PAD,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log.info("Model loaded (epoch %d, val_loss=%.4f)", checkpoint["epoch"], checkpoint["val_loss"])

    # --- build anomaly test set ---
    test_path = Path(d_cfg["test_pkl"])
    if test_path.suffix == ".parquet":
        test_df = pd.read_parquet(test_path)
    else:
        test_df = pd.read_pickle(test_path)

    code_col = d_cfg["code_col"]
    anomaly_df = build_anomaly_test_set(test_df, code_col=code_col, n_per_type=args.n_per_type)
    log.info("Anomaly test set: %d records", len(anomaly_df))

    # --- score ---
    codes_list = anomaly_df["codes"].tolist()
    max_len = cfg["training"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]

    scores = score_dataset(model, codes_list, vocab, max_len, batch_size, device)
    labels = anomaly_df["label"].values

    # --- metrics ---
    metrics = compute_metrics(labels, scores)

    # Per-type breakdown
    per_type: dict[str, dict[str, Any]] = {}
    for atype in anomaly_df["anomaly_type"].unique():
        mask = anomaly_df["anomaly_type"].values == atype
        per_type[atype] = {
            "count": int(mask.sum()),
            "mean_score": round(float(scores[mask].mean()), 6),
        }
    metrics["per_type"] = per_type

    log.info("ROC AUC: %.4f", metrics["roc_auc"])
    log.info("Average Precision: %.4f", metrics["average_precision"])
    log.info("F1 @ p80: %.4f", metrics["f1_at_p80"])
    log.info(
        "Mean score -- normal: %.4f, anomalous: %.4f",
        metrics["mean_score_normal"], metrics["mean_score_anomalous"],
    )

    # --- save ---
    log_dir = Path(o_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "eval_results.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
