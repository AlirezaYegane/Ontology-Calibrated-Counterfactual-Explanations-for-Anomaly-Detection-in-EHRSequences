"""
src/evaluation/frequency_baseline.py
=====================================
Day 21 -- Frequency-based anomaly detection baseline.

Scores each sequence as ``mean(-log(freq))`` where ``freq`` is the
relative frequency of each token in the training set.  Rare tokens
receive high scores.

CLI::

    python -m src.evaluation.frequency_baseline \\
        --train-pkl data/processed/mimiciv_train_ont.pkl \\
        --test-pkl data/processed/mimiciv_test_ont.pkl \\
        --code-col codes_ont
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.preprocessing.anomaly_injection import build_anomaly_test_set

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frequency model
# ---------------------------------------------------------------------------


def build_frequency_table(
    df: pd.DataFrame,
    code_col: str = "codes_ont",
) -> dict[str, float]:
    """Build a token frequency table from a training DataFrame.

    Returns a dict mapping each token to its relative frequency.
    """
    counter: Counter[str] = Counter()
    total = 0

    for tokens in df[code_col]:
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if isinstance(tokens, list):
            counter.update(tokens)
            total += len(tokens)

    if total == 0:
        return {}

    freq_table = {tok: cnt / total for tok, cnt in counter.items()}
    log.info("Frequency table: %d unique tokens, %d total occurrences", len(freq_table), total)
    return freq_table


def score_sequence(
    codes: list[str],
    freq_table: dict[str, float],
    min_freq: float = 1e-7,
) -> float:
    """Compute anomaly score for a single sequence.

    Score = mean(-log(freq)) over all tokens.  Unseen tokens receive
    ``-log(min_freq)`` as a penalty.
    """
    if not codes:
        return 0.0

    neg_log_freqs = []
    for tok in codes:
        freq = freq_table.get(tok, min_freq)
        neg_log_freqs.append(-math.log(max(freq, min_freq)))

    return sum(neg_log_freqs) / len(neg_log_freqs)


def score_all(
    codes_list: list[list[str]],
    freq_table: dict[str, float],
) -> np.ndarray:
    """Score a list of sequences, returning a 1-D array."""
    return np.array([score_sequence(c, freq_table) for c in codes_list])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 21 -- Frequency-based anomaly baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-pkl", required=True, type=Path, help="Training split pickle/parquet")
    p.add_argument("--test-pkl", required=True, type=Path, help="Test split pickle/parquet")
    p.add_argument("--code-col", default="codes_ont", help="Column with token lists")
    p.add_argument("--n-per-type", type=int, default=500, help="Records per anomaly type")
    p.add_argument("--output-dir", type=Path, default=Path("logs/detector"), help="Output directory")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- load data ---
    def _load(p: Path) -> pd.DataFrame:
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_pickle(p)

    train_df = _load(args.train_pkl)
    test_df = _load(args.test_pkl)

    # --- build frequency table ---
    freq_table = build_frequency_table(train_df, code_col=args.code_col)

    # --- build anomaly test set ---
    anomaly_df = build_anomaly_test_set(test_df, code_col=args.code_col, n_per_type=args.n_per_type)
    log.info("Anomaly test set: %d records", len(anomaly_df))

    # --- score ---
    codes_list = anomaly_df["codes"].tolist()
    scores = score_all(codes_list, freq_table)
    labels = anomaly_df["label"].values

    # --- metrics ---
    roc_auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))

    results: dict[str, Any] = {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap, 4),
        "mean_score_normal": round(float(scores[labels == 0].mean()), 6),
        "mean_score_anomalous": round(float(scores[labels == 1].mean()), 6),
        "n_normal": int((labels == 0).sum()),
        "n_anomalous": int((labels == 1).sum()),
        "vocab_size": len(freq_table),
    }

    # Per-type breakdown
    per_type: dict[str, dict[str, Any]] = {}
    for atype in anomaly_df["anomaly_type"].unique():
        mask = anomaly_df["anomaly_type"].values == atype
        per_type[atype] = {
            "count": int(mask.sum()),
            "mean_score": round(float(scores[mask].mean()), 6),
        }
    results["per_type"] = per_type

    log.info("Frequency baseline -- ROC AUC: %.4f, AP: %.4f", roc_auc, ap)
    log.info(
        "Mean score -- normal: %.4f, anomalous: %.4f",
        results["mean_score_normal"], results["mean_score_anomalous"],
    )

    # --- save ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "frequency_baseline.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
