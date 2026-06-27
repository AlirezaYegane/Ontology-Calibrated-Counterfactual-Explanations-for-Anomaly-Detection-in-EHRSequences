"""
src/evaluation/evaluate_detector_unsup.py
==========================================
Phase 3 -- Evaluate the unsupervised detector's record-level anomaly scores.

Given a loaded detector and labelled records, compute scores and (when both
classes are present) ROC-AUC / AP via the project's stats utilities. Designed to
be driven by the Phase 3 eval script; safe on degenerate label sets.
"""

from __future__ import annotations

from typing import Any

from src.models.detector_unsup import UnsupervisedSequenceDetector


def score_records(
    detector: UnsupervisedSequenceDetector,
    records: list[dict[str, Any]],
    seq_key: str = "codes",
) -> list[float]:
    sequences = [
        [str(t) for t in r.get(seq_key, [])]
        if isinstance(r.get(seq_key), (list, tuple))
        else []
        for r in records
    ]
    return detector.anomaly_scores(sequences)


def evaluate(
    detector: UnsupervisedSequenceDetector,
    records: list[dict[str, Any]],
    seq_key: str = "codes",
    label_key: str = "is_synthetic_anomaly",
) -> dict[str, Any]:
    scores = score_records(detector, records, seq_key=seq_key)
    labels: list[int] = []
    for r in records:
        try:
            labels.append(int(r.get(label_key, 0)))
        except (TypeError, ValueError):
            labels.append(0)

    out: dict[str, Any] = {
        "n_records": len(records),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "score_mean": (sum(scores) / len(scores)) if scores else None,
    }
    if len(set(labels)) >= 2 and scores:
        from src.evaluation.stats import bootstrap_auc_ap

        out["metrics"] = bootstrap_auc_ap(labels, scores, n_boot=200, seed=42)
    else:
        out["metrics"] = None
        out["note"] = (
            "Degenerate labels (single class) -> detection metrics not computed."
        )
    return out


__all__ = ["score_records", "evaluate"]
