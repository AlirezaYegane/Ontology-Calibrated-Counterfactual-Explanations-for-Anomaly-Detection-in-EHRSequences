"""
src/evaluation/calibration.py
=============================
Phase 3 -- Threshold calibration utilities.

All threshold SELECTION happens on a validation split; APPLICATION happens on a
separate test split. The API makes test-set threshold tuning hard to do by
accident: you select on validation scores, then apply the returned threshold to
test scores. Pure functions, NumPy-based, no sklearn dependency required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    selected_on: str  # always "validation"
    strategy: str


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def _prf(
    y_true: np.ndarray, scores: np.ndarray, thr: float
) -> tuple[float, float, float]:
    y_pred = (scores >= thr).astype(int)
    tp, fp, fn, _ = _confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return precision, recall, f1


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(scores)
    if len(uniq) == 1:
        return np.array([uniq[0], uniq[0] + 1e-9])
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    return np.concatenate([[uniq[0] - 1e-9], mids, [uniq[-1] + 1e-9]])


def select_best_f1_threshold(val_labels, val_scores) -> ThresholdResult:
    """Select the threshold maximizing F1 on the VALIDATION split."""
    y = np.asarray(val_labels)
    s = np.asarray(val_scores, dtype=float)
    best = ThresholdResult(
        float(s.max() + 1e-9) if len(s) else 0.0,
        0.0,
        0.0,
        -1.0,
        "validation",
        "best_f1",
    )
    for thr in _candidate_thresholds(s):
        p, r, f1 = _prf(y, s, thr)
        if f1 > best.f1:
            best = ThresholdResult(float(thr), p, r, f1, "validation", "best_f1")
    return best


def select_fixed_recall_threshold(
    val_labels, val_scores, target_recall: float
) -> ThresholdResult:
    """Lowest threshold (highest precision) achieving at least `target_recall`."""
    y = np.asarray(val_labels)
    s = np.asarray(val_scores, dtype=float)
    best: ThresholdResult | None = None
    for thr in sorted(_candidate_thresholds(s), reverse=True):
        p, r, f1 = _prf(y, s, thr)
        if r >= target_recall:
            best = ThresholdResult(
                float(thr), p, r, f1, "validation", f"fixed_recall>={target_recall}"
            )
            break
    if best is None:
        # cannot reach target recall: use the most permissive threshold
        thr = float(s.min() - 1e-9) if len(s) else 0.0
        p, r, f1 = _prf(y, s, thr)
        best = ThresholdResult(
            thr, p, r, f1, "validation", f"fixed_recall>={target_recall}(unmet)"
        )
    return best


def select_fixed_precision_threshold(
    val_labels, val_scores, target_precision: float
) -> ThresholdResult:
    """Lowest threshold (highest recall) achieving at least `target_precision`."""
    y = np.asarray(val_labels)
    s = np.asarray(val_scores, dtype=float)
    best: ThresholdResult | None = None
    for thr in sorted(_candidate_thresholds(s)):
        p, r, f1 = _prf(y, s, thr)
        if p >= target_precision:
            best = ThresholdResult(
                float(thr),
                p,
                r,
                f1,
                "validation",
                f"fixed_precision>={target_precision}",
            )
            break
    if best is None:
        thr = float(s.max() + 1e-9) if len(s) else 0.0
        p, r, f1 = _prf(y, s, thr)
        best = ThresholdResult(
            thr, p, r, f1, "validation", f"fixed_precision>={target_precision}(unmet)"
        )
    return best


def apply_threshold(test_labels, test_scores, threshold: float) -> dict[str, float]:
    """Apply a VALIDATION-selected threshold to the TEST split."""
    y = np.asarray(test_labels)
    s = np.asarray(test_scores, dtype=float)
    y_pred = (s >= threshold).astype(int)
    tp, fp, fn, tn = _confusion(y, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    n = len(y)
    return {
        "threshold": float(threshold),
        "applied_on": "test",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "predicted_positive_rate": (tp + fp) / n if n else 0.0,
    }


__all__ = [
    "ThresholdResult",
    "select_best_f1_threshold",
    "select_fixed_recall_threshold",
    "select_fixed_precision_threshold",
    "apply_threshold",
]
