"""Phase 3 -- tests for threshold calibration utilities."""

from __future__ import annotations

from src.evaluation.calibration import (
    apply_threshold,
    select_best_f1_threshold,
    select_fixed_precision_threshold,
    select_fixed_recall_threshold,
)

# Separable-ish validation set: positives score higher.
VAL_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
VAL_SCORES = [0.1, 0.2, 0.3, 0.45, 0.55, 0.7, 0.8, 0.9]


def test_best_f1_threshold_separates() -> None:
    res = select_best_f1_threshold(VAL_LABELS, VAL_SCORES)
    assert res.selected_on == "validation"
    assert res.f1 == 1.0  # perfectly separable
    assert 0.45 < res.threshold < 0.55


def test_fixed_recall_threshold() -> None:
    res = select_fixed_recall_threshold(VAL_LABELS, VAL_SCORES, target_recall=1.0)
    assert res.recall >= 1.0
    # applying to a test set returns structured metrics on test
    out = apply_threshold([0, 1], [0.2, 0.8], res.threshold)
    assert out["applied_on"] == "test"


def test_fixed_precision_threshold() -> None:
    res = select_fixed_precision_threshold(VAL_LABELS, VAL_SCORES, target_precision=1.0)
    assert res.precision >= 1.0


def test_apply_threshold_confusion_counts() -> None:
    out = apply_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert out["tp"] == 2 and out["tn"] == 2 and out["fp"] == 0 and out["fn"] == 0
    assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0


def test_degenerate_scores_do_not_crash() -> None:
    res = select_best_f1_threshold([0, 1], [0.5, 0.5])
    assert res.threshold is not None
