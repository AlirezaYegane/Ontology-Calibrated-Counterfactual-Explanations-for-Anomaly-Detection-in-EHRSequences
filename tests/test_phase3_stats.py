"""Phase 3 -- tests for bootstrap CI / paired-difference statistics."""

from __future__ import annotations

import numpy as np

from src.evaluation.stats import (
    average_precision,
    bootstrap_ci,
    paired_bootstrap_diff,
    roc_auc,
)


def test_roc_auc_perfect_and_random() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_average_precision_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.normal(0, 1, 50), rng.normal(1.5, 1, 50)])
    ci = bootstrap_ci(roc_auc, y, scores, n_boot=200, seed=1)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["n_boot_valid"] > 0


def test_paired_bootstrap_diff_detects_better_model() -> None:
    rng = np.random.default_rng(2)
    y = np.array([0] * 60 + [1] * 60)
    good = np.concatenate(
        [rng.normal(0, 1, 60), rng.normal(2.5, 1, 60)]
    )  # strong separation
    weak = rng.normal(0, 1, 120)  # no signal
    res = paired_bootstrap_diff(roc_auc, y, good, weak, n_boot=300, seed=3)
    assert res["observed_diff"] > 0  # good model has higher AUC
    assert res["p_value"] < 0.05  # significant


def test_degenerate_labels_safe() -> None:
    y = np.array([1, 1, 1])
    ci = bootstrap_ci(roc_auc, y, np.array([0.1, 0.2, 0.3]), n_boot=50)
    # single-class -> no valid resamples, returns nan CI without crashing
    assert ci["n_boot_valid"] == 0
