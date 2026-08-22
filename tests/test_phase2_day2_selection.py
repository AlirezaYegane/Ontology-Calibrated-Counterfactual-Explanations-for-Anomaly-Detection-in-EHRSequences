from __future__ import annotations

import numpy as np
import pytest

from src.baselines.selection import (
    aggregate_upper_scores,
    ensure_validation_only_paths,
    select_best_non_extreme,
)


def test_validation_only_guard_rejects_test() -> None:
    with pytest.raises(ValueError):
        ensure_validation_only_paths(
            "train.pkl",
            "test.pkl",
        )


def test_validation_only_guard_accepts_val() -> None:
    ensure_validation_only_paths(
        "train.pkl",
        "val.pkl",
    )


def test_aggregate_upper_scores() -> None:
    values = np.asarray(
        [1.0, 2.0, 3.0, 4.0, 10.0],
        dtype=float,
    )

    result = aggregate_upper_scores(
        values,
        quantiles=[0.80, 0.90],
        top_ks=[2, 3],
    )

    assert result["worst"] == 10.0
    assert result["mean"] == pytest.approx(4.0)
    assert result["topk2"] == pytest.approx(7.0)
    assert result["topk3"] == pytest.approx(
        (3.0 + 4.0 + 10.0) / 3.0
    )


def test_selection_excludes_extreme() -> None:
    candidates = [
        {
            "candidate_id": "extreme",
            "model": "B1",
            "is_extreme": True,
            "overall": {
                "pr_auc": 0.90,
                "roc_auc": 0.90,
            },
            "bias": {
                "max_abs_spearman": 0.90,
            },
        },
        {
            "candidate_id": "robust_a",
            "model": "B1",
            "is_extreme": False,
            "overall": {
                "pr_auc": 0.70,
                "roc_auc": 0.75,
            },
            "bias": {
                "max_abs_spearman": 0.30,
            },
        },
        {
            "candidate_id": "robust_b",
            "model": "B1",
            "is_extreme": False,
            "overall": {
                "pr_auc": 0.71,
                "roc_auc": 0.72,
            },
            "bias": {
                "max_abs_spearman": 0.40,
            },
        },
    ]

    selected = select_best_non_extreme(
        candidates,
        "B1",
    )

    assert selected["candidate_id"] == "robust_b"


def test_selection_uses_bias_as_tiebreaker() -> None:
    candidates = [
        {
            "candidate_id": "a",
            "model": "B0",
            "is_extreme": False,
            "overall": {
                "pr_auc": 0.70,
                "roc_auc": 0.75,
            },
            "bias": {
                "max_abs_spearman": 0.40,
            },
        },
        {
            "candidate_id": "b",
            "model": "B0",
            "is_extreme": False,
            "overall": {
                "pr_auc": 0.70,
                "roc_auc": 0.73,
            },
            "bias": {
                "max_abs_spearman": 0.20,
            },
        },
    ]

    selected = select_best_non_extreme(
        candidates,
        "B0",
    )

    assert selected["candidate_id"] == "b"
