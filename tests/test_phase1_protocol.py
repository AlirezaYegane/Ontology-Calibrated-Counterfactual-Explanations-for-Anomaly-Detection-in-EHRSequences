"""Tests for src/evaluation/protocol.py (Phase 1b non-circular protocol)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.protocol import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    assert_no_subject_overlap,
    find_forbidden_columns,
    make_leave_anomaly_type_out_splits,
    summarize_label_distribution,
    summarize_split_overlap,
    validate_no_forbidden_columns,
)


def _toy_labeled_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4, 5],
            "codes": [
                ["DX_10_A"],
                ["DX_10_B"],
                ["MED_X"],
                ["DX_9_1"],
                ["PROC_9_2"],
                ["DX_10_C"],
            ],
            "is_synthetic_anomaly": [0, 1, 0, 1, 1, 0],
            "anomaly_type": [
                None,
                "demographic_conflict",
                None,
                "missing_diagnosis",
                "demographic_conflict",
                None,
            ],
        }
    )


def test_assert_no_subject_overlap_catches_overlap() -> None:
    train = pd.DataFrame({"subject_id": [1, 2, 3]})
    test = pd.DataFrame({"subject_id": [3, 4]})  # subject 3 overlaps
    with pytest.raises(ValueError, match="Subject leakage"):
        assert_no_subject_overlap({"train": train, "test": test})


def test_assert_no_subject_overlap_passes_when_disjoint() -> None:
    train = pd.DataFrame({"subject_id": [1, 2, 3]})
    test = pd.DataFrame({"subject_id": [4, 5]})
    assert_no_subject_overlap({"train": train, "test": test})  # must not raise


def test_summarize_split_overlap_reports_counts() -> None:
    train = pd.DataFrame({"subject_id": [1, 2, 3]})
    test = pd.DataFrame({"subject_id": [3, 4]})
    summary = summarize_split_overlap({"train": train, "test": test})
    assert summary["total_overlap"] == 1
    assert summary["per_split_subject_counts"] == {"train": 3, "test": 2}
    pair = summary["pairwise_overlap"][0]
    assert pair["overlap_count"] == 1
    assert "3" in pair["overlap_examples"]


def test_validate_no_forbidden_columns_catches_leakage() -> None:
    df = pd.DataFrame({"codes": [["A"]], "corruption_note": ["Removed DX_9_1"]})
    with pytest.raises(ValueError, match="Forbidden"):
        validate_no_forbidden_columns(df)


def test_validate_no_forbidden_columns_passes_when_clean() -> None:
    df = pd.DataFrame({"codes": [["A"]], "gender": ["M"]})
    validate_no_forbidden_columns(df)  # must not raise


def test_find_forbidden_columns_is_case_insensitive() -> None:
    found = find_forbidden_columns(["Codes", "Bad_Code", "Expected_Code"])
    assert set(map(str.lower, found)) == {"bad_code", "expected_code"}


def test_known_leakage_columns_present_in_constant() -> None:
    for col in (
        "bad_code",
        "expected_code",
        "replacement_code",
        "source_row_id",
        "corruption_note",
    ):
        assert col in FORBIDDEN_LEAKAGE_COLUMNS


def test_leave_anomaly_type_out_excludes_held_type_from_train() -> None:
    df = _toy_labeled_df()
    out = make_leave_anomaly_type_out_splits(df, held_out_type="demographic_conflict")
    train, ev = out["train"], out["eval"]

    # Held-out type must not appear in training at all.
    assert (train["anomaly_type"] == "demographic_conflict").sum() == 0
    # Held-out type anomalies must appear in eval.
    assert (ev["anomaly_type"] == "demographic_conflict").sum() == 2
    # The other anomaly type must be in train, not isolated in eval.
    assert (train["anomaly_type"] == "missing_diagnosis").sum() == 1
    assert out["meta"]["held_out_type"] == "demographic_conflict"


def test_leave_anomaly_type_out_rejects_unknown_type() -> None:
    df = _toy_labeled_df()
    with pytest.raises(ValueError, match="not among anomaly types"):
        make_leave_anomaly_type_out_splits(df, held_out_type="does_not_exist")


def test_summarize_label_distribution() -> None:
    df = _toy_labeled_df()
    out = summarize_label_distribution(df)
    assert out["n_records"] == 6
    assert out["n_positive"] == 3
    assert out["n_negative"] == 3
    assert "by_type" in out
