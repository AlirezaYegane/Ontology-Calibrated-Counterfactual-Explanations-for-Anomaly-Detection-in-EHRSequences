"""
tests/test_anomaly_injection.py
================================
Unit tests for Day 17 anomaly injection module.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.preprocessing.anomaly_injection import (
    InjectedAnomaly,
    build_anomaly_test_set,
    inject_demographic_conflict,
    inject_missing_indication,
    inject_random_code_swap,
)


# ---------------------------------------------------------------------------
# inject_missing_indication
# ---------------------------------------------------------------------------


class TestMissingIndication:
    def test_removes_diagnoses(self) -> None:
        codes = ["SNOMED:123", "RXNORM:456", "SNOMED:789", "PROC_ICD9:001"]
        rng = random.Random(0)
        result = inject_missing_indication(codes, rng)

        assert isinstance(result, InjectedAnomaly)
        assert result.anomaly_type == "missing_indication"
        # No SNOMED tokens should remain
        assert all(not c.startswith("SNOMED:") for c in result.anomalous_codes)
        # Non-SNOMED tokens should be preserved
        assert "RXNORM:456" in result.anomalous_codes
        assert "PROC_ICD9:001" in result.anomalous_codes
        # Original codes should be unchanged
        assert result.original_codes == tuple(codes)

    def test_all_snomed_produces_nonempty(self) -> None:
        codes = ["SNOMED:111", "SNOMED:222"]
        rng = random.Random(0)
        result = inject_missing_indication(codes, rng)
        # Should produce at least one token (UNK_EMPTY fallback)
        assert len(result.anomalous_codes) >= 1

    def test_no_snomed_returns_unchanged(self) -> None:
        codes = ["RXNORM:100", "PROC_ICD9:200"]
        rng = random.Random(0)
        result = inject_missing_indication(codes, rng)
        assert list(result.anomalous_codes) == codes


# ---------------------------------------------------------------------------
# inject_random_code_swap
# ---------------------------------------------------------------------------


class TestRandomCodeSwap:
    def test_changes_one_code(self) -> None:
        codes = ["SNOMED:123", "RXNORM:456", "SNOMED:789"]
        pool = ["SNOMED:AAA", "SNOMED:BBB", "RXNORM:CCC"]
        rng = random.Random(42)
        result = inject_random_code_swap(codes, pool, rng)

        assert result is not None
        assert result.anomaly_type == "random_code_swap"
        # Exactly one code should differ
        diffs = [
            i for i, (a, b) in enumerate(
                zip(result.original_codes, result.anomalous_codes)
            )
            if a != b
        ]
        assert len(diffs) == 1
        # The swapped position should have been a SNOMED code
        assert result.original_codes[diffs[0]].startswith("SNOMED:")
        # Length should be unchanged
        assert len(result.anomalous_codes) == len(result.original_codes)

    def test_no_snomed_returns_none(self) -> None:
        codes = ["RXNORM:100", "PROC_ICD9:200"]
        pool = ["SNOMED:AAA"]
        rng = random.Random(0)
        result = inject_random_code_swap(codes, pool, rng)
        assert result is None


# ---------------------------------------------------------------------------
# inject_demographic_conflict
# ---------------------------------------------------------------------------


class TestDemographicConflict:
    def test_adds_pregnancy_to_male(self) -> None:
        codes = ["SNOMED:123", "RXNORM:456"]
        rng = random.Random(0)
        result = inject_demographic_conflict(codes, "M", None, rng)

        assert result is not None
        assert result.anomaly_type == "demographic_conflict"
        assert len(result.anomalous_codes) == len(codes) + 1
        # The added code should be one of the pregnancy codes
        added = set(result.anomalous_codes) - set(result.original_codes)
        assert len(added) == 1

    def test_female_returns_none(self) -> None:
        codes = ["SNOMED:123"]
        rng = random.Random(0)
        result = inject_demographic_conflict(codes, "F", None, rng)
        assert result is None

    def test_custom_pregnancy_codes(self) -> None:
        codes = ["SNOMED:123"]
        custom = ["SNOMED:CUSTOM_PREG"]
        rng = random.Random(0)
        result = inject_demographic_conflict(codes, "M", custom, rng)
        assert result is not None
        assert "SNOMED:CUSTOM_PREG" in result.anomalous_codes


# ---------------------------------------------------------------------------
# build_anomaly_test_set
# ---------------------------------------------------------------------------


class TestBuildAnomalyTestSet:
    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "codes_ont": [
                ["SNOMED:123", "RXNORM:456", "SNOMED:789"],
                ["SNOMED:111", "SNOMED:222", "RXNORM:333"],
                ["SNOMED:444", "PROC_ICD9:555"],
                ["RXNORM:666", "SNOMED:777"],
            ] * 50,
            "gender": ["M", "F", "M", "F"] * 50,
        })

    def test_has_correct_labels(self, sample_df: pd.DataFrame) -> None:
        result = build_anomaly_test_set(sample_df, n_per_type=10, seed=42)

        assert "label" in result.columns
        assert "anomaly_type" in result.columns
        assert "codes" in result.columns
        assert "gender" in result.columns

        # Should have both normal and anomalous records
        assert set(result["label"].unique()) == {0, 1}

        # Normal records should have label 0
        normals = result[result["anomaly_type"] == "normal"]
        assert (normals["label"] == 0).all()

        # Anomalous records should have label 1
        anomalous = result[result["label"] == 1]
        assert (anomalous["anomaly_type"] != "normal").all()

    def test_anomaly_types_present(self, sample_df: pd.DataFrame) -> None:
        result = build_anomaly_test_set(sample_df, n_per_type=10, seed=42)
        types = set(result["anomaly_type"].unique())
        assert "normal" in types
        assert "missing_indication" in types
        assert "random_code_swap" in types
        assert "demographic_conflict" in types

    def test_n_per_type_respected(self, sample_df: pd.DataFrame) -> None:
        result = build_anomaly_test_set(sample_df, n_per_type=5, seed=42)
        normals = result[result["anomaly_type"] == "normal"]
        assert len(normals) == 5
        mi = result[result["anomaly_type"] == "missing_indication"]
        assert len(mi) == 5
