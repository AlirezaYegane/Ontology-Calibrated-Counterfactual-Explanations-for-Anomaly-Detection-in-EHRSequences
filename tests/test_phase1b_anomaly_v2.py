"""Phase 1b -- tests for the real non-circular anomaly-v2 injectors."""

from __future__ import annotations

from src.preprocessing.anomaly_injection_v2 import (
    AnomalyGenConfig,
    AnomalyTypeV2,
    inject_demographic_incompatibility,
    inject_forbidden_cooccurrence,
    inject_medication_indication_mismatch,
)

CFG = AnomalyGenConfig(seed=42)


def test_demographic_gender_flip_no_token_injected() -> None:
    # Female record with a pregnancy code -> flip gender to M, sequence unchanged.
    rec = {
        "codes": ["DX_10_O0900", "DX_10_I10", "MED_ASPIRIN"],
        "gender": "F",
        "age_group": "30-44",
        "subject_id": "1",
    }
    ev = inject_demographic_incompatibility(rec, CFG)
    assert ev is not None
    assert ev.anomaly_type == AnomalyTypeV2.DEMOGRAPHIC_INCOMPATIBILITY
    # sequence identical (no injected token) -> non-circular
    assert ev.model_visible["codes"] == rec["codes"]
    assert ev.model_visible["gender"] == "M"  # only the field changed
    # answer key only in hidden_eval
    assert ev.hidden_eval["original_gender"] == "F"
    assert "DX_10_O0900" in ev.hidden_eval["conflicting_codes"]


def test_demographic_returns_none_without_sex_specific_code() -> None:
    rec = {"codes": ["DX_10_I10", "MED_ASPIRIN"], "gender": "F", "subject_id": "2"}
    assert inject_demographic_incompatibility(rec, CFG) is None


def test_medication_indication_mismatch_removes_indication() -> None:
    rec = {
        "codes": ["MED_INSULIN", "DX_10_E119", "DX_10_I10"],
        "gender": "M",
        "subject_id": "3",
    }
    ev = inject_medication_indication_mismatch(rec, CFG)
    assert ev is not None
    assert ev.anomaly_type == AnomalyTypeV2.MEDICATION_INDICATION_MISMATCH
    # the diabetes indication is removed from the model-visible sequence
    assert "DX_10_E119" not in ev.model_visible["codes"]
    assert "MED_INSULIN" in ev.model_visible["codes"]  # drug stays (common token)
    assert "DX_10_E119" in ev.hidden_eval["removed_indication_codes"]


def test_medication_returns_none_without_drug_or_indication() -> None:
    # insulin but no diabetes -> already a mismatch, nothing to remove
    rec = {"codes": ["MED_INSULIN", "DX_10_I10"], "gender": "M", "subject_id": "4"}
    assert inject_medication_indication_mismatch(rec, CFG) is None


def test_forbidden_cooccurrence_adds_common_partner() -> None:
    rec = {"codes": ["DX_10_E119", "MED_METFORMIN"], "gender": "F", "subject_id": "5"}
    ev = inject_forbidden_cooccurrence(rec, CFG)
    assert ev is not None
    assert ev.anomaly_type == AnomalyTypeV2.FORBIDDEN_COOCCURRENCE
    added = ev.hidden_eval["added_code"]
    assert added in ev.model_visible["codes"]
    assert added.upper().startswith("DX_10_E10")  # type-1 added to type-2 record


def test_forbidden_returns_none_when_both_already_present() -> None:
    rec = {"codes": ["DX_10_E119", "DX_10_E109"], "gender": "F", "subject_id": "6"}
    assert inject_forbidden_cooccurrence(rec, CFG) is None


def test_each_anomaly_type_constructible() -> None:
    recs = {
        AnomalyTypeV2.DEMOGRAPHIC_INCOMPATIBILITY: (
            {"codes": ["DX_10_O0900"], "gender": "F", "subject_id": "a"},
            inject_demographic_incompatibility,
        ),
        AnomalyTypeV2.MEDICATION_INDICATION_MISMATCH: (
            {"codes": ["MED_INSULIN", "DX_10_E119"], "gender": "M", "subject_id": "b"},
            inject_medication_indication_mismatch,
        ),
        AnomalyTypeV2.FORBIDDEN_COOCCURRENCE: (
            {"codes": ["DX_10_E119"], "gender": "F", "subject_id": "c"},
            inject_forbidden_cooccurrence,
        ),
    }
    for atype, (rec, fn) in recs.items():
        ev = fn(rec, CFG)
        assert ev is not None and ev.anomaly_type == atype
