"""Tests for src/preprocessing/anomaly_injection_v2.py (real non-circular injectors)."""

from __future__ import annotations

from src.preprocessing.anomaly_injection_v2 import (
    FORBIDDEN_MODEL_VISIBLE_COLUMNS,
    AnomalyEvent,
    AnomalyGenConfig,
    AnomalyTypeV2,
    INJECTOR_REGISTRY,
    validate_model_visible_fields,
)


def test_imports_and_registry_clean() -> None:
    assert set(INJECTOR_REGISTRY.keys()) == set(AnomalyTypeV2)


def test_config_is_deterministic() -> None:
    cfg = AnomalyGenConfig(seed=123)
    a = cfg.derive_rng("med").random()
    b = AnomalyGenConfig(seed=123).derive_rng("med").random()
    assert a == b
    assert a != cfg.derive_rng("demo").random()


def test_validate_model_visible_fields_rejects_answer_keys() -> None:
    import pytest

    with pytest.raises(ValueError, match="forbidden"):
        validate_model_visible_fields({"codes": ["A"], "original_gender": "F"})
    with pytest.raises(ValueError, match="forbidden"):
        validate_model_visible_fields({"codes": ["A"], "removed_code": "DX_10_E11"})


def test_model_visible_fields_accepts_clean_payload() -> None:
    validate_model_visible_fields({"codes": ["A"], "gender": "M", "age_group": "65-79"})


def test_anomaly_event_field_separation() -> None:
    event = AnomalyEvent(
        anomaly_type=AnomalyTypeV2.FORBIDDEN_COOCCURRENCE,
        model_visible={"codes": ["A"], "gender": "M", "age_group": "x"},
        audit={"method": "add_exclusive_partner"},
        hidden_eval={"added_code": "DX_10_E10_9"},
    )
    row = event.to_model_visible_row()
    assert "added_code" not in row
    assert event.hidden_eval["added_code"] == "DX_10_E10_9"
    assert not (set(map(str.lower, row.keys())) & FORBIDDEN_MODEL_VISIBLE_COLUMNS)
