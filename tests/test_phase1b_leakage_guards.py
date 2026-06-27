"""Phase 1b -- tests that benchmark-v2 never leaks answer keys to model input."""

from __future__ import annotations

import pytest

from src.evaluation.protocol import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    validate_no_forbidden_columns,
)
from src.preprocessing.anomaly_injection_v2 import (
    AnomalyGenConfig,
    inject_demographic_incompatibility,
    inject_forbidden_cooccurrence,
    inject_medication_indication_mismatch,
)

CFG = AnomalyGenConfig(seed=7)

CASES = [
    (
        {"codes": ["DX_10_O0900", "MED_X"], "gender": "F", "subject_id": "1"},
        inject_demographic_incompatibility,
    ),
    (
        {"codes": ["MED_INSULIN", "DX_10_E119"], "gender": "M", "subject_id": "2"},
        inject_medication_indication_mismatch,
    ),
    (
        {"codes": ["DX_10_E119"], "gender": "F", "subject_id": "3"},
        inject_forbidden_cooccurrence,
    ),
]


def test_model_visible_has_no_forbidden_columns() -> None:
    for rec, fn in CASES:
        ev = fn(rec, CFG)
        assert ev is not None
        row = ev.to_model_visible_row()  # runs internal leakage guard
        # model-visible keys must be safe for the protocol-level guard too
        validate_no_forbidden_columns(list(row.keys()))
        assert set(row.keys()) <= {"codes", "gender", "age_group"}


def test_answer_key_lives_only_in_hidden_eval() -> None:
    rec, fn = CASES[1]  # medication mismatch
    ev = fn(rec, CFG)
    assert ev is not None
    row = ev.to_model_visible_row()
    # the removed indication code must NOT be recoverable from model-visible fields
    removed = ev.hidden_eval["removed_indication_codes"]
    assert removed  # answer key present in hidden_eval
    assert "removed_indication_codes" not in row


def test_to_model_visible_row_rejects_injected_answer_key() -> None:
    rec, fn = CASES[0]
    ev = fn(rec, CFG)
    assert ev is not None
    # simulate accidental leakage: putting an answer key into model_visible
    ev.model_visible["original_gender"] = "F"
    with pytest.raises(ValueError, match="forbidden"):
        ev.to_model_visible_row()


def test_demographic_does_not_leak_via_injected_token() -> None:
    # The model-visible sequence for a gender-flip anomaly is IDENTICAL to the
    # source normal's sequence -> no token reveals the anomaly.
    rec = {"codes": ["DX_10_O0900", "DX_10_I10"], "gender": "F", "subject_id": "9"}
    ev = inject_demographic_incompatibility(rec, CFG)
    assert ev is not None
    assert ev.model_visible["codes"] == rec["codes"]  # zero token change


def test_protocol_forbidden_set_covers_v2_keys() -> None:
    for col in ("original_code", "removed_code", "codes_original", "source_row_id"):
        assert col in FORBIDDEN_LEAKAGE_COLUMNS
