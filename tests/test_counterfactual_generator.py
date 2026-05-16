from __future__ import annotations

from src.explanations.counterfactual import (
    detect_violations,
    generate_counterfactual,
    propose_one_step_edits,
)


def test_demographic_conflict_removes_pregnancy_code_for_male_record() -> None:
    row = {
        "sequence_tokens": ["ICD10_O24_PREGNANCY_DIABETES", "ICD10_E11_DIABETES"],
        "gender": "M",
        "anomaly_type": "demographic_conflict",
    }

    result = generate_counterfactual(row["sequence_tokens"], row)

    assert result.status == "improved"
    assert result.edit_count == 1
    assert result.edits[0].kind == "remove"
    assert "ICD10_O24_PREGNANCY_DIABETES" not in result.counterfactual_codes
    assert result.counterfactual_violation_score < result.original_violation_score


def test_missing_diagnosis_adds_expected_code() -> None:
    row = {
        "sequence_tokens": ["RXNORM_12345"],
        "anomaly_type": "missing_diagnosis",
        "expected_code": "ICD10_J18_PNEUMONIA",
    }

    result = generate_counterfactual(row["sequence_tokens"], row)

    assert result.status == "improved"
    assert result.edit_count == 1
    assert result.edits[0].kind == "add"
    assert "ICD10_J18_PNEUMONIA" in result.counterfactual_codes
    assert result.counterfactual_violation_score < result.original_violation_score


def test_no_candidate_is_reported_when_no_safe_edit_exists() -> None:
    row = {
        "sequence_tokens": ["ICD10_E11_DIABETES"],
        "anomaly_type": "missing_diagnosis",
    }

    result = generate_counterfactual(row["sequence_tokens"], row)

    assert result.status == "no_candidate"
    assert result.edit_count == 0


def test_proposed_edits_are_constrained_to_row_evidence() -> None:
    row = {
        "sequence_tokens": ["RXNORM_12345"],
        "anomaly_type": "missing_diagnosis",
        "expected_code": "ICD10_A41_SEPSIS",
    }

    edits = propose_one_step_edits(row["sequence_tokens"], row)

    assert len(edits) == 1
    assert edits[0].kind == "add"
    assert edits[0].new_code == "ICD10_A41_SEPSIS"


def test_detect_violations_uses_expected_code_absence() -> None:
    row = {
        "sequence_tokens": ["RXNORM_999"],
        "anomaly_type": "medication_mismatch",
        "expected_code": "ICD10_N39_UTI",
    }

    violations = detect_violations(row["sequence_tokens"], row)

    assert len(violations) == 1
    assert violations[0].expected_codes == ("ICD10_N39_UTI",)


def test_demographic_conflict_uses_explicit_bad_code_from_repair_ready_data() -> None:
    row = {
        "sequence_tokens": ["DX_10_Z3400", "DX_9_25000"],
        "anomaly_type": "demographic_conflict",
        "bad_code": "DX_10_Z3400",
        "repair_edit_hint": "remove_bad_code",
    }

    result = generate_counterfactual(row["sequence_tokens"], row)

    assert result.status == "improved"
    assert result.edit_count == 1
    assert result.edits[0].kind == "remove"
    assert result.edits[0].code == "DX_10_Z3400"
    assert "DX_10_Z3400" not in result.counterfactual_codes
    assert result.counterfactual_violation_score < result.original_violation_score
