"""Phase 2b -- tests for token -> source-code normalization."""

from __future__ import annotations

from src.preprocessing.code_normalization import (
    icd9_candidates,
    icd10_candidates,
    normalize_diagnosis_token,
    normalize_drug_token,
)


def test_icd9_numeric_dotting() -> None:
    cands = icd9_candidates("4019")
    assert "4019" in cands
    assert "401.9" in cands


def test_icd9_v_and_e_codes() -> None:
    assert "V27.0" in icd9_candidates("V270")
    assert "E849.7" in icd9_candidates("E8497")


def test_icd10_dotting() -> None:
    assert "E78.5" in icd10_candidates("E785")
    assert "Z98.61" in icd10_candidates("Z9861")
    # short codes have no dotted form beyond themselves
    assert icd10_candidates("E11") == ["E11"]


def test_normalize_diagnosis_token_dispatch() -> None:
    assert "401.9" in normalize_diagnosis_token("DX_9_4019")
    assert "E78.5" in normalize_diagnosis_token("DX_10_E785")


def test_normalize_drug_token() -> None:
    assert "ACETAMINOPHEN" in normalize_drug_token("MED_ACETAMINOPHEN")
    # trailing route descriptor produces a conservative fallback
    cands = normalize_drug_token("MED_ACETAMINOPHEN_IV")
    assert "ACETAMINOPHEN_IV" in cands
    assert "ACETAMINOPHEN" in cands


def test_no_false_dotting_for_short_codes() -> None:
    # 3-char numeric body should not be dotted into something invalid
    assert icd9_candidates("401") == ["401"]
