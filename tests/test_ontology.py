from __future__ import annotations

from src.ontology import (
    ClinicalRecord,
    DemographicRule,
    MutualExclusionRule,
    OntologyEngine,
    OntologyIndex,
    RequiredCodesRule,
)


def build_toy_engine() -> OntologyEngine:
    index = OntologyIndex(
        preferred_terms={
            "DX_PREG": "Pregnancy",
            "DX_INF": "Infection",
            "RX_ABX": "Antibiotic",
            "DX_MALE_ONLY": "Male-specific condition",
            "DX_FEMALE_ONLY": "Female-specific condition",
            "DX_HEART_FAILURE": "Heart failure",
            "DX_CARDIOMYOPATHY": "Cardiomyopathy",
            "DX_ARRHYTHMIA": "Arrhythmia",
            "PARENT_CARDIAC": "Cardiac disorder",
        },
        parents={
            "DX_HEART_FAILURE": ["PARENT_CARDIAC"],
            "DX_CARDIOMYOPATHY": ["PARENT_CARDIAC"],
            "DX_ARRHYTHMIA": ["PARENT_CARDIAC"],
        },
        children={
            "PARENT_CARDIAC": ["DX_HEART_FAILURE", "DX_CARDIOMYOPATHY", "DX_ARRHYTHMIA"],
        },
        required_diagnoses_for_code={
            "RX_ABX": ["DX_INF"],
        },
        mutually_exclusive_pairs={
            ("DX_MALE_ONLY", "DX_FEMALE_ONLY"),
        },
    )

    rules = [
        DemographicRule(
            rule_id="sex_check",
            sex_to_forbidden_codes={
                "M": {"DX_PREG"},
            },
        ),
        RequiredCodesRule(rule_id="required_support"),
        MutualExclusionRule(rule_id="mutual_exclusion"),
    ]

    return OntologyEngine(index=index, rules=rules)


def test_demographic_mismatch_detected() -> None:
    engine = build_toy_engine()
    record = ClinicalRecord(record_id="r1", codes=("DX_PREG",), sex="M")

    score, violations = engine.score_violations(record)

    assert score == 1.0
    assert len(violations) == 1
    assert violations[0].kind == "demographic_mismatch"


def test_missing_required_code_detected() -> None:
    engine = build_toy_engine()
    record = ClinicalRecord(record_id="r2", codes=("RX_ABX",), sex="F")

    score, violations = engine.score_violations(record)

    assert score == 1.0
    assert len(violations) == 1
    assert violations[0].kind == "missing_required_code"


def test_mutual_exclusion_detected() -> None:
    engine = build_toy_engine()
    record = ClinicalRecord(
        record_id="r3",
        codes=("DX_MALE_ONLY", "DX_FEMALE_ONLY"),
        sex="F",
    )

    score, violations = engine.score_violations(record)

    assert score == 1.0
    assert len(violations) == 1
    assert violations[0].kind == "mutual_exclusion"


def test_get_replacements_returns_siblings_and_parents() -> None:
    engine = build_toy_engine()

    replacements = engine.get_replacements("DX_HEART_FAILURE", top_k=10)

    assert "DX_CARDIOMYOPATHY" in replacements
    assert "DX_ARRHYTHMIA" in replacements
    assert "PARENT_CARDIAC" in replacements


def test_no_violation_for_supported_antibiotic_case() -> None:
    engine = build_toy_engine()
    record = ClinicalRecord(record_id="r4", codes=("RX_ABX", "DX_INF"), sex="F")

    score, violations = engine.score_violations(record)

    assert score == 0.0
    assert violations == []
