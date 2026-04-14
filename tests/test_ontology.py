from __future__ import annotations

import json
from pathlib import Path

from src.ontology import (
    ClinicalRecord,
    DemographicRule,
    MutualExclusionRule,
    OntologyEngine,
    OntologyIndex,
    RequiredCodesRule,
    load_ontology_engine,
    load_ontology_index,
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


# ---------------------------------------------------------------------------
# Tests for loader.py
# ---------------------------------------------------------------------------


def _write_mock_files(tmp_path: Path) -> Path:
    """Create small mock SNOMED hierarchy and terms files for loader tests."""
    # Hierarchy: pregnancy root 77386006 has child 72892002 (normal pregnancy)
    # Diabetes: 44054006 (type 2), 46635009 (type 1) -- for mutual exclusion
    hierarchy = {
        "parents": {
            "72892002": ["77386006"],
            "44054006": ["73211009"],
            "46635009": ["73211009"],
        },
        "children": {
            "77386006": ["72892002"],
            "73211009": ["44054006", "46635009"],
        },
    }
    (tmp_path / "snomed_hierarchy.json").write_text(
        json.dumps(hierarchy), encoding="utf-8",
    )

    terms = {
        "77386006": "Pregnant",
        "72892002": "Normal pregnancy",
        "73211009": "Diabetes mellitus",
        "44054006": "Type 2 diabetes mellitus",
        "46635009": "Type 1 diabetes mellitus",
    }
    (tmp_path / "snomed_terms.json").write_text(
        json.dumps(terms), encoding="utf-8",
    )

    return tmp_path


def test_loader_index_populates_hierarchy(tmp_path: Path) -> None:
    data_dir = _write_mock_files(tmp_path)
    index = load_ontology_index(data_dir)

    # Parents and children should be SNOMED-prefixed
    assert "SNOMED:77386006" in index.get_parents("SNOMED:72892002")
    assert "SNOMED:72892002" in index.get_children("SNOMED:77386006")


def test_loader_index_populates_terms(tmp_path: Path) -> None:
    data_dir = _write_mock_files(tmp_path)
    index = load_ontology_index(data_dir)

    assert index.get_term("SNOMED:77386006") == "Pregnant"
    assert index.get_term("SNOMED:44054006") == "Type 2 diabetes mellitus"
    # Unknown code falls back to the code itself
    assert index.get_term("SNOMED:9999999") == "SNOMED:9999999"


def test_loader_with_mock_files(tmp_path: Path) -> None:
    data_dir = _write_mock_files(tmp_path)
    engine = load_ontology_engine(data_dir)

    # 1. Pregnancy code in a male patient should trigger demographic violation
    record_preg_male = ClinicalRecord(
        record_id="t1",
        codes=("SNOMED:72892002",),  # normal pregnancy (descendant of 77386006)
        sex="M",
    )
    score, violations = engine.score_violations(record_preg_male)
    assert score > 0
    kinds = [v.kind for v in violations]
    assert "demographic_mismatch" in kinds

    # 2. Pregnancy code in a female patient should be fine
    record_preg_female = ClinicalRecord(
        record_id="t2",
        codes=("SNOMED:72892002",),
        sex="F",
    )
    score_f, violations_f = engine.score_violations(record_preg_female)
    assert score_f == 0.0
    assert violations_f == []

    # 3. Insulin without diabetes should trigger missing-required-code
    record_insulin_no_dx = ClinicalRecord(
        record_id="t3",
        codes=("RXNORM:5856",),  # insulin
        sex="F",
    )
    score_ins, violations_ins = engine.score_violations(record_insulin_no_dx)
    assert score_ins > 0
    kinds_ins = [v.kind for v in violations_ins]
    assert "missing_required_code" in kinds_ins

    # 4. Insulin with diabetes present should be fine
    record_insulin_with_dx = ClinicalRecord(
        record_id="t4",
        codes=("RXNORM:5856", "SNOMED:44054006"),  # insulin + DM type 2
        sex="F",
    )
    score_ok, violations_ok = engine.score_violations(record_insulin_with_dx)
    missing_violations = [v for v in violations_ok if v.kind == "missing_required_code"]
    assert len(missing_violations) == 0

    # 5. Type 1 + Type 2 diabetes should trigger mutual exclusion
    record_dm_both = ClinicalRecord(
        record_id="t5",
        codes=("SNOMED:46635009", "SNOMED:44054006"),
        sex="F",
    )
    score_mx, violations_mx = engine.score_violations(record_dm_both)
    assert score_mx > 0
    kinds_mx = [v.kind for v in violations_mx]
    assert "mutual_exclusion" in kinds_mx


def test_loader_missing_files_returns_empty_engine(tmp_path: Path) -> None:
    """Loader should not crash when hierarchy/terms files are absent."""
    engine = load_ontology_engine(tmp_path)
    assert engine.index.preferred_terms == {}
    assert engine.index.parents == {}
    record = ClinicalRecord(record_id="empty", codes=(), sex="F")
    score, violations = engine.score_violations(record)
    assert score == 0.0
