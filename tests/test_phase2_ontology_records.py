"""Phase 2 -- tests for the canonical ontology data model (records.py)."""

from __future__ import annotations

from src.ontology import (
    ClinicalRecord,
    OntologyConcept,
    OntologyRuleResult,
    OntologyViolation,
)


def test_package_imports_cleanly() -> None:
    # If this import + the module-level imports above succeed, the previously
    # broken src.ontology surface is fixed.
    import src.ontology as o

    for name in (
        "ClinicalRecord",
        "OntologyIndex",
        "OntologyEngine",
        "DemographicRule",
    ):
        assert hasattr(o, name)


def test_clinical_record_basic_fields() -> None:
    rec = ClinicalRecord(
        record_id="r1",
        codes=("DX_10_A", "MED_X"),
        sex="M",
        age_group="65-79",
        subject_id="123",
        hadm_id="456",
    )
    assert rec.codes == ("DX_10_A", "MED_X")
    assert rec.sex == "M"
    assert rec.gender == "M"  # alias
    assert rec.age_group == "65-79"
    assert rec.subject_id == "123"
    assert rec.hadm_id == "456"


def test_clinical_record_from_mapping() -> None:
    row = {
        "subject_id": 123,
        "hadm_id": 456,
        "codes": ["DX_10_A", "MED_X"],
        "gender": "F",
        "age_group": "18-29",
    }
    rec = ClinicalRecord.from_mapping(row)
    assert rec.codes == ("DX_10_A", "MED_X")
    assert rec.sex == "F"
    assert rec.record_id == "123_456"
    assert rec.subject_id == "123"


def test_ontology_concept() -> None:
    c = OntologyConcept(
        concept_id="SNOMED:73211009",
        preferred_term="Diabetes mellitus",
        children=("SNOMED:44054006",),
    )
    assert c.concept_id == "SNOMED:73211009"
    assert "SNOMED:44054006" in c.children


def test_violation_to_dict() -> None:
    v = OntologyViolation(
        rule_id="r",
        kind="demographic_mismatch",
        message="m",
        codes=("DX_PREG",),
        severity=2.0,
    )
    d = v.to_dict()
    assert d["kind"] == "demographic_mismatch"
    assert d["codes"] == ["DX_PREG"]
    assert d["severity"] == 2.0


def test_rule_result_score_and_passed() -> None:
    empty = OntologyRuleResult(record_id="r0", violations=())
    assert empty.passed is True
    assert empty.score == 0.0

    vlist = (
        OntologyViolation("r", "demographic_mismatch", "m", ("a",), 1.0),
        OntologyViolation("r", "mutual_exclusion", "m", ("b", "c"), 1.5),
    )
    res = OntologyRuleResult(record_id="r1", violations=vlist)
    assert res.passed is False
    assert res.score == 2.5
    assert res.kinds() == ["demographic_mismatch", "mutual_exclusion"]
