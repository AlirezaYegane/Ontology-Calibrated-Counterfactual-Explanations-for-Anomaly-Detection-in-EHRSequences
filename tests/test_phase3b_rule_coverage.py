"""Phase 3b -- coverage behavior of the real rule packs per anomaly family.

Deterministic end-to-end checks on an engine built from the curated rule packs
(no real ontology files needed: the seeded SNOMED roots + source-token families
make every family's behavior reproducible). Verifies that each anomaly family
fires and that a clean normal record does not.
"""

from __future__ import annotations

from src.ontology.engine import OntologyEngine
from src.ontology.index import OntologyIndex
from src.ontology.records import ClinicalRecord
from src.ontology.rule_loader import build_real_ontology_rules


def _engine() -> OntologyEngine:
    index = OntologyIndex()
    rules, _ = build_real_ontology_rules(index)
    return OntologyEngine(index=index, rules=rules)


def _kinds(engine: OntologyEngine, rec: ClinicalRecord) -> list[str]:
    _, violations = engine.score_violations(rec)
    return [v.kind for v in violations]


def test_clean_normal_record_does_not_fire() -> None:
    engine = _engine()
    # Female, single diabetes type (T2), warfarin justified by atrial fibrillation
    # (SNOMED:49436004 is a seeded anticoagulant-context root).
    normal = ClinicalRecord(
        codes=("RXNORM:11289", "SNOMED:49436004"),
        sex="F",
        source_tokens=("DX_10_E119", "DX_10_I48", "MED_WARFARIN"),
    )
    assert _kinds(engine, normal) == []


def test_demographic_family_fires() -> None:
    engine = _engine()
    male_pregnancy = ClinicalRecord(
        codes=(), sex="M", source_tokens=("DX_10_O80", "DX_10_I10")
    )
    assert "demographic_mismatch" in _kinds(engine, male_pregnancy)
    female_prostate = ClinicalRecord(codes=(), sex="F", source_tokens=("DX_10_N40",))
    assert "demographic_mismatch" in _kinds(engine, female_prostate)


def test_forbidden_cooccurrence_family_fires() -> None:
    engine = _engine()
    # type-2 patient with an added type-1 code (note the non-standard E10_9 body
    # that fails to map but is still caught via the source-token family path).
    both_types = ClinicalRecord(
        codes=(), sex="M", source_tokens=("DX_10_E119", "DX_10_E10_9")
    )
    assert "mutual_exclusion" in _kinds(engine, both_types)


def test_medication_family_fires_and_is_satisfied_with_context() -> None:
    engine = _engine()
    # warfarin with NO thromboembolic context -> fires
    no_context = ClinicalRecord(
        codes=("RXNORM:11289",), sex="F", source_tokens=("MED_WARFARIN", "DX_10_I10")
    )
    assert "missing_required_code" in _kinds(engine, no_context)
    # insulin (unmapped drug) with NO diabetes context -> fires via name path
    insulin_no_dx = ClinicalRecord(
        codes=(), sex="F", source_tokens=("MED_INSULIN", "DX_10_I10")
    )
    assert "missing_required_code" in _kinds(engine, insulin_no_dx)
    # insulin WITH diabetes context (seeded root) -> satisfied
    insulin_with_dx = ClinicalRecord(
        codes=("SNOMED:73211009",), sex="F", source_tokens=("MED_INSULIN",)
    )
    assert "missing_required_code" not in _kinds(engine, insulin_with_dx)


def test_normal_violation_rate_low_on_synthetic_mix() -> None:
    """A batch of clean normal records should have a LOW violation rate."""
    engine = _engine()
    normals = [
        ClinicalRecord(codes=(), sex="F", source_tokens=("DX_10_I10", "DX_10_E119")),
        ClinicalRecord(codes=(), sex="M", source_tokens=("DX_10_J18", "DX_10_N18")),
        ClinicalRecord(codes=(), sex="F", source_tokens=("DX_10_K219",)),
        ClinicalRecord(codes=(), sex="M", source_tokens=("DX_10_I2510",)),
    ]
    fired = sum(1 for r in normals if _kinds(engine, r))
    assert fired == 0
