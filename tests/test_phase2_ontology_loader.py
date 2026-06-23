"""Phase 2 -- tests for the ontology loader using synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

from src.ontology import ClinicalRecord, load_ontology_engine, load_ontology_index

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ontology"


def test_loader_index_from_fixtures() -> None:
    index = load_ontology_index(FIXTURE_DIR)
    # hierarchy prefixed with SNOMED:
    assert "SNOMED:77386006" in index.get_parents("SNOMED:72892002")
    assert "SNOMED:72892002" in index.get_children("SNOMED:77386006")
    # terms
    assert index.get_term("SNOMED:73211009") == "Diabetes mellitus"
    # crosswalk maps loaded from fixtures
    assert index.map_icd_to_snomed("250.00") == ["SNOMED:44054006"]
    assert index.map_drug_to_rxcui("INSULIN") == "5856"


def test_loader_engine_demographic_rule_fixture() -> None:
    engine = load_ontology_engine(FIXTURE_DIR)
    # normal pregnancy (descendant of pregnancy root) in a male -> demographic mismatch
    rec = ClinicalRecord(record_id="t1", codes=("SNOMED:72892002",), sex="M")
    score, violations = engine.score_violations(rec)
    assert score > 0
    assert "demographic_mismatch" in [v.kind for v in violations]

    # same code in a female -> no violation
    rec_f = ClinicalRecord(record_id="t2", codes=("SNOMED:72892002",), sex="F")
    score_f, violations_f = engine.score_violations(rec_f)
    assert score_f == 0.0
    assert violations_f == []


def test_loader_engine_required_and_mutual_exclusion_fixture() -> None:
    engine = load_ontology_engine(FIXTURE_DIR)

    # insulin (RXNORM:5856) without any diabetes dx -> missing_required_code
    rec_ins = ClinicalRecord(record_id="t3", codes=("RXNORM:5856",), sex="F")
    score_ins, violations_ins = engine.score_violations(rec_ins)
    assert "missing_required_code" in [v.kind for v in violations_ins]

    # insulin + type-2 diabetes (descendant of diabetes root) -> satisfied
    rec_ok = ClinicalRecord(
        record_id="t4", codes=("RXNORM:5856", "SNOMED:44054006"), sex="F"
    )
    _, violations_ok = engine.score_violations(rec_ok)
    assert "missing_required_code" not in [v.kind for v in violations_ok]

    # type-1 + type-2 diabetes -> mutual exclusion
    rec_mx = ClinicalRecord(
        record_id="t5", codes=("SNOMED:46635009", "SNOMED:44054006"), sex="F"
    )
    _, violations_mx = engine.score_violations(rec_mx)
    assert "mutual_exclusion" in [v.kind for v in violations_mx]


def test_loader_missing_assets_is_safe(tmp_path: Path) -> None:
    """Loader must not crash when no ontology files are present."""
    engine = load_ontology_engine(tmp_path)
    assert engine.index.preferred_terms == {}
    assert engine.index.parents == {}
    rec = ClinicalRecord(record_id="empty", codes=(), sex="F")
    score, violations = engine.score_violations(rec)
    assert score == 0.0
    assert violations == []
