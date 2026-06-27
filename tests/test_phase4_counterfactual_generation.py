"""Phase 4 -- functional tests for counterfactual generation per repair family."""

from __future__ import annotations

from src.explanations.counterfactual import generate_counterfactual
from src.ontology.engine import OntologyEngine
from src.ontology.index import OntologyIndex
from src.ontology.rule_loader import build_real_ontology_rules
from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights


def _scorer() -> OntologyAwareScorer:
    index = OntologyIndex()
    rules, _ = build_real_ontology_rules(index)
    engine = OntologyEngine(index=index, rules=rules)
    return OntologyAwareScorer(
        ontology_mode="real",
        engine=engine,
        weights=ScoreWeights(),
        require_assets=False,
    )


def _gen(record, **kw):
    scorer = _scorer()
    return generate_counterfactual(record, scorer, scorer.engine.index, **kw)


# --------------------------------------------------------------------------
# Demographic
# --------------------------------------------------------------------------


def test_demographic_repair_removes_conflict_and_is_minimal() -> None:
    rec = {
        "model_visible_sequence": ["DX_10_O80", "DX_10_I10", "MED_ASPIRIN"],
        "gender": "M",
        "age_group": "30-44",
    }
    res = _gen(rec, max_edits=3)
    assert res.validity is True
    assert res.num_edits == 1  # minimal
    assert any(
        e.operation == "remove" and e.token_before == "DX_10_O80" for e in res.edits
    )
    assert res.s_ont_after < res.s_ont_before
    assert not any(
        v["kind"] == "demographic_mismatch" for v in res.rule_violations_after
    )


def test_demographic_repair_uses_only_model_visible() -> None:
    # prostate code in a female -> flagged and removed (keep a 2nd token so the
    # repaired record is non-empty and therefore valid).
    res = _gen(
        {"model_visible_sequence": ["DX_10_N40", "DX_10_J18"], "gender": "F"},
        max_edits=2,
    )
    assert res.validity is True
    assert any(
        e.operation == "remove" and e.token_before == "DX_10_N40" for e in res.edits
    )
    assert res.repaired_codes == ["DX_10_J18"]


def test_demographic_repair_empty_record_is_invalid() -> None:
    # if the ONLY token is the conflict, removing it empties the record -> the
    # conservative validity check rejects it (documented failure mode).
    res = _gen({"model_visible_sequence": ["DX_10_N40"], "gender": "F"}, max_edits=2)
    assert res.validity is False
    assert res.failure_reason == "empty_record"


# --------------------------------------------------------------------------
# Medication
# --------------------------------------------------------------------------


def test_medication_repair_or_clear_failure() -> None:
    rec = {"model_visible_sequence": ["MED_INSULIN", "DX_10_I10"], "gender": "F"}
    res = _gen(rec, max_edits=3)
    assert res.validity is True
    # either remove the unsupported drug or add a curated context dx
    ops = {e.operation for e in res.edits}
    assert ops & {"remove", "add"}
    assert res.delta_s_ont > 0


def test_medication_satisfied_record_is_not_flagged() -> None:
    # insulin WITH a diabetes context concept present -> insulin justified -> no
    # violation. (Use a canonical SNOMED diabetes token so it is recognized even
    # without a crosswalk in this minimal test index.)
    rec = {
        "model_visible_sequence": ["MED_INSULIN", "SNOMED:73211009"],
        "gender": "F",
    }
    res = _gen(rec)
    assert res.status == "no_violation"
    assert res.validity is False
    assert res.failure_reason == "no_ontology_violation_to_repair"


# --------------------------------------------------------------------------
# Forbidden co-occurrence
# --------------------------------------------------------------------------


def test_forbidden_repair_reduces_violations_without_new_ones() -> None:
    rec = {
        "model_visible_sequence": ["DX_10_E119", "DX_10_E10_9", "DX_10_I10"],
        "gender": "M",
        "age_group": "65-79",
    }
    res = _gen(rec, max_edits=3)
    assert res.validity is True
    assert res.num_edits == 1  # remove one side of the pair
    assert len(res.rule_violations_after) < len(res.rule_violations_before)
    assert not any(v["kind"] == "mutual_exclusion" for v in res.rule_violations_after)


def test_forbidden_does_not_use_hidden_to_pick_side() -> None:
    # The generator must pick a side by score reduction, deterministically.
    rec = {"model_visible_sequence": ["DX_10_E119", "DX_10_E10_9"], "gender": "M"}
    a = _gen(rec).to_dict()
    b = _gen(rec).to_dict()
    assert a["edits"] == b["edits"]  # deterministic


# --------------------------------------------------------------------------
# Determinism / budget / failure modes
# --------------------------------------------------------------------------


def test_determinism_with_seed() -> None:
    rec = {
        "model_visible_sequence": ["DX_10_O80", "DX_10_E119", "DX_10_E10_9"],
        "gender": "M",
    }
    a = _gen(rec, seed=42).to_dict()
    b = _gen(rec, seed=42).to_dict()
    assert a == b


def test_clean_record_reports_no_violation() -> None:
    res = _gen({"model_visible_sequence": ["DX_10_I10", "MED_ASPIRIN"], "gender": "F"})
    assert res.status == "no_violation"
    assert res.num_edits == 0


def test_edit_budget_is_respected() -> None:
    rec = {
        "model_visible_sequence": ["DX_10_O80", "DX_10_E119", "DX_10_E10_9"],
        "gender": "M",
    }
    res = _gen(rec, max_edits=2)
    assert res.num_edits <= 2
