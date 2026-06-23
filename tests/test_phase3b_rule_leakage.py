"""Phase 3b -- leakage guards for the ontology rule engine.

Verifies the rules score ONLY on model-visible clinical content + demographics
and never on benchmark answer keys (label / anomaly_type / hidden_eval_metadata /
audit_metadata / bad_code / expected_code / replacement_code).
"""

from __future__ import annotations

from src.ontology.engine import OntologyEngine
from src.ontology.index import OntologyIndex
from src.ontology.records import ClinicalRecord
from src.ontology.rule_loader import build_real_ontology_rules
from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

LEAKAGE_FIELDS = {
    "label": 1,
    "anomaly_type": "demographic_incompatibility",
    "hidden_eval_metadata": {
        "original_gender": "F",
        "removed_indication_codes": ["DX_10_E11"],
    },
    "audit_metadata": {"method": "gender_flip"},
    "bad_code": "DX_10_O80",
    "expected_code": "DX_10_N40",
    "replacement_code": "DX_10_E11",
    "original_gender": "F",
}


def _real_scorer() -> OntologyAwareScorer:
    index = OntologyIndex()
    rules, _ = build_real_ontology_rules(index)
    engine = OntologyEngine(index=index, rules=rules)
    return OntologyAwareScorer(
        ontology_mode="real",
        engine=engine,
        weights=ScoreWeights(),
        require_assets=False,
    )


def test_scorer_ignores_answer_key_columns() -> None:
    scorer = _real_scorer()
    clean = {"codes": ["DX_10_O80"], "gender": "M"}
    leaky = {**clean, **LEAKAGE_FIELDS}
    s_clean = scorer.score(clean, s_det=0.0)["s_ont"]
    s_leaky = scorer.score(leaky, s_det=0.0)["s_ont"]
    # Adding every answer-key column must not change the ontology score.
    assert s_clean == s_leaky
    assert s_clean > 0  # the male+pregnancy contradiction is still detected


def test_clinical_record_from_mapping_does_not_ingest_answer_keys() -> None:
    row = {"codes": ["DX_10_O80"], "gender": "M", **LEAKAGE_FIELDS}
    rec = ClinicalRecord.from_mapping(row)
    flat = repr(rec).lower()
    for bad in (
        "anomaly_type",
        "hidden_eval",
        "audit_metadata",
        "expected_code",
        "replacement_code",
    ):
        assert bad not in flat
    # only model-visible content survives
    assert rec.sex == "M"


def test_demographic_rule_uses_gender_and_codes_only() -> None:
    scorer = _real_scorer()
    # Same codes, different gender -> different result (proves it uses gender,
    # not a hidden flag).
    male = scorer.score({"codes": ["DX_10_O80"], "gender": "M"}, s_det=0.0)["s_ont"]
    female = scorer.score({"codes": ["DX_10_O80"], "gender": "F"}, s_det=0.0)["s_ont"]
    assert male > 0
    assert female == 0.0


def test_hidden_metadata_does_not_drive_score() -> None:
    scorer = _real_scorer()
    # A record whose hidden metadata SAYS 'anomaly' but whose model-visible
    # content is clean must score 0 (not driven by the hidden label).
    row = {
        "codes": ["DX_10_I10", "MED_METOPROLOL"],
        "gender": "F",
        "label": 1,
        "anomaly_type": "forbidden_cooccurrence",
        "hidden_eval_metadata": {"added_code": "DX_10_E10_9"},
    }
    assert scorer.score(row, s_det=0.0)["s_ont"] == 0.0


def test_forbidden_columns_constant_is_enforced_by_injector_guard() -> None:
    # The benchmark builder guards model-visible fields; confirm the answer-key
    # column names we test here are in that forbidden set.
    from src.preprocessing.anomaly_injection_v2 import FORBIDDEN_MODEL_VISIBLE_COLUMNS

    for col in ("bad_code", "expected_code", "replacement_code", "original_gender"):
        assert col in FORBIDDEN_MODEL_VISIBLE_COLUMNS
