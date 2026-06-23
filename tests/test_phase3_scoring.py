"""Phase 3 -- tests for explicit real/legacy/disabled ontology-aware scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ontology import OntologyEngine, OntologyIndex
from src.ontology.rule_engine import DemographicRule
from src.scoring.ontology_aware import (
    OntologyAwareScorer,
    ScoreWeights,
    compute_calibrated_score,
    map_tokens_to_ontology_codes,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ontology"


def test_calibrated_score_equation_explicit() -> None:
    # ontology excluded -> S_cal == S_det (no dilution by an absent component)
    assert compute_calibrated_score(0.4, 0.0, include_ont=False) == pytest.approx(0.4)
    # with ontology active, higher S_ont raises the score (monotonic)
    assert compute_calibrated_score(0.4, 2.0) > compute_calibrated_score(0.4, 0.0)


def test_w_gen_zero_by_default_ignores_sgen() -> None:
    with_gen = compute_calibrated_score(0.4, 0.0, s_gen=0.9)
    without = compute_calibrated_score(0.4, 0.0, s_gen=None)
    assert with_gen == pytest.approx(without)  # w_gen=0 -> s_gen ignored


def test_w_gen_used_only_when_positive_weight() -> None:
    w = ScoreWeights(w_det=0.5, w_ont=0.3, w_gen=0.2)
    base = compute_calibrated_score(0.4, 0.0, s_gen=None, weights=w)
    with_gen = compute_calibrated_score(0.4, 0.0, s_gen=1.0, weights=w)
    assert with_gen > base


def test_disabled_mode_zero_ont() -> None:
    scorer = OntologyAwareScorer(ontology_mode="disabled")
    out = scorer.score(
        {"sequence_tokens": ["DX_10_O0900", "MED_X"], "gender": "male"}, s_det=0.5
    )
    assert out["s_ont"] == 0.0
    assert out["ontology_status"] == "disabled"
    assert out["s_cal"] == pytest.approx(0.5)
    assert out["s_gen"] is None
    assert out["weights"]["w_gen"] == 0.0


def test_legacy_mode_uses_compute_s_ont() -> None:
    scorer = OntologyAwareScorer(ontology_mode="legacy")
    out = scorer.score(
        {"sequence_tokens": ["MED_12345", "RXNORM_999"], "gender": "male"}, s_det=0.5
    )
    assert out["ontology_status"] == "ok_legacy"
    assert out["s_ont"] > 0  # missing-diagnosis-for-medication fires


def test_real_mode_requires_engine() -> None:
    with pytest.raises(ValueError):
        OntologyAwareScorer(ontology_mode="real")  # no engine, require_assets default


def test_real_mode_no_silent_fallback_when_unavailable() -> None:
    # require_assets=False -> structured unavailable, NOT legacy results
    scorer = OntologyAwareScorer(
        ontology_mode="real", engine=None, require_assets=False
    )
    out = scorer.score({"sequence_tokens": ["MED_12345"], "gender": "male"}, s_det=0.5)
    assert out["ontology_status"] == "ontology_unavailable"
    assert out["s_ont"] == 0.0


def test_real_mode_scores_with_engine_and_mapping() -> None:
    # Build a tiny real-style engine: ICD->SNOMED maps a code to a forbidden
    # pregnancy concept; demographic rule flags it for a male.
    index = OntologyIndex(
        parents={},
        children={},
        icd_to_snomed={"O09.90": ["SNOMED:72892002"]},
    )
    rules = [
        DemographicRule(
            rule_id="sex", sex_to_forbidden_codes={"M": {"SNOMED:72892002"}}
        )
    ]
    engine = OntologyEngine(index=index, rules=rules)
    scorer = OntologyAwareScorer(ontology_mode="real", engine=engine)

    # DX_10_O0990 -> normalize -> "O09.90" -> SNOMED:72892002 (forbidden for male)
    out = scorer.score({"sequence_tokens": ["DX_10_O0990"], "gender": "M"}, s_det=0.5)
    assert out["ontology_status"] == "ok"
    assert out["s_ont"] > 0
    assert any(v["kind"] == "demographic_mismatch" for v in out["violations"])
    assert out["mapping"]["n_mapped"] == 1


def test_map_tokens_to_ontology_codes() -> None:
    index = OntologyIndex(
        icd_to_snomed={"401.9": ["SNOMED:38341003"]},
        drug_to_rxcui={"ACETAMINOPHEN": "161"},
    )
    res = map_tokens_to_ontology_codes(
        ["DX_9_4019", "MED_ACETAMINOPHEN", "DX_9_99999"], index
    )
    assert "SNOMED:38341003" in res["mapped_codes"]
    assert "RXNORM:161" in res["mapped_codes"]
    assert "DX_9_99999" in res["unmapped_tokens"]
