"""Phase 7 -- counterfactual-final artifact checks + generator ablation flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P7 = ROOT / "artifacts" / "phase7"

REQUIRED_FIELDS = (
    "repair_attempted_count",
    "ontology_flagged_count",
    "repair_success_count",
    "repair_success_rate_overall",
    "repair_success_rate_among_flagged",
    "success_by_anomaly_type",
    "success_by_rule_type",
    "mean_delta_s_ont",
    "mean_num_edits",
    "median_num_edits",
    "edit_type_distribution",
    "failure_reason_counts",
    "validity_rate",
    "edit_strategy_ablation",
)


def test_counterfactual_final_has_required_fields() -> None:
    p = P7 / "counterfactual_final.json"
    if not p.exists():
        pytest.skip("counterfactual_final.json not generated yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    for f in REQUIRED_FIELDS:
        assert f in d, f


def test_edit_strategy_ablation_has_policies() -> None:
    p = P7 / "counterfactual_final.json"
    if not p.exists():
        pytest.skip("counterfactual_final.json not generated yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    for policy in ("remove_only", "replace_only", "add_context_allowed", "full_policy"):
        assert policy in d["edit_strategy_ablation"], policy


def test_allowed_operations_filter_restricts_edits() -> None:
    # the generator's allowed_operations must actually restrict edit types.
    from src.explanations.counterfactual import generate_counterfactual
    from src.ontology.engine import OntologyEngine
    from src.ontology.index import OntologyIndex
    from src.ontology.rule_loader import build_real_ontology_rules
    from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

    index = OntologyIndex()
    rules, _ = build_real_ontology_rules(index)
    scorer = OntologyAwareScorer(
        ontology_mode="real",
        engine=OntologyEngine(index=index, rules=rules),
        weights=ScoreWeights(),
        require_assets=False,
    )
    rec = {"model_visible_sequence": ["DX_10_O80", "DX_10_I10"], "gender": "M"}
    res = generate_counterfactual(
        rec, scorer, index, max_edits=3, allowed_operations=("remove",)
    )
    assert all(e.operation == "remove" for e in res.edits)
    # restricting to replace-only should not fabricate remove edits
    res2 = generate_counterfactual(
        rec, scorer, index, max_edits=3, allowed_operations=("replace",)
    )
    assert all(e.operation == "replace" for e in res2.edits)
