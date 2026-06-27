"""Phase 4 -- smoke test for the evaluation harness on a tiny synthetic fixture."""

from __future__ import annotations

import json

from src.ontology.engine import OntologyEngine
from src.ontology.index import OntologyIndex
from src.ontology.rule_loader import build_real_ontology_rules
from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

from scripts.run_phase4_counterfactual_eval import evaluate_records

# A tiny synthetic "benchmark" of anomalous rows (model-visible + analysis label).
FIXTURE = [
    {
        "model_visible_sequence": ["DX_10_O80", "DX_10_I10"],
        "gender": "M",
        "age_group": "30-44",
        "label": 1,
        "anomaly_type": "demographic_incompatibility",
        "hidden_eval_metadata": {"original_gender": "F"},  # must never reach generator
    },
    {
        "model_visible_sequence": ["DX_10_E119", "DX_10_E10_9"],
        "gender": "M",
        "age_group": "65-79",
        "label": 1,
        "anomaly_type": "forbidden_cooccurrence",
        "audit_metadata": {"added_code": "DX_10_E10_9"},
    },
    {
        "model_visible_sequence": ["MED_INSULIN", "DX_10_I10"],
        "gender": "F",
        "age_group": "80+",
        "label": 1,
        "anomaly_type": "medication_indication_mismatch",
    },
]


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


def test_eval_writes_outputs_and_metrics(tmp_path) -> None:
    scorer = _scorer()
    summary = evaluate_records(
        FIXTURE,
        scorer,
        scorer.engine.index,
        detector=None,
        detector_status="disabled",
        split="fixture",
        seed=42,
        max_edits=3,
        beam_size=10,
        max_records=0,
        out_dir=tmp_path,
    )

    # files created
    for fname in (
        "counterfactual_results.jsonl",
        "counterfactual_summary.json",
        "counterfactual_summary.md",
        "edit_type_breakdown.csv",
        "failure_cases.jsonl",
    ):
        assert (tmp_path / fname).exists(), fname

    # required metric fields exist
    for field in (
        "repair_attempted_count",
        "repair_success_count",
        "repair_success_rate",
        "repair_success_rate_among_flagged",
        "mean_delta_s_ont",
        "mean_num_edits",
        "median_num_edits",
        "mean_ontology_distance",
        "validity_rate",
        "failure_reason_counts",
        "success_by_anomaly_type",
        "success_by_rule_type",
    ):
        assert field in summary, field

    assert summary["repair_attempted_count"] == 3
    # all three flagged anomalies should be repaired in this clean fixture
    assert summary["repair_success_count"] == 3


def test_eval_outputs_contain_no_hidden_metadata(tmp_path) -> None:
    scorer = _scorer()
    evaluate_records(
        FIXTURE, scorer, scorer.engine.index, out_dir=tmp_path, max_records=0
    )
    results = (
        (tmp_path / "counterfactual_results.jsonl").read_text(encoding="utf-8").lower()
    )
    for forbidden in (
        "hidden_eval_metadata",
        "audit_metadata",
        "original_gender",
        "added_code",
    ):
        assert forbidden not in results


def test_eval_results_rows_are_wellformed(tmp_path) -> None:
    scorer = _scorer()
    evaluate_records(
        FIXTURE, scorer, scorer.engine.index, out_dir=tmp_path, max_records=0
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "counterfactual_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 3
    for r in rows:
        assert "status" in r and "edits" in r and "s_ont_before" in r
        assert "anomaly_type" in r  # analysis bucket field
