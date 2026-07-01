"""Phase 7 -- final evaluation artifact + invariant checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P7 = ROOT / "artifacts" / "phase7"


def _load(name: str) -> dict:
    p = P7 / name
    if not p.exists():
        pytest.skip(
            f"{name} not generated yet (run scripts/run_phase7_final_evaluation.py)"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def test_final_evaluation_has_required_fields() -> None:
    d = _load("final_evaluation.json")
    for f in ("variant_metrics", "strongest_variant", "score_equation", "answers"):
        assert f in d
    variants = {v["variant"] for v in d["variant_metrics"]}
    assert "ontology_only_real" in variants
    assert "legacy_baseline" in variants


def test_sgen_excluded_from_core() -> None:
    d = _load("final_evaluation.json")
    assert d["sgen_included_in_core"] is False
    assert d["w_gen"] == 0.0
    assert "w_gen = 0" in d["score_equation"] or "w_gen=0" in d["score_equation"]


def test_old_circular_benchmark_not_final_evidence() -> None:
    d = _load("final_evaluation.json")
    assert d["old_circular_benchmark_used_as_final_evidence"] is False
    assert "benchmark-v2" in d["benchmark"]


def test_variants_have_bootstrap_cis() -> None:
    d = _load("final_evaluation.json")
    for v in d["variant_metrics"]:
        assert isinstance(v["roc_auc_ci"], list) and len(v["roc_auc_ci"]) == 2


def test_statistical_tests_cover_core_comparisons() -> None:
    d = _load("final_stat_tests.json")
    pairs = {(t["a"], t["b"]) for t in d["paired_bootstrap_roc_auc"]}
    assert ("ontology_only_real", "legacy_baseline") in pairs


def test_no_per_record_dumps_committed_outside_ignored() -> None:
    # any per-record files must live under an ignored dir
    for p in P7.rglob("per_record*"):
        assert "ignored" in p.parts, f"per-record file outside ignored/: {p}"
