"""Phase 7 -- ablation artifact checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P7 = ROOT / "artifacts" / "phase7"


def _abl() -> dict:
    p = P7 / "ablation_results.json"
    if not p.exists():
        pytest.skip("ablation_results.json not generated yet")
    return json.loads(p.read_text(encoding="utf-8"))


def test_ontology_ablation_has_required_variants() -> None:
    d = _abl()
    variants = {v["variant"] for v in d["ontology_rule_ablation"]}
    for required in (
        "real_ontology_rules_full",
        "legacy_icd_prefix_rules",
        "demographic_rules_only",
        "medication_rules_only",
        "forbidden_cooccurrence_rules_only",
        "ontology_disabled",
    ):
        assert required in variants, required


def test_component_ablation_excludes_sgen_from_core() -> None:
    d = _abl()
    comps = {c["variant"]: c for c in d["score_component_ablation"]}
    assert (
        "S_ont_only" in comps and "S_det_only" in comps and "S_ont_plus_S_det" in comps
    )
    sgen_row = comps.get("S_ont_plus_S_det_plus_Sgen")
    assert sgen_row is not None
    assert sgen_row.get("status") == "EXCLUDED_FROM_CORE"
    assert sgen_row.get("roc_auc") is None  # not scored into the core


def test_full_ruleset_beats_single_families() -> None:
    d = _abl()
    by = {v["variant"]: v for v in d["ontology_rule_ablation"]}
    full = by["real_ontology_rules_full"]["roc_auc"]
    for fam in (
        "demographic_rules_only",
        "medication_rules_only",
        "forbidden_cooccurrence_rules_only",
    ):
        assert full >= by[fam]["roc_auc"], f"full should synergize over {fam}"


def test_per_anomaly_family_breakdown_present() -> None:
    d = _abl()
    full = next(
        v
        for v in d["ontology_rule_ablation"]
        if v["variant"] == "real_ontology_rules_full"
    )
    fa = full["per_anomaly_family_roc_auc"]
    for fam in (
        "demographic_incompatibility",
        "medication_indication_mismatch",
        "forbidden_cooccurrence",
    ):
        assert fam in fa
