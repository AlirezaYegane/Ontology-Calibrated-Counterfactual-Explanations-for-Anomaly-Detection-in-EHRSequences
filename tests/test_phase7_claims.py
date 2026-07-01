"""Phase 7 -- final claims-decision completeness + validity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P7 = ROOT / "artifacts" / "phase7"

VALID_STATUSES = {
    "supported_now",
    "partially_supported",
    "unsupported",
    "future_work",
    "removed_from_core",
}
REQUIRED_CLAIMS = {
    "benchmark_v2_non_circular",
    "real_ontology_integration",
    "real_ontology_beats_legacy",
    "detector_improves_detection",
    "combined_beats_ontology_only",
    "sgen_improves_detection",
    "counterfactual_leakage_free",
    "counterfactual_effective_for_flagged",
    "clinical_validity_external",
    "external_dataset_generalization",
    "reproducible",
}


def _claims() -> dict:
    p = P7 / "final_claims_decision.json"
    if not p.exists():
        pytest.skip("final_claims_decision.json not generated yet")
    return json.loads(p.read_text(encoding="utf-8"))


def test_all_required_claims_present() -> None:
    d = _claims()
    claims = d["claims"]
    for c in REQUIRED_CLAIMS:
        assert c in claims, c


def test_all_claim_statuses_valid() -> None:
    d = _claims()
    for name, c in d["claims"].items():
        assert c["status"] in VALID_STATUSES, f"{name}: {c['status']}"
        assert c.get("evidence"), f"{name} missing evidence"


def test_unsupported_claims_not_upgraded() -> None:
    d = _claims()
    # blunt honesty: detector-additive and Sgen must NOT be marked supported
    assert d["claims"]["detector_improves_detection"]["status"] in (
        "unsupported",
        "future_work",
    )
    assert d["claims"]["combined_beats_ontology_only"]["status"] == "unsupported"
    assert d["claims"]["sgen_improves_detection"]["status"] == "removed_from_core"
    # real ontology beats legacy must be supported (Phase 7 confirmed, p<0.05)
    assert d["claims"]["real_ontology_beats_legacy"]["status"] == "supported_now"
