"""Phase 8 -- reproducibility package + data-safety checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REPRO_DOCS = [
    "REPRODUCIBILITY.md",
    "docs/reproducibility/environment.md",
    "docs/reproducibility/data_access.md",
    "docs/reproducibility/runbook.md",
    "docs/reproducibility/artifact_manifest.md",
    "docs/reproducibility/phase8_reproducibility_guide.md",
]

PHASE8 = ROOT / "artifacts" / "phase8"


@pytest.mark.parametrize("rel", REPRO_DOCS)
def test_reproducibility_doc_exists(rel: str) -> None:
    p = ROOT / rel
    assert p.exists(), f"missing {rel}"
    assert len(p.read_text(encoding="utf-8").strip()) > 100


def test_artifact_manifest_exists() -> None:
    assert (PHASE8 / "artifact_manifest.json").exists()
    assert (PHASE8 / "artifact_manifest.md").exists()


def test_artifact_manifest_json_valid() -> None:
    d = json.loads((PHASE8 / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert d["phase"] == 8
    assert "excluded_from_git" in d and d["excluded_from_git"]
    assert any("phase7" in p.get("phase", "") for p in d["phases"])


def test_phase8_summary_valid() -> None:
    p = PHASE8 / "phase8_summary.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["status"] == "phase8_complete_external_validation_deferred"
    assert d["sgen_in_core"] is False
    # preserved main results
    assert d["main_results_preserved"]["ontology_only_real"]["roc_auc"] == 0.7881


def test_paper_asset_index_exists() -> None:
    assert (PHASE8 / "paper_asset_index.json").exists()
    assert (PHASE8 / "paper_asset_index.md").exists()


def test_gitignore_protects_restricted() -> None:
    txt = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ["data/processed/", "*.pkl", "*.pt", "*.parquet", "per_record"]:
        assert pattern in txt, f".gitignore missing {pattern}"


def test_no_per_record_dumps_in_phase8() -> None:
    """No Phase 8 committed artifact may be a per-record / heavy score dump."""
    assert PHASE8.exists()
    bad_substrings = ("per_record", "checkpoint")
    bad_suffixes = (".pkl", ".pt", ".pth", ".parquet", ".zip")
    offenders = []
    for f in PHASE8.rglob("*"):
        if f.is_dir():
            continue
        name = f.name.lower()
        if any(s in name for s in bad_substrings) or name.endswith(bad_suffixes):
            offenders.append(str(f.relative_to(ROOT)))
    assert not offenders, (
        f"forbidden per-record/heavy files under artifacts/phase8: {offenders}"
    )


def test_phase8_artifacts_are_small_text() -> None:
    """Committed Phase 8 artifacts should be small aggregate text (json/md), not blobs."""
    for f in PHASE8.rglob("*"):
        if f.is_dir():
            continue
        assert f.suffix.lower() in {".json", ".md"}, (
            f"unexpected artifact type: {f.name}"
        )
        assert f.stat().st_size < 200_000, (
            f"{f.name} unexpectedly large for an aggregate artifact"
        )
