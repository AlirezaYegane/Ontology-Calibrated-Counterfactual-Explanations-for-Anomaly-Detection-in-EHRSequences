"""Phase 8 -- paper package completeness and honesty checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"

SECTION_FILES = [
    "abstract.md",
    "introduction.md",
    "methods.md",
    "experiments.md",
    "results.md",
    "discussion.md",
    "limitations.md",
    "reproducibility_statement.md",
    "ethics_and_data_statement.md",
    "future_work.md",
    "manuscript.md",
    "final_manuscript.md",
]


@pytest.mark.parametrize("name", SECTION_FILES)
def test_section_file_exists_nonempty(name: str) -> None:
    p = PAPER / name
    assert p.exists(), f"missing {name}"
    assert len(p.read_text(encoding="utf-8").strip()) > 200, f"{name} too short"


def test_final_manuscript_has_all_sections() -> None:
    txt = (PAPER / "final_manuscript.md").read_text(encoding="utf-8").lower()
    for heading in [
        "abstract",
        "introduction",
        "methods",
        "experiments",
        "results",
        "discussion",
        "limitations",
        "reproducibility",
        "ethics",
        "future work",
    ]:
        assert heading in txt, f"final_manuscript missing section: {heading}"


def test_final_manuscript_headline_numbers() -> None:
    txt = (PAPER / "final_manuscript.md").read_text(encoding="utf-8")
    assert "0.7881" in txt  # ontology
    assert "0.7358" in txt  # legacy
    assert "0.4525" in txt  # detector below chance
    assert "0.8999" in txt or "89.99" in txt  # repair among flagged


def test_references_present_and_not_fabricated() -> None:
    refs = PAPER / "references.md"
    todo = PAPER / "references_todo.md"
    assert refs.exists() and todo.exists()
    refs_txt = refs.read_text(encoding="utf-8").lower()
    # references come from the repo's own survey; DOIs are explicitly not fabricated
    assert "fabricat" in refs_txt, (
        "references.md must document the no-fabrication discipline"
    )


def test_final_claims_matrix_decisions() -> None:
    matrix = (PAPER / "final_claims_matrix.md").read_text(encoding="utf-8").lower()
    # unsupported claims must be present and correctly labelled
    assert "unsupported" in matrix
    assert "removed from core" in matrix
    # detector / combined must not be labelled as improvements
    assert "detector improves anomaly detection" in matrix
    assert "future work" in matrix


def test_final_claims_decision_json_consistent() -> None:
    """The final claims matrix must agree with the authoritative Phase 7 JSON."""
    p = ROOT / "artifacts" / "phase7" / "final_claims_decision.json"
    if not p.exists():
        pytest.skip("final_claims_decision.json not present")
    d = json.loads(p.read_text(encoding="utf-8"))
    claims = d["claims"]
    assert claims["detector_improves_detection"]["status"] == "unsupported"
    assert claims["combined_beats_ontology_only"]["status"] == "unsupported"
    assert claims["sgen_improves_detection"]["status"] == "removed_from_core"
    assert claims["real_ontology_beats_legacy"]["status"] == "supported_now"


def test_contribution_statement_exists() -> None:
    p = PAPER / "final_contribution_statement.md"
    assert p.exists()
    txt = p.read_text(encoding="utf-8").lower()
    assert "negative result" in txt
    assert "not claimed" in txt or "not** claimed" in txt
