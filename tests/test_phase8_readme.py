"""Phase 8 -- README finalization checks.

Verifies the humanized README carries the final result table, states the MIMIC-IV-only
scope, and honestly reports the negative / blocked results (detector, Sgen, eICU).
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    assert README.exists(), "README.md must exist"
    return README.read_text(encoding="utf-8")


def test_readme_has_final_result_table(readme: str) -> None:
    # headline ROC-AUC values for ontology-only and legacy must be present
    assert "0.7881" in readme, "ontology_only_real ROC-AUC missing"
    assert "0.7358" in readme, "legacy_baseline ROC-AUC missing"
    assert "ontology_only_real" in readme
    assert "legacy_baseline" in readme


def test_readme_states_mimic_iv_only(readme: str) -> None:
    low = readme.lower()
    assert "mimic-iv" in low
    assert "benchmark-v2" in low
    # explicit "only" scoping somewhere in the README
    assert "mimic-iv only" in low or "mimic-iv benchmark-v2 only" in low


def test_readme_states_eicu_blocked(readme: str) -> None:
    low = readme.lower()
    assert "eicu" in low
    assert "blocked" in low
    assert "schema mismatch" in low


def test_readme_states_sgen_excluded(readme: str) -> None:
    low = readme.lower()
    assert (
        ("w_gen = 0" in low) or ("removed from the core" in low) or ("excluded" in low)
    )
    assert "sgen" in low or "generative" in low


def test_readme_reports_detector_negative(readme: str) -> None:
    low = readme.lower()
    assert "0.4525" in readme  # detector below-chance ROC-AUC
    assert "below chance" in low or "non-additive" in low or "negative" in low


def test_readme_not_stale(readme: str) -> None:
    assert "Days 1-14 of the 90-day roadmap" not in readme
    assert "### 6. Train detector (planned)" not in readme


def test_readme_has_reviewer_note(readme: str) -> None:
    low = readme.lower()
    assert "for reviewers" in low
    assert "aggregate" in low
