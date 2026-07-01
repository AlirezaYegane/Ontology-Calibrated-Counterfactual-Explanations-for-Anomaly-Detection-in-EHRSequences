"""Phase 7 -- table/figure generation artifact checks."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "phase7" / "tables"
FIGS = ROOT / "artifacts" / "phase7" / "figures"

EXPECTED_TABLES = [
    "table1_dataset_summary.csv",
    "table2_main_results.csv",
    "table3_ablation_results.csv",
    "table5_statistical_tests.csv",
]


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet (run scripts/run_phase7_tables.py)")


def test_expected_tables_exist_and_have_headers() -> None:
    for name in EXPECTED_TABLES:
        p = TABLES / name
        _require(p)
        rows = list(csv.reader(p.open(encoding="utf-8")))
        assert len(rows) >= 2, f"{name} has no data rows"
        assert len(rows[0]) >= 2, f"{name} header malformed"


def test_main_results_table_has_ontology_variant() -> None:
    p = TABLES / "table2_main_results.csv"
    _require(p)
    text = p.read_text(encoding="utf-8")
    assert "ontology_only_real" in text


def test_figure_data_present() -> None:
    _require(FIGS / "fig2_main_auc_bar.csv")
    _require(FIGS / "fig1_pipeline_summary.json")


def test_tables_have_no_raw_tokens() -> None:
    # aggregate tables must not embed raw MIMIC tokens
    for name in EXPECTED_TABLES:
        p = TABLES / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8").upper()
        for forbidden in ("DX_10_", "DX_9_", "MED_", "SUBJECT_ID"):
            assert forbidden not in text, f"{name} contains {forbidden}"
