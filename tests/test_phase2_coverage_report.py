"""Phase 2 -- tests for the ontology coverage-report script.

Verifies the script runs and reports truthfully when assets are missing, without
fabricating coverage numbers.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_ontology_coverage_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_ontology_coverage_report", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocked_when_no_assets(tmp_path: Path) -> None:
    mod = _load_module()
    ontology_dir = tmp_path / "ontology_empty"
    ontology_dir.mkdir()
    out_dir = tmp_path / "out"

    rc = mod.main(["--ontology-dir", str(ontology_dir), "--out-dir", str(out_dir)])
    assert rc == 0

    report = json.loads((out_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked_missing_real_ontology_assets"
    assert report["coverage"] is None
    assert "snomed_hierarchy.json" in report["missing_ontology_files"]
    # truthful artifacts exist
    assert (out_dir / "unmapped_codes.csv").exists()
    assert (out_dir / "crosswalk_sanity.md").exists()


def test_blocked_when_crosswalks_present_but_no_ehr(tmp_path: Path) -> None:
    mod = _load_module()
    # Point at fixture crosswalks but give no EHR dataset.
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "ontology"
    out_dir = tmp_path / "out2"
    rc = mod.main(
        [
            "--ontology-dir",
            str(fixture_dir),
            "--ehr",
            str(tmp_path / "does_not_exist.pkl"),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked_missing_ehr_dataset"
