"""Phase 8 -- exercise the finalization check script itself."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase8_final_checks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_phase8_final_checks", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_phase8_final_checks"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_exists() -> None:
    assert SCRIPT.exists(), "scripts/run_phase8_final_checks.py must exist"


def test_all_checks_pass() -> None:
    mod = _load_module()
    results = mod.run_all_checks()
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"failed checks: {failed}"


def test_main_returns_zero() -> None:
    mod = _load_module()
    assert mod.main() == 0


def test_expected_checks_present() -> None:
    mod = _load_module()
    names = {r.name for r in mod.run_all_checks()}
    for expected in [
        "phase7_artifacts_present",
        "readme_not_stale",
        "final_manuscript_present",
        "reproducibility_docs_present",
        "final_claims_matrix_present",
        "sgen_not_core",
        "eicu_not_validated",
        "result_table_values_present",
        "gitignore_protects_restricted",
        "no_per_record_in_phase8",
    ]:
        assert expected in names, f"missing check: {expected}"
