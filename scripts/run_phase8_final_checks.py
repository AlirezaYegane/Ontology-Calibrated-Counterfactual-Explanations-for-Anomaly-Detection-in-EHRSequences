"""Phase 8 -- lightweight repository finalization checks.

Verifies that the Phase 8 deliverables (humanized README, final manuscript,
reproducibility package, final claims matrix) are present and internally consistent,
that no overclaims remain in the README / final docs, that ``.gitignore`` protects the
restricted patterns, and that no per-record score dumps are committed under
``artifacts/phase8``.

This is a *documentation/consistency* check, not a scientific rerun. It runs on CPU with
no restricted data. Exit code 0 means all checks passed; 1 means at least one failed.

Importable: ``run_all_checks()`` returns a list of ``CheckResult`` for the test suite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- expected deliverables -------------------------------------------------------------

REQUIRED_PHASE7_ARTIFACTS = [
    "artifacts/phase7/final_evaluation.json",
    "artifacts/phase7/final_stat_tests.json",
    "artifacts/phase7/ablation_results.json",
    "artifacts/phase7/counterfactual_final.json",
    "artifacts/phase7/external_validation_status.json",
    "artifacts/phase7/final_claims_decision.json",
    "artifacts/phase7/tables/table2_main_results.csv",
]

REQUIRED_REPRODUCIBILITY_DOCS = [
    "REPRODUCIBILITY.md",
    "docs/reproducibility/environment.md",
    "docs/reproducibility/data_access.md",
    "docs/reproducibility/runbook.md",
    "docs/reproducibility/artifact_manifest.md",
    "docs/reproducibility/phase8_reproducibility_guide.md",
]

FINAL_MANUSCRIPT = "docs/paper/final_manuscript.md"
FINAL_CLAIMS_MATRIX = "docs/paper/final_claims_matrix.md"

# Headline numbers that must survive into the README and manuscript.
ONTOLOGY_AUC = "0.7881"
LEGACY_AUC = "0.7358"

# .gitignore must protect at least these restricted patterns.
REQUIRED_GITIGNORE_PATTERNS = [
    "data/processed/",
    "*.pkl",
    "*.pt",
    "*.parquet",
    "per_record",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


# --- individual checks -----------------------------------------------------------------


def check_phase7_artifacts() -> CheckResult:
    missing = [a for a in REQUIRED_PHASE7_ARTIFACTS if not _exists(a)]
    return CheckResult(
        "phase7_artifacts_present",
        not missing,
        "all present" if not missing else f"missing: {missing}",
    )


def check_readme_not_stale() -> CheckResult:
    txt = _read("README.md")
    if not txt:
        return CheckResult("readme_not_stale", False, "README.md missing")
    stale_markers = [
        "Days 1-14 of the 90-day roadmap",
        "### 6. Train detector (planned)",
    ]
    hits = [m for m in stale_markers if m in txt]
    return CheckResult(
        "readme_not_stale",
        not hits,
        "no stale markers" if not hits else f"stale markers found: {hits}",
    )


def check_final_manuscript() -> CheckResult:
    return CheckResult(
        "final_manuscript_present",
        _exists(FINAL_MANUSCRIPT),
        FINAL_MANUSCRIPT
        if _exists(FINAL_MANUSCRIPT)
        else f"missing {FINAL_MANUSCRIPT}",
    )


def check_reproducibility_docs() -> CheckResult:
    missing = [d for d in REQUIRED_REPRODUCIBILITY_DOCS if not _exists(d)]
    return CheckResult(
        "reproducibility_docs_present",
        not missing,
        "all present" if not missing else f"missing: {missing}",
    )


def check_claims_matrix() -> CheckResult:
    return CheckResult(
        "final_claims_matrix_present",
        _exists(FINAL_CLAIMS_MATRIX),
        FINAL_CLAIMS_MATRIX
        if _exists(FINAL_CLAIMS_MATRIX)
        else f"missing {FINAL_CLAIMS_MATRIX}",
    )


def check_sgen_not_core() -> CheckResult:
    """README + final manuscript must state Sgen is excluded (w_gen = 0), not core."""
    readme = _read("README.md").lower()
    manu = _read(FINAL_MANUSCRIPT).lower()
    ok = True
    detail = []
    for name, txt in (("README", readme), ("manuscript", manu)):
        excluded = (
            ("w_gen = 0" in txt)
            or ("removed from the core" in txt)
            or ("excluded" in txt)
        )
        if not excluded:
            ok = False
            detail.append(f"{name} does not state Sgen exclusion")
    return CheckResult(
        "sgen_not_core",
        ok,
        "Sgen documented as excluded (w_gen=0)" if ok else "; ".join(detail),
    )


def check_eicu_not_validated() -> CheckResult:
    """README + final manuscript must say eICU external validation is blocked."""
    readme = _read("README.md").lower()
    manu = _read(FINAL_MANUSCRIPT).lower()
    ok = True
    detail = []
    for name, txt in (("README", readme), ("manuscript", manu)):
        if "blocked" not in txt or "eicu" not in txt:
            ok = False
            detail.append(f"{name} does not state eICU blocked")
    return CheckResult(
        "eicu_not_validated",
        ok,
        "eICU documented as blocked" if ok else "; ".join(detail),
    )


def check_result_table_values() -> CheckResult:
    """The headline ROC-AUC values must appear in README and final manuscript."""
    ok = True
    detail = []
    for rel in ("README.md", FINAL_MANUSCRIPT):
        txt = _read(rel)
        for val in (ONTOLOGY_AUC, LEGACY_AUC):
            if val not in txt:
                ok = False
                detail.append(f"{rel} missing {val}")
    return CheckResult(
        "result_table_values_present",
        ok,
        f"{ONTOLOGY_AUC}/{LEGACY_AUC} present" if ok else "; ".join(detail),
    )


def check_gitignore_patterns() -> CheckResult:
    txt = _read(".gitignore")
    missing = [p for p in REQUIRED_GITIGNORE_PATTERNS if p not in txt]
    return CheckResult(
        "gitignore_protects_restricted",
        not missing,
        "all patterns present" if not missing else f"missing patterns: {missing}",
    )


def check_no_per_record_in_phase8() -> CheckResult:
    """No committed-style per-record / heavy artifacts under artifacts/phase8."""
    phase8 = ROOT / "artifacts" / "phase8"
    if not phase8.exists():
        return CheckResult("no_per_record_in_phase8", False, "artifacts/phase8 missing")
    bad_substrings = ("per_record", "checkpoint")
    bad_suffixes = (".pkl", ".pt", ".pth", ".parquet", ".zip")
    offenders = []
    for f in phase8.rglob("*"):
        if f.is_dir():
            continue
        name = f.name.lower()
        if any(s in name for s in bad_substrings) or name.endswith(bad_suffixes):
            offenders.append(str(f.relative_to(ROOT)))
    return CheckResult(
        "no_per_record_in_phase8",
        not offenders,
        "clean (aggregate only)" if not offenders else f"forbidden files: {offenders}",
    )


CHECKS = [
    check_phase7_artifacts,
    check_readme_not_stale,
    check_final_manuscript,
    check_reproducibility_docs,
    check_claims_matrix,
    check_sgen_not_core,
    check_eicu_not_validated,
    check_result_table_values,
    check_gitignore_patterns,
    check_no_per_record_in_phase8,
]


def run_all_checks() -> list[CheckResult]:
    return [c() for c in CHECKS]


def main() -> int:
    results = run_all_checks()
    print("Phase 8 final checks")
    print("=" * 60)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
    n_pass = sum(1 for r in results if r.passed)
    print("=" * 60)
    print(f"{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
