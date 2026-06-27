"""
scripts/build_ontology_coverage_report.py
==========================================
Phase 2 -- Ontology mapping coverage report.

Estimates how many EHR tokens can be mapped to a real medical ontology
(SNOMED via ICD crosswalks, RxNorm via drug-name maps). If the licensed/processed
ontology assets are absent, the script does NOT fabricate coverage: it emits a
truthful ``status = "blocked_missing_real_ontology_assets"`` report listing the
missing files.

Usage
-----
    python scripts/build_ontology_coverage_report.py
    python scripts/build_ontology_coverage_report.py \\
        --ontology-dir ontologies/processed \\
        --ehr data/processed/mimiciv_val.pkl

Outputs
-------
    artifacts/phase2/coverage_report.json
    artifacts/phase2/unmapped_codes.csv
    artifacts/phase2/crosswalk_sanity.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONTOLOGY_DIR = PROJECT_ROOT / "ontologies" / "processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "phase2"

# Files the loader/coverage can consume (produced by the existing parsers).
EXPECTED_ONTOLOGY_FILES = (
    "snomed_hierarchy.json",
    "snomed_terms.json",
    "icd9_to_snomed.json",
    "icd10_to_snomed.json",
    "drugname_to_rxcui.json",
)

# Candidate processed EHR datasets to measure coverage against (first existing).
DEFAULT_EHR_CANDIDATES = (
    "data/processed/mimiciv_val.pkl",
    "data/processed/mimiciv_test.pkl",
    "data/processed/mimiciv_sequences.parquet",
)

DIAG_PREFIXES = ("DX_", "DIAG_", "ICD")
PROC_PREFIXES = ("PROC_", "CPT_")
MED_PREFIXES = ("MED_", "RX", "NDC_", "DRUG_", "RAW_DRUG")


def classify(token: str) -> str:
    up = token.upper()
    if up.startswith(DIAG_PREFIXES):
        return "diagnosis"
    if up.startswith(PROC_PREFIXES):
        return "procedure"
    if up.startswith(MED_PREFIXES):
        return "medication"
    return "other"


def find_present_and_missing(ontology_dir: Path) -> tuple[list[str], list[str]]:
    present, missing = [], []
    for fname in EXPECTED_ONTOLOGY_FILES:
        (present if (ontology_dir / fname).exists() else missing).append(fname)
    return present, missing


def resolve_ehr(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p if p.exists() else None
    for cand in DEFAULT_EHR_CANDIDATES:
        p = PROJECT_ROOT / cand
        if p.exists():
            return p
    return None


def write_outputs(
    out_dir: Path,
    report: dict[str, Any],
    unmapped_rows: list[dict[str, Any]],
    sanity_md: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )
    with (out_dir / "unmapped_codes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["token", "namespace", "count"])
        writer.writeheader()
        for row in unmapped_rows:
            writer.writerow(row)
    (out_dir / "crosswalk_sanity.md").write_text(sanity_md, encoding="utf-8")


def blocked_report(
    ontology_dir: Path, missing: list[str], present: list[str]
) -> dict[str, Any]:
    return {
        "phase": 2,
        "report": "ontology_coverage",
        "status": "blocked_missing_real_ontology_assets",
        "ontology_dir": str(ontology_dir.relative_to(PROJECT_ROOT))
        if ontology_dir.is_relative_to(PROJECT_ROOT)
        else str(ontology_dir),
        "present_ontology_files": present,
        "missing_ontology_files": missing,
        "coverage": None,
        "note": (
            "No real/processed ontology crosswalk assets were found. Coverage is NOT "
            "computed and NOT fabricated. Acquire UMLS/SNOMED/RxNorm and run the existing "
            "parsers (scripts/parse_snomed.py, src/preprocessing/build_umls_maps.py, "
            "scripts/parse_rxnorm.py, src/preprocessing/build_rxnorm_maps.py) to populate "
            f"{ontology_dir} with: {', '.join(EXPECTED_ONTOLOGY_FILES)}."
        ),
    }


def parse_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return []


def compute_real_coverage(
    ontology_dir: Path,
    ehr_path: Path,
    max_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pandas as pd

    icd_map: dict[str, list[str]] = {}
    for fname in ("icd9_to_snomed.json", "icd10_to_snomed.json"):
        fp = ontology_dir / fname
        if fp.exists():
            icd_map.update(json.loads(fp.read_text(encoding="utf-8")))
    drug_map: dict[str, str] = {}
    drug_fp = ontology_dir / "drugname_to_rxcui.json"
    if drug_fp.exists():
        drug_map = json.loads(drug_fp.read_text(encoding="utf-8"))
    ingredient_set: set[str] = set()
    ing_fp = ontology_dir / "rxnorm_ingredient_index.json"
    if ing_fp.exists():
        ingredient_set = set(json.loads(ing_fp.read_text(encoding="utf-8")).keys())

    if ehr_path.suffix == ".parquet":
        df = pd.read_parquet(ehr_path)
    else:
        df = pd.read_pickle(ehr_path)
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    seq_col = next(
        (
            c
            for c in ("codes", "sequence_tokens", "tokens", "sequence")
            if c in df.columns
        ),
        None,
    )
    if seq_col is None:
        raise RuntimeError(
            f"No sequence column in EHR file. Columns: {list(df.columns)}"
        )

    counts = {
        "diagnosis": Counter(),
        "procedure": Counter(),
        "medication": Counter(),
        "other": Counter(),
    }
    mapped = {"diagnosis": 0, "medication": 0}
    totals = {"diagnosis": 0, "procedure": 0, "medication": 0, "other": 0}
    unmapped: Counter = Counter()

    # Use the hardened normalization util (ICD-9/10 dotting, drug-name keys).
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.preprocessing.code_normalization import (
        match_medication_ingredient,
        normalize_diagnosis_token,
        normalize_drug_token,
    )

    def med_is_mapped(tok: str) -> bool:
        if any(c in drug_map for c in normalize_drug_token(tok)):
            return True
        if ingredient_set and match_medication_ingredient(tok, ingredient_set):
            return True
        return False

    for toks in df[seq_col].tolist():
        for tok in parse_tokens(toks):
            ns = classify(tok)
            totals[ns] += 1
            counts[ns][tok] += 1
            if ns == "diagnosis":
                if any(c in icd_map for c in normalize_diagnosis_token(tok)):
                    mapped["diagnosis"] += 1
                else:
                    unmapped[tok] += 1
            elif ns == "medication":
                if med_is_mapped(tok):
                    mapped["medication"] += 1
                else:
                    unmapped[tok] += 1

    def rate(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    coverage = {
        "rows_scanned": int(len(df)),
        "token_totals": totals,
        "diagnosis_mapped": mapped["diagnosis"],
        "diagnosis_coverage": rate(mapped["diagnosis"], totals["diagnosis"]),
        "medication_mapped": mapped["medication"],
        "medication_coverage": rate(mapped["medication"], totals["medication"]),
        "procedure_coverage": "not_mapped_in_phase2",
        "unique_unmapped_tokens": len(unmapped),
    }
    unmapped_rows = [
        {"token": tok, "namespace": classify(tok), "count": c}
        for tok, c in unmapped.most_common(500)
    ]
    return coverage, unmapped_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 ontology coverage report.")
    parser.add_argument(
        "--ontology-dir",
        default=None,
        help="Processed ontology dir (default ontologies/processed).",
    )
    parser.add_argument(
        "--ehr",
        default=None,
        help="Processed EHR file (default: first existing candidate).",
    )
    parser.add_argument(
        "--out-dir", default=None, help="Output dir (default artifacts/phase2)."
    )
    parser.add_argument(
        "--max-rows", type=int, default=20000, help="Max EHR rows to scan (0 = all)."
    )
    args = parser.parse_args(argv)

    ontology_dir = (
        Path(args.ontology_dir) if args.ontology_dir else DEFAULT_ONTOLOGY_DIR
    )
    if not ontology_dir.is_absolute():
        ontology_dir = PROJECT_ROOT / ontology_dir
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    present, missing = find_present_and_missing(ontology_dir)
    # Need at least one crosswalk to compute anything meaningful.
    have_crosswalks = any(
        f in present
        for f in (
            "icd9_to_snomed.json",
            "icd10_to_snomed.json",
            "drugname_to_rxcui.json",
        )
    )

    if not have_crosswalks:
        report = blocked_report(ontology_dir, missing, present)
        sanity = (
            "# Crosswalk Sanity Check\n\n"
            "**Status: BLOCKED — no real ontology crosswalk assets found.**\n\n"
            f"Searched: `{ontology_dir}`\n\n"
            f"- Present: {present or 'none'}\n"
            f"- Missing: {missing}\n\n"
            "Coverage was not computed and was not fabricated. Acquire licensed UMLS/SNOMED/"
            "RxNorm data and run the existing parsers to populate the processed ontology "
            "directory, then re-run this script.\n"
        )
        write_outputs(out_dir, report, [], sanity)
        print(f"[coverage] BLOCKED: missing crosswalk assets in {ontology_dir}")
        print(f"[coverage] Missing: {missing}")
        print(f"[coverage] Wrote {(out_dir / 'coverage_report.json')}")
        return 0

    ehr_path = resolve_ehr(args.ehr)
    if ehr_path is None:
        report = {
            "phase": 2,
            "report": "ontology_coverage",
            "status": "blocked_missing_ehr_dataset",
            "ontology_dir": str(ontology_dir),
            "present_ontology_files": present,
            "missing_ontology_files": missing,
            "note": "Ontology crosswalks present, but no processed EHR dataset found to measure coverage.",
        }
        write_outputs(
            out_dir,
            report,
            [],
            "# Crosswalk Sanity Check\n\nBLOCKED: no EHR dataset found.\n",
        )
        print("[coverage] BLOCKED: ontology present but no EHR dataset found.")
        return 0

    coverage, unmapped_rows = compute_real_coverage(
        ontology_dir, ehr_path, args.max_rows
    )
    report = {
        "phase": 2,
        "report": "ontology_coverage",
        "status": "computed_real_coverage",
        "ontology_dir": str(ontology_dir),
        "ehr_file": str(ehr_path.relative_to(PROJECT_ROOT))
        if ehr_path.is_relative_to(PROJECT_ROOT)
        else str(ehr_path),
        "present_ontology_files": present,
        "missing_ontology_files": missing,
        "coverage": coverage,
        "thresholds": {"diagnosis_target": 0.80, "medication_target": 0.70},
        "meets_thresholds": {
            "diagnosis": coverage["diagnosis_coverage"] >= 0.80,
            "medication": coverage["medication_coverage"] >= 0.70,
        },
    }
    sanity = (
        "# Crosswalk Sanity Check\n\n"
        f"Status: **computed real coverage** on `{report['ehr_file']}`.\n\n"
        f"- Diagnosis coverage: {coverage['diagnosis_coverage']:.4f} "
        f"({coverage['diagnosis_mapped']}/{coverage['token_totals']['diagnosis']})\n"
        f"- Medication coverage: {coverage['medication_coverage']:.4f} "
        f"({coverage['medication_mapped']}/{coverage['token_totals']['medication']})\n"
        f"- Unique unmapped tokens: {coverage['unique_unmapped_tokens']} "
        "(see unmapped_codes.csv)\n"
    )
    write_outputs(out_dir, report, unmapped_rows, sanity)
    print(f"[coverage] Computed real coverage on {report['ehr_file']}")
    print(
        f"[coverage] diagnosis={coverage['diagnosis_coverage']} medication={coverage['medication_coverage']}"
    )
    print(f"[coverage] Wrote {(out_dir / 'coverage_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
