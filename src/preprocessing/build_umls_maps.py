"""
src/preprocessing/build_umls_maps.py
=====================================
Day 8 -- Build ICD-to-SNOMED crosswalk dictionaries from UMLS MRCONSO.

Reads MRCONSO.RRF (pipe-delimited) and produces:

  * icd9_to_snomed.json   -- ICD-9-CM code -> list of SNOMED CT concept IDs
  * icd10_to_snomed.json  -- ICD-10-CM code -> list of SNOMED CT concept IDs
  * snomed_terms.json     -- SNOMED CT concept ID -> preferred term string

The mapping strategy: for each CUI, collect source codes from ICD9CM,
ICD10CM, and SNOMEDCT_US.  Where a CUI has both an ICD code and a SNOMED
code, create a cross-mapping entry.

CLI::

    python -m src.preprocessing.build_umls_maps \\
        --mrconso ontologies/umls/MRCONSO.RRF \\
        --output-dir ontologies/umls_maps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# MRCONSO.RRF columns (pipe-delimited, no header)
_MRCONSO_COLS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
    "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
    "STR", "SRL", "SUPPRESS", "CVF",
]

_SAB_ICD9 = "ICD9CM"
_SAB_ICD10 = "ICD10CM"
_SAB_SNOMED = "SNOMEDCT_US"

_SABS_OF_INTEREST = frozenset({_SAB_ICD9, _SAB_ICD10, _SAB_SNOMED})


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_mrconso(mrconso_path: Path) -> pd.DataFrame:
    """Load MRCONSO.RRF filtered to English rows in ICD9CM, ICD10CM, SNOMEDCT_US."""
    log.info("Loading %s", mrconso_path)
    df = pd.read_csv(
        mrconso_path, sep="|", header=None, names=_MRCONSO_COLS,
        dtype=str, index_col=False, low_memory=False,
    )
    # English only, relevant SABs, non-suppressed
    mask = (
        (df["LAT"] == "ENG")
        & df["SAB"].isin(_SABS_OF_INTEREST)
        & (df["SUPPRESS"] != "O")
    )
    filtered = df.loc[mask, ["CUI", "SAB", "CODE", "STR", "TTY"]].copy()
    log.info(
        "Filtered to %d rows across %s",
        len(filtered), sorted(_SABS_OF_INTEREST),
    )
    return filtered


def build_crosswalks(
    mrconso: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    """Build ICD-9->SNOMED, ICD-10->SNOMED maps and a SNOMED term lookup.

    Returns (icd9_to_snomed, icd10_to_snomed, snomed_terms).
    """
    # Group codes by CUI
    cui_to_icd9: dict[str, set[str]] = defaultdict(set)
    cui_to_icd10: dict[str, set[str]] = defaultdict(set)
    cui_to_snomed: dict[str, set[str]] = defaultdict(set)
    snomed_terms: dict[str, str] = {}

    for row in mrconso.itertuples(index=False):
        cui = row.CUI
        sab = row.SAB
        code = row.CODE

        if sab == _SAB_ICD9:
            cui_to_icd9[cui].add(code)
        elif sab == _SAB_ICD10:
            cui_to_icd10[cui].add(code)
        elif sab == _SAB_SNOMED:
            cui_to_snomed[cui].add(code)
            if code not in snomed_terms:
                snomed_terms[code] = row.STR

    # Build crosswalks via shared CUI
    icd9_to_snomed: dict[str, list[str]] = defaultdict(list)
    icd10_to_snomed: dict[str, list[str]] = defaultdict(list)

    for cui in cui_to_snomed:
        snomed_ids = sorted(cui_to_snomed[cui])
        for icd9 in cui_to_icd9.get(cui, []):
            icd9_to_snomed[icd9].extend(snomed_ids)
        for icd10 in cui_to_icd10.get(cui, []):
            icd10_to_snomed[icd10].extend(snomed_ids)

    # De-duplicate
    icd9_map = {k: sorted(set(v)) for k, v in icd9_to_snomed.items()}
    icd10_map = {k: sorted(set(v)) for k, v in icd10_to_snomed.items()}

    log.info(
        "Crosswalks: ICD-9->SNOMED %d codes, ICD-10->SNOMED %d codes, SNOMED terms %d",
        len(icd9_map), len(icd10_map), len(snomed_terms),
    )
    return icd9_map, icd10_map, snomed_terms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 8 -- Build ICD-to-SNOMED crosswalk from UMLS MRCONSO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mrconso", required=True, type=Path,
        help="Path to MRCONSO.RRF",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for JSON mapping files",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %s (%d entries)", path.name, len(data))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mrconso = load_mrconso(args.mrconso)
    icd9_map, icd10_map, snomed_terms = build_crosswalks(mrconso)

    _save_json(icd9_map, args.output_dir / "icd9_to_snomed.json")
    _save_json(icd10_map, args.output_dir / "icd10_to_snomed.json")
    _save_json(snomed_terms, args.output_dir / "snomed_terms.json")

    log.info("Done.")


if __name__ == "__main__":
    main()
