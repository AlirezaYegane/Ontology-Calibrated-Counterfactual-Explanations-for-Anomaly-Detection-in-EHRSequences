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
from collections import defaultdict
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# MRCONSO.RRF columns (pipe-delimited, no header)
_MRCONSO_COLS = [
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
]

_SAB_ICD9 = "ICD9CM"
_SAB_ICD10 = "ICD10CM"
_SAB_SNOMED = "SNOMEDCT_US"

_SABS_OF_INTEREST = frozenset({_SAB_ICD9, _SAB_ICD10, _SAB_SNOMED})


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_mrconso(mrconso_path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Load MRCONSO.RRF filtered to English rows in ICD9CM, ICD10CM, SNOMEDCT_US.

    MRCONSO.RRF can be very large (multiple GB for a full UMLS release), so this
    reads in chunks and keeps only the relevant rows/columns to bound memory.
    The signature/return contract (a filtered DataFrame with columns
    CUI/SAB/CODE/STR/TTY) is unchanged.
    """
    log.info("Loading %s (chunked)", mrconso_path)
    keep_cols = ["CUI", "SAB", "CODE", "STR", "TTY"]
    parts: list[pd.DataFrame] = []
    total = 0
    reader = pd.read_csv(
        mrconso_path,
        sep="|",
        header=None,
        names=_MRCONSO_COLS,
        dtype=str,
        index_col=False,
        low_memory=False,
        usecols=["CUI", "LAT", "SAB", "CODE", "STR", "TTY", "SUPPRESS"],
        chunksize=chunksize,
    )
    for chunk in reader:
        total += len(chunk)
        mask = (
            (chunk["LAT"] == "ENG")
            & chunk["SAB"].isin(_SABS_OF_INTEREST)
            & (chunk["SUPPRESS"] != "O")
        )
        kept = chunk.loc[mask, keep_cols]
        if len(kept):
            parts.append(kept.copy())
    filtered = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=keep_cols)
    )
    log.info(
        "Filtered to %d / %d rows across %s",
        len(filtered),
        total,
        sorted(_SABS_OF_INTEREST),
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
        len(icd9_map),
        len(icd10_map),
        len(snomed_terms),
    )
    return icd9_map, icd10_map, snomed_terms


# ---------------------------------------------------------------------------
# MRMAP enrichment (SNOMED CT -> ICD-10-CM authoritative map)
# ---------------------------------------------------------------------------

# MRMAP.RRF column order (UMLS).
_MRMAP_COLS = [
    "MAPSETCUI",
    "MAPSETSAB",
    "MAPSUBSETID",
    "MAPRANK",
    "MAPID",
    "MAPSID",
    "FROMID",
    "FROMSID",
    "FROMEXPR",
    "FROMTYPE",
    "FROMRULE",
    "FROMRES",
    "REL",
    "RELA",
    "TOID",
    "TOSID",
    "TOEXPR",
    "TOTYPE",
    "TORULE",
    "TORES",
    "MAPRULE",
    "MAPRES",
    "MAPTYPE",
    "MAPATN",
    "MAPATV",
    "CVF",
]


def _clean_icd_expr(expr: str) -> str:
    """Clean an MRMAP ICD target expression (e.g. ``S93.491?`` -> ``S93.491``)."""
    expr = expr.strip()
    # strip SNOMED map flags / trailing non-code characters
    while expr and expr[-1] not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.":
        expr = expr[:-1]
    return expr


def parse_mrmap_icd10_to_snomed(mrmap_path: Path) -> dict[str, set[str]]:
    """Parse the SNOMED CT -> ICD-10-CM map from MRMAP.RRF, inverted to
    ICD-10-CM code -> {SNOMED concept ids}.

    Only the authoritative ``MAPSETSAB == SNOMEDCT_US`` / ``TOTYPE == SDUI`` rows
    are used; rows flagged not-mappable (``REL == 'XR'``) or with empty targets are
    skipped. No fuzzy/string matching is performed.
    """
    icd10_to_snomed: dict[str, set[str]] = defaultdict(set)
    idx = {name: i for i, name in enumerate(_MRMAP_COLS)}
    n_rows = 0
    with mrmap_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("|")
            if len(f) < len(_MRMAP_COLS):
                continue
            if f[idx["MAPSETSAB"]] != "SNOMEDCT_US" or f[idx["TOTYPE"]] != "SDUI":
                continue
            if f[idx["REL"]] == "XR":  # explicitly not mappable
                continue
            snomed_id = f[idx["FROMEXPR"]].strip()
            icd = _clean_icd_expr(f[idx["TOEXPR"]])
            if not snomed_id or not icd:
                continue
            icd10_to_snomed[icd].add(snomed_id)
            n_rows += 1
    log.info(
        "MRMAP: %d ICD-10-CM codes from %d usable map rows",
        len(icd10_to_snomed),
        n_rows,
    )
    return icd10_to_snomed


def build_icd9_to_icd10_via_cui(mrconso: pd.DataFrame) -> dict[str, set[str]]:
    """Build ICD-9 -> ICD-10 links where both share a UMLS CUI (synonymy).

    This is an authoritative UMLS assertion (same CUI = same concept), used only
    to bridge ICD-9 codes to SNOMED via their ICD-10 siblings + MRMAP. No fuzzy
    string matching is involved.
    """
    cui_to_icd9: dict[str, set[str]] = defaultdict(set)
    cui_to_icd10: dict[str, set[str]] = defaultdict(set)
    for row in mrconso.itertuples(index=False):
        if row.SAB == _SAB_ICD9:
            cui_to_icd9[row.CUI].add(row.CODE)
        elif row.SAB == _SAB_ICD10:
            cui_to_icd10[row.CUI].add(row.CODE)

    icd9_to_icd10: dict[str, set[str]] = defaultdict(set)
    for cui, icd9s in cui_to_icd9.items():
        icd10s = cui_to_icd10.get(cui)
        if not icd10s:
            continue
        for icd9 in icd9s:
            icd9_to_icd10[icd9] |= icd10s
    return icd9_to_icd10


def bridge_icd9_via_icd10(
    icd9_base: dict[str, list[str]],
    icd9_to_icd10: dict[str, set[str]],
    icd10_to_snomed: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Enrich the ICD-9 -> SNOMED map by bridging through ICD-10 + MRMAP.

    For each ICD-9 code, add the SNOMED ids of its CUI-linked ICD-10 siblings
    (whose maps come from MRMAP). Existing shared-CUI mappings are preserved.
    Returns (merged map, provenance: 'shared_cui' | 'bridge_icd10_mrmap' | 'both').
    """
    merged: dict[str, set[str]] = {k: set(v) for k, v in icd9_base.items()}
    provenance: dict[str, str] = {}
    base_keys = set(icd9_base)

    bridged_keys: set[str] = set()
    for icd9, icd10s in icd9_to_icd10.items():
        bridged: set[str] = set()
        for icd10 in icd10s:
            bridged.update(icd10_to_snomed.get(icd10, []))
        if bridged:
            merged.setdefault(icd9, set())
            before = len(merged[icd9])
            merged[icd9] |= bridged
            if len(merged[icd9]) > before or icd9 not in base_keys:
                bridged_keys.add(icd9)

    for code in set(merged):
        in_base = code in base_keys
        in_bridge = code in bridged_keys
        if in_base and in_bridge:
            provenance[code] = "both"
        elif in_bridge:
            provenance[code] = "bridge_icd10_mrmap"
        else:
            provenance[code] = "shared_cui"

    return {k: sorted(v) for k, v in merged.items()}, provenance


def merge_icd_maps(
    base: dict[str, list[str]],
    extra: dict[str, set[str]],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Union *extra* into *base* without deleting existing mappings.

    Returns the merged map and a provenance dict (code -> 'shared_cui' | 'mrmap'
    | 'both').
    """
    merged: dict[str, set[str]] = {k: set(v) for k, v in base.items()}
    provenance: dict[str, str] = {}
    base_keys = set(base)
    extra_keys = set(extra)

    for code in base_keys | extra_keys:
        in_base = code in base_keys
        in_extra = code in extra_keys
        if in_base and in_extra:
            provenance[code] = "both"
        elif in_extra:
            provenance[code] = "mrmap"
        else:
            provenance[code] = "shared_cui"
        merged.setdefault(code, set())
        if in_extra:
            merged[code] |= extra[code]

    merged_sorted = {k: sorted(v) for k, v in merged.items()}
    return merged_sorted, provenance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 8 -- Build ICD-to-SNOMED crosswalk from UMLS MRCONSO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mrconso",
        required=True,
        type=Path,
        help="Path to MRCONSO.RRF",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for JSON mapping files",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
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
