"""
src/preprocessing/build_rxnorm_maps.py
=======================================
Day 8 -- Build drug-name-to-RxCUI mapping dictionary from RXNCONSO.RRF.

Reads RXNCONSO.RRF and produces:

  * drugname_to_rxcui.json -- normalized uppercase drug name -> RxCUI

The mapping uses English RxNorm entries (SAB=RXNORM) and builds a
lookup keyed by the normalized drug name (uppercased, whitespace-collapsed)
that mirrors the ``MED_NAME:`` token format used by the extraction scripts.

CLI::

    python -m src.preprocessing.build_rxnorm_maps \\
        --rxnconso ontologies/rxnorm/rrf/RXNCONSO.RRF \\
        --output-dir ontologies/umls_maps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_RXNCONSO_COLS = [
    "RXCUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
    "RXAUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
    "STR", "SRL", "SUPPRESS", "CVF",
]

# TTYs that represent prescribable / clinical drug concepts
_CLINICAL_TTYS = frozenset({
    "SCD", "SBD", "GPCK", "BPCK",  # clinical drugs / packs
    "IN", "MIN", "PIN",             # ingredients
    "BN",                            # brand names
    "SCDC", "SBDC",                  # clinical drug components
})


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_rxnconso(rxnconso_path: Path) -> pd.DataFrame:
    """Load RXNCONSO.RRF filtered to English RXNORM entries."""
    log.info("Loading %s", rxnconso_path)
    df = pd.read_csv(
        rxnconso_path, sep="|", header=None, names=_RXNCONSO_COLS,
        dtype=str, index_col=False, low_memory=False,
    )
    mask = (
        (df["LAT"] == "ENG")
        & (df["SAB"] == "RXNORM")
        & (df["SUPPRESS"] != "O")
    )
    filtered = df.loc[mask, ["RXCUI", "STR", "TTY"]].copy()
    log.info("Filtered to %d RXNORM English rows", len(filtered))
    return filtered


def _normalize_name(name: str) -> str:
    """Normalize a drug name to match MED_NAME token format."""
    return "_".join(name.upper().split())


def build_drugname_map(rxnconso: pd.DataFrame) -> dict[str, str]:
    """Build normalized-drug-name -> RxCUI mapping.

    When multiple RxCUIs map to the same normalized name, prefer clinical
    TTYs (IN, SCD, SBD, etc.) over less specific types.
    """
    name_to_rxcui: dict[str, str] = {}
    name_to_priority: dict[str, int] = {}

    for row in rxnconso.itertuples(index=False):
        name = row.STR
        if pd.isna(name) or not name.strip():
            continue
        normalized = _normalize_name(name)
        tty = row.TTY if pd.notna(row.TTY) else ""
        priority = 1 if tty in _CLINICAL_TTYS else 0

        if normalized not in name_to_rxcui or priority > name_to_priority[normalized]:
            name_to_rxcui[normalized] = row.RXCUI
            name_to_priority[normalized] = priority

    log.info("Drug-name-to-RxCUI entries: %d", len(name_to_rxcui))
    return name_to_rxcui


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 8 -- Build drug-name-to-RxCUI mapping from RXNCONSO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rxnconso", required=True, type=Path,
        help="Path to RXNCONSO.RRF",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for JSON mapping file",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rxnconso = load_rxnconso(args.rxnconso)
    drugname_map = build_drugname_map(rxnconso)

    out_path = args.output_dir / "drugname_to_rxcui.json"
    out_path.write_text(
        json.dumps(drugname_map, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved %s (%d entries)", out_path.name, len(drugname_map))
    log.info("Done.")


if __name__ == "__main__":
    main()
