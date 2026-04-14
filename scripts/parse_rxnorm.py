"""
scripts/parse_rxnorm.py
========================
Day 7 -- Parse RxNorm RRF distribution files into usable CSVs.

Reads:
  * RXNCONSO.RRF  -- concepts (RxCUI, name, SAB, TTY)
  * RXNREL.RRF    -- inter-concept relationships

Writes:
  * rxnorm_concepts.csv
  * rxnorm_relationships.csv

CLI::

    python scripts/parse_rxnorm.py \\
        --rxnorm-dir ontologies/rxnorm/rrf \\
        --output-dir ontologies/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# RRF column schemas (pipe-delimited, no header)
_RXNCONSO_COLS = [
    "RXCUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
    "RXAUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
    "STR", "SRL", "SUPPRESS", "CVF",
]

_RXNREL_COLS = [
    "RXCUI1", "RXAUI1", "STYPE1", "REL", "RXCUI2", "RXAUI2",
    "STYPE2", "RELA", "RUI", "SRUI", "SAB", "SL", "DIR", "RG",
    "SUPPRESS", "CVF",
]


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


def _find_rrf(directory: Path, name: str) -> Path:
    """Find an RRF file by name, searching recursively."""
    direct = directory / name
    if direct.is_file():
        return direct
    candidates = sorted(directory.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"{name} not found under: {directory}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_rxnconso(rxnorm_dir: Path) -> pd.DataFrame:
    """Parse RXNCONSO.RRF and return English concepts."""
    path = _find_rrf(rxnorm_dir, "RXNCONSO.RRF")
    log.info("Loading %s", path)
    df = pd.read_csv(
        path, sep="|", header=None, names=_RXNCONSO_COLS,
        dtype=str, index_col=False, low_memory=False,
    )
    # Keep English entries only
    eng = df[df["LAT"] == "ENG"].copy()
    out = eng[["RXCUI", "STR", "SAB", "TTY", "CODE", "SUPPRESS"]].copy()
    out = out.rename(columns={
        "RXCUI": "rxcui",
        "STR": "name",
        "SAB": "sab",
        "TTY": "tty",
        "CODE": "code",
        "SUPPRESS": "suppress",
    })
    log.info("English concepts: %d rows, %d unique RxCUI", len(out), out["rxcui"].nunique())
    return out


def parse_rxnrel(rxnorm_dir: Path) -> pd.DataFrame:
    """Parse RXNREL.RRF and return relationships."""
    path = _find_rrf(rxnorm_dir, "RXNREL.RRF")
    log.info("Loading %s", path)
    df = pd.read_csv(
        path, sep="|", header=None, names=_RXNREL_COLS,
        dtype=str, index_col=False, low_memory=False,
    )
    out = df[["RXCUI1", "RXCUI2", "REL", "RELA", "SAB"]].copy()
    out = out.rename(columns={
        "RXCUI1": "rxcui1",
        "RXCUI2": "rxcui2",
        "REL": "rel",
        "RELA": "rela",
        "SAB": "sab",
    })
    out = out.dropna(subset=["rxcui1", "rxcui2"])
    log.info("Relationships: %d rows", len(out))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 7 -- Parse RxNorm RRF files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rxnorm-dir", required=True, type=Path,
        help="Directory containing RXNCONSO.RRF and RXNREL.RRF",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for processed CSV files",
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

    concepts = parse_rxnconso(args.rxnorm_dir)
    concepts_path = args.output_dir / "rxnorm_concepts.csv"
    concepts.to_csv(concepts_path, index=False)
    log.info("Saved %s (%d rows)", concepts_path.name, len(concepts))

    rels = parse_rxnrel(args.rxnorm_dir)
    rels_path = args.output_dir / "rxnorm_relationships.csv"
    rels.to_csv(rels_path, index=False)
    log.info("Saved %s (%d rows)", rels_path.name, len(rels))


if __name__ == "__main__":
    main()
