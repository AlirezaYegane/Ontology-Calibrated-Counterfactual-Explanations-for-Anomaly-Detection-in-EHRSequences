"""
scripts/parse_snomed.py
========================
Day 7 -- Parse SNOMED CT RF2 snapshot files into usable CSVs and a
hierarchy JSON.

Reads:
  * sct2_Concept_Snapshot_*.txt
  * sct2_Description_Snapshot_*.txt
  * sct2_Relationship_Snapshot_*.txt

Writes:
  * snomed_concepts.csv   -- active concepts with preferred terms
  * snomed_relationships.csv -- active IS-A relationships
  * snomed_hierarchy.json -- parent/child adjacency lists

CLI::

    python scripts/parse_snomed.py \\
        --snomed-dir ontologies/snomed/SnomedCT_*.../Snapshot \\
        --output-dir ontologies/processed
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

# SNOMED CT constants
_IS_A = "116680003"
_FSN_TYPE = "900000000000003001"
_PREFERRED_TYPE = "900000000000013009"
_US_MODULE = "900000000000207008"


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


def _find_rf2(directory: Path, prefix: str) -> Path:
    """Find an RF2 file matching *prefix* in *directory* (recursive)."""
    candidates = sorted(directory.rglob(f"{prefix}*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"No RF2 file matching '{prefix}*.txt' under: {directory}"
        )
    if len(candidates) > 1:
        log.info("Multiple matches for '%s'; using: %s", prefix, candidates[0])
    return candidates[0]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_concepts(snomed_dir: Path) -> pd.DataFrame:
    """Parse sct2_Concept_Snapshot and return active concepts."""
    path = _find_rf2(snomed_dir, "sct2_Concept_Snapshot")
    log.info("Loading concepts from %s", path.name)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    active = df[df["active"] == "1"][["id", "effectivetime", "moduleid"]].copy()
    active = active.rename(columns={"id": "concept_id"})
    log.info("Active concepts: %d", len(active))
    return active


def parse_descriptions(snomed_dir: Path) -> pd.DataFrame:
    """Parse sct2_Description_Snapshot and return preferred terms."""
    path = _find_rf2(snomed_dir, "sct2_Description_Snapshot")
    log.info("Loading descriptions from %s", path.name)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    active = df[df["active"] == "1"].copy()

    # Prefer synonym (Preferred) over FSN for readability
    preferred = active[active["typeid"] == _PREFERRED_TYPE].copy()
    fsn = active[active["typeid"] == _FSN_TYPE].copy()

    # De-duplicate: one preferred term per concept
    preferred = preferred.drop_duplicates(subset=["conceptid"], keep="first")
    fsn = fsn.drop_duplicates(subset=["conceptid"], keep="first")

    # Use preferred term where available, fall back to FSN
    terms = preferred[["conceptid", "term"]].copy()
    fsn_only = fsn[~fsn["conceptid"].isin(terms["conceptid"])][["conceptid", "term"]]
    terms = pd.concat([terms, fsn_only], ignore_index=True)
    terms = terms.rename(columns={"conceptid": "concept_id", "term": "preferred_term"})
    log.info("Preferred terms resolved: %d concepts", len(terms))
    return terms


def parse_relationships(snomed_dir: Path) -> pd.DataFrame:
    """Parse sct2_Relationship_Snapshot and return active IS-A edges."""
    path = _find_rf2(snomed_dir, "sct2_Relationship_Snapshot")
    log.info("Loading relationships from %s", path.name)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    isa = df[(df["active"] == "1") & (df["typeid"] == _IS_A)].copy()
    isa = isa[["sourceid", "destinationid"]].rename(
        columns={"sourceid": "child_id", "destinationid": "parent_id"}
    )
    log.info("Active IS-A relationships: %d", len(isa))
    return isa


def build_hierarchy(relationships: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Build parent/child adjacency lists from IS-A relationship rows."""
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)

    for row in relationships.itertuples(index=False):
        parents[row.child_id].append(row.parent_id)
        children[row.parent_id].append(row.child_id)

    return {
        "parents": {k: sorted(set(v)) for k, v in parents.items()},
        "children": {k: sorted(set(v)) for k, v in children.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 7 -- Parse SNOMED CT RF2 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--snomed-dir", required=True, type=Path,
        help="Path to SNOMED CT Snapshot directory containing RF2 files",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for processed CSV and JSON files",
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

    # --- concepts + descriptions ---
    concepts = parse_concepts(args.snomed_dir)
    descriptions = parse_descriptions(args.snomed_dir)
    merged = concepts.merge(descriptions, on="concept_id", how="left")

    concepts_path = args.output_dir / "snomed_concepts.csv"
    merged.to_csv(concepts_path, index=False)
    log.info("Saved %s (%d rows)", concepts_path.name, len(merged))

    # --- relationships ---
    relationships = parse_relationships(args.snomed_dir)
    rels_path = args.output_dir / "snomed_relationships.csv"
    relationships.to_csv(rels_path, index=False)
    log.info("Saved %s (%d rows)", rels_path.name, len(relationships))

    # --- hierarchy JSON ---
    hierarchy = build_hierarchy(relationships)
    hierarchy_path = args.output_dir / "snomed_hierarchy.json"
    hierarchy_path.write_text(
        json.dumps(hierarchy, indent=1, sort_keys=True), encoding="utf-8",
    )
    log.info(
        "Saved %s (parents: %d, children: %d)",
        hierarchy_path.name,
        len(hierarchy["parents"]),
        len(hierarchy["children"]),
    )


if __name__ == "__main__":
    main()
