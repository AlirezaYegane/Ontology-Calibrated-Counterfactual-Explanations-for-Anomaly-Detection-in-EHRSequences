"""
scripts/build_rxnorm_graph.py
==============================
Day 8 -- Build a directed NetworkX graph from parsed RxNorm CSVs.

Reads:
  * ontologies/processed/rxnorm_concepts.csv
  * ontologies/processed/rxnorm_relationships.csv

Writes:
  * ontologies/processed/rxnorm_graph.gpickle

CLI::

    python scripts/build_rxnorm_graph.py \\
        --input-dir ontologies/processed \\
        --output-path ontologies/processed/rxnorm_graph.gpickle
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def build_graph(concepts_path: Path, relationships_path: Path):
    """Build a directed NetworkX graph from concept and relationship CSVs.

    Returns a ``networkx.DiGraph``.  NetworkX is imported here to keep it
    an optional dependency at module level.
    """
    import networkx as nx

    concepts = pd.read_csv(concepts_path, dtype=str)
    rels = pd.read_csv(relationships_path, dtype=str)

    log.info(
        "Loaded %d concept rows, %d relationship rows",
        len(concepts), len(rels),
    )

    # Unique RxCUIs with a representative name
    name_lookup: dict[str, str] = {}
    for row in concepts.itertuples(index=False):
        rxcui = row.rxcui
        if rxcui not in name_lookup:
            name_lookup[rxcui] = row.name

    G = nx.DiGraph()

    # Add nodes
    for rxcui, name in name_lookup.items():
        G.add_node(rxcui, name=name)

    # Add edges
    for row in rels.itertuples(index=False):
        src, dst = row.rxcui1, row.rxcui2
        if src and dst:
            rela = getattr(row, "rela", None) or ""
            rel = getattr(row, "rel", None) or ""
            G.add_edge(src, dst, rel=rel, rela=rela)

    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 8 -- Build RxNorm directed graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory with rxnorm_concepts.csv and rxnorm_relationships.csv",
    )
    p.add_argument(
        "--output-path", required=True, type=Path,
        help="Output .gpickle path for the graph",
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

    concepts_path = args.input_dir / "rxnorm_concepts.csv"
    rels_path = args.input_dir / "rxnorm_relationships.csv"

    G = build_graph(concepts_path, rels_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    size = args.output_path.stat().st_size
    log.info(
        "Saved %s (%d nodes, %d edges, %d bytes / %.1f MiB)",
        args.output_path.name,
        G.number_of_nodes(),
        G.number_of_edges(),
        size,
        size / (1024 * 1024),
    )


if __name__ == "__main__":
    main()
