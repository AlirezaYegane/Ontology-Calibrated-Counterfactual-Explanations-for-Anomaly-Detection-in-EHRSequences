"""Phase 2 -- tests for ontology distance / neighborhood utilities."""

from __future__ import annotations

from src.ontology import (
    OntologyIndex,
    ancestor_distance,
    neighborhood,
    shortest_path_distance,
)


def build_index() -> OntologyIndex:
    # ENDO -> DM -> {DM1, DM2};  separate island: CARD -> CAD
    return OntologyIndex(
        parents={"DM": ["ENDO"], "DM1": ["DM"], "DM2": ["DM"], "CAD": ["CARD"]},
        children={"ENDO": ["DM"], "DM": ["DM1", "DM2"], "CARD": ["CAD"]},
        preferred_terms={
            "ENDO": "e",
            "DM": "d",
            "DM1": "d1",
            "DM2": "d2",
            "CARD": "c",
            "CAD": "cad",
        },
    )


def test_shortest_path_self_and_adjacent() -> None:
    idx = build_index()
    assert shortest_path_distance(idx, "DM", "DM") == 0
    assert shortest_path_distance(idx, "DM", "DM1") == 1
    assert shortest_path_distance(idx, "ENDO", "DM1") == 2


def test_shortest_path_siblings() -> None:
    idx = build_index()
    # DM1 -> DM -> DM2
    assert shortest_path_distance(idx, "DM1", "DM2") == 2


def test_shortest_path_unreachable_and_unknown() -> None:
    idx = build_index()
    # disconnected islands
    assert shortest_path_distance(idx, "DM1", "CAD") is None
    # unknown concept
    assert shortest_path_distance(idx, "DM1", "NOPE") is None
    assert shortest_path_distance(idx, "NOPE", "NOPE") is None


def test_ancestor_distance() -> None:
    idx = build_index()
    # DM is an ancestor of DM1 at distance 1
    assert ancestor_distance(idx, "DM1", "DM") == 1
    # DM1 and DM2 share LCA DM: 1 + 1
    assert ancestor_distance(idx, "DM1", "DM2") == 2
    # no common ancestor across islands
    assert ancestor_distance(idx, "DM1", "CAD") is None


def test_neighborhood_deterministic_and_safe() -> None:
    idx = build_index()
    nb = neighborhood(idx, "DM1", radius=1)
    assert nb == sorted(nb)  # deterministic order
    assert "DM" in nb  # parent
    assert "DM2" in nb  # sibling
    assert neighborhood(idx, "NOPE") == []
    assert neighborhood(idx, "DM1", radius=0) == []
