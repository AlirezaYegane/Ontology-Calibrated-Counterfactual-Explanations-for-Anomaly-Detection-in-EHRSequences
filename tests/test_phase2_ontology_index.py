"""Phase 2 -- tests for OntologyIndex traversal and crosswalk lookups."""

from __future__ import annotations

from src.ontology import OntologyIndex


def build_index() -> OntologyIndex:
    return OntologyIndex(
        preferred_terms={"DM": "Diabetes", "DM1": "Type 1 DM", "DM2": "Type 2 DM"},
        parents={"DM1": ["DM"], "DM2": ["DM"], "DM": ["ENDO"]},
        children={"DM": ["DM1", "DM2"], "ENDO": ["DM"]},
        icd_to_snomed={"250.00": ["DM2"]},
        drug_to_rxcui={"INSULIN": "5856"},
        rx_to_classes={"5856": ["ANTIDIABETIC"]},
    )


def test_parents_children() -> None:
    idx = build_index()
    assert idx.get_parents("DM1") == ["DM"]
    assert set(idx.get_children("DM")) == {"DM1", "DM2"}


def test_ancestors_and_descendants_transitive() -> None:
    idx = build_index()
    assert idx.get_ancestors("DM1") == {"DM", "ENDO"}
    assert idx.get_descendants("ENDO") == {"DM", "DM1", "DM2"}


def test_siblings_and_neighbors() -> None:
    idx = build_index()
    assert idx.get_siblings("DM1") == ["DM2"]
    assert "DM2" in idx.get_neighbors("DM1")
    assert "DM" in idx.get_neighbors("DM1")


def test_unknown_concept_safe() -> None:
    idx = build_index()
    assert idx.get_parents("NOPE") == []
    assert idx.get_ancestors("NOPE") == set()
    assert idx.get_descendants("NOPE") == set()
    assert idx.has_concept("NOPE") is False
    assert idx.has_concept("DM1") is True


def test_crosswalk_lookups() -> None:
    idx = build_index()
    assert idx.map_icd_to_snomed("250.00") == ["DM2"]
    assert idx.map_icd_to_snomed("999") == []
    assert idx.map_drug_to_rxcui("INSULIN") == "5856"
    assert idx.map_drug_to_rxcui("UNKNOWN") is None
    assert idx.get_rx_classes("5856") == ["ANTIDIABETIC"]


def test_no_self_cycle_in_transitive() -> None:
    # Defensive: a self-loop must not cause infinite recursion.
    idx = OntologyIndex(parents={"A": ["A", "B"]}, children={"B": ["A"], "A": ["A"]})
    assert "A" not in idx.get_ancestors("A")
    assert "B" in idx.get_ancestors("A")
