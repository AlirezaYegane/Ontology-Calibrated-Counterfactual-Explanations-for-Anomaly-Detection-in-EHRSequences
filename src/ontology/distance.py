"""
src/ontology/distance.py
========================
Phase 2 -- Ontology distance and neighborhood utilities.

Used later by counterfactual generation (Phase 4) to propose minimal,
ontology-plausible edits (siblings/parents) and to measure how far an edit moves
within the hierarchy. All functions are deterministic and degrade safely for
unknown concepts (returning ``None`` distances or empty neighborhoods).
"""

from __future__ import annotations

from collections import deque

from .index import OntologyIndex


def _undirected_neighbors(index: OntologyIndex, code: str) -> list[str]:
    """Parents + children (the undirected IS-A adjacency), deterministic order."""
    out: list[str] = []
    seen: set[str] = set()
    for nxt in list(index.get_parents(code)) + list(index.get_children(code)):
        if nxt != code and nxt not in seen:
            seen.add(nxt)
            out.append(nxt)
    return out


def shortest_path_distance(
    index: OntologyIndex,
    a: str,
    b: str,
    max_nodes: int = 100_000,
) -> int | None:
    """Undirected IS-A shortest-path distance between two concepts.

    Returns
    -------
    ``0`` if ``a == b``; a positive integer hop count if a path exists; ``None``
    if either concept is unknown to the hierarchy or no path is found.
    """
    if a == b:
        return 0 if index.has_concept(a) else None
    if not index.has_concept(a) or not index.has_concept(b):
        return None

    visited: set[str] = {a}
    queue: deque[tuple[str, int]] = deque([(a, 0)])
    while queue and len(visited) <= max_nodes:
        node, dist = queue.popleft()
        for nxt in _undirected_neighbors(index, node):
            if nxt == b:
                return dist + 1
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def ancestor_distance(index: OntologyIndex, a: str, b: str) -> int | None:
    """Distance via the nearest common ancestor (fallback metric).

    Defined as ``depth(a→lca) + depth(b→lca)`` over ancestor sets. Returns
    ``None`` if the concepts share no common ancestor (and are not equal/related
    by direct ancestry).
    """
    if a == b:
        return 0 if index.has_concept(a) else None

    anc_a = _ancestor_depths(index, a)
    anc_b = _ancestor_depths(index, b)

    # direct ancestry (one is an ancestor of the other)
    if b in anc_a:
        return anc_a[b]
    if a in anc_b:
        return anc_b[a]

    common = set(anc_a) & set(anc_b)
    if not common:
        return None
    return min(anc_a[c] + anc_b[c] for c in common)


def _ancestor_depths(
    index: OntologyIndex, code: str, max_depth: int = 64
) -> dict[str, int]:
    """Map each ancestor of *code* to its hop distance (BFS over parents)."""
    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque([(code, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for parent in index.get_parents(node):
            if parent not in depths or depth + 1 < depths[parent]:
                depths[parent] = depth + 1
                queue.append((parent, depth + 1))
    return depths


def neighborhood(index: OntologyIndex, code: str, radius: int = 1) -> list[str]:
    """Deterministic set of nearby concepts within *radius* IS-A hops.

    ``radius=1`` yields parents, children, and siblings. Returns ``[]`` for
    unknown concepts.
    """
    if radius < 1 or not index.has_concept(code):
        return []

    frontier = {code}
    collected: set[str] = set()
    for _ in range(radius):
        nxt_frontier: set[str] = set()
        for node in frontier:
            for nb in _undirected_neighbors(index, node):
                if nb != code:
                    collected.add(nb)
                    nxt_frontier.add(nb)
        # include siblings explicitly at radius>=1
        for sib in index.get_siblings(code):
            collected.add(sib)
        frontier = nxt_frontier
    return sorted(collected)


__all__ = [
    "shortest_path_distance",
    "ancestor_distance",
    "neighborhood",
]
