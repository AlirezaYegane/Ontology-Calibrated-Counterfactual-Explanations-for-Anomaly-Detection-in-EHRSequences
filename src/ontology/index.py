from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class OntologyIndex:
    preferred_terms: dict[str, str] = field(default_factory=dict)
    parents: dict[str, list[str]] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)

    code_domains: dict[str, str] = field(default_factory=dict)
    rx_to_classes: dict[str, list[str]] = field(default_factory=dict)
    required_diagnoses_for_code: dict[str, list[str]] = field(default_factory=dict)
    mutually_exclusive_pairs: set[tuple[str, str]] = field(default_factory=set)

    # Phase 2 crosswalk maps (optional; populated by the loader when present).
    icd_to_snomed: dict[str, list[str]] = field(default_factory=dict)
    drug_to_rxcui: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Basic lookups
    # ------------------------------------------------------------------

    def get_term(self, code: str) -> str:
        return self.preferred_terms.get(code, code)

    def has_concept(self, code: str) -> bool:
        """True if the concept appears anywhere in the hierarchy or terms."""
        return (
            code in self.preferred_terms
            or code in self.parents
            or code in self.children
        )

    def get_parents(self, code: str) -> list[str]:
        return self.parents.get(code, [])

    def get_children(self, code: str) -> list[str]:
        return self.children.get(code, [])

    # ------------------------------------------------------------------
    # Transitive traversal (deterministic, depth-bounded)
    # ------------------------------------------------------------------

    def get_ancestors(self, code: str, max_depth: int = 64) -> set[str]:
        """All transitive parents of *code* (excluding *code* itself)."""
        return self._transitive(code, self.parents, max_depth)

    def get_descendants(self, code: str, max_depth: int = 64) -> set[str]:
        """All transitive children of *code* (excluding *code* itself)."""
        return self._transitive(code, self.children, max_depth)

    @staticmethod
    def _transitive(
        start: str,
        adjacency: dict[str, list[str]],
        max_depth: int,
    ) -> set[str]:
        out: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nxt in adjacency.get(node, []):
                if nxt != start and nxt not in out:
                    out.add(nxt)
                    queue.append((nxt, depth + 1))
        return out

    # ------------------------------------------------------------------
    # Crosswalk lookups (degrade gracefully when maps are absent)
    # ------------------------------------------------------------------

    def map_icd_to_snomed(self, icd_code: str) -> list[str]:
        return list(self.icd_to_snomed.get(icd_code, []))

    def map_drug_to_rxcui(self, drug_name: str) -> str | None:
        return self.drug_to_rxcui.get(drug_name)

    def get_rx_classes(self, rxcui: str) -> list[str]:
        return list(self.rx_to_classes.get(rxcui, []))

    def get_siblings(self, code: str) -> list[str]:
        siblings: set[str] = set()
        for parent in self.get_parents(code):
            for child in self.get_children(parent):
                if child != code:
                    siblings.add(child)
        return sorted(siblings)

    def get_neighbors(self, code: str) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in (
            self.get_parents(code) + self.get_children(code) + self.get_siblings(code)
        ):
            if item != code and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def get_replacements(self, code: str, top_k: int = 5) -> list[str]:
        candidates = (
            self.get_siblings(code) + self.get_parents(code) + self.get_children(code)
        )
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate != code and candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped[:top_k]
