from __future__ import annotations

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

    def get_term(self, code: str) -> str:
        return self.preferred_terms.get(code, code)

    def get_parents(self, code: str) -> list[str]:
        return self.parents.get(code, [])

    def get_children(self, code: str) -> list[str]:
        return self.children.get(code, [])

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
        for item in self.get_parents(code) + self.get_children(code) + self.get_siblings(code):
            if item != code and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def get_replacements(self, code: str, top_k: int = 5) -> list[str]:
        candidates = self.get_siblings(code) + self.get_parents(code) + self.get_children(code)
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate != code and candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped[:top_k]
