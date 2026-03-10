from __future__ import annotations

from dataclasses import dataclass

from .index import OntologyIndex
from .rules import OntologyRule
from .types import ClinicalRecord, OntologyViolation


@dataclass
class OntologyEngine:
    index: OntologyIndex
    rules: list[OntologyRule]

    def ontology_check(self, record: ClinicalRecord) -> list[OntologyViolation]:
        all_violations: list[OntologyViolation] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()

        for rule in self.rules:
            for violation in rule.check(record, self.index):
                key = (violation.rule_id, violation.kind, tuple(sorted(violation.codes)))
                if key not in seen:
                    seen.add(key)
                    all_violations.append(violation)

        return all_violations

    def score_violations(self, record: ClinicalRecord, alpha: float = 1.0) -> tuple[float, list[OntologyViolation]]:
        violations = self.ontology_check(record)
        score = alpha * sum(item.severity for item in violations)
        return score, violations

    def get_replacements(self, code: str, top_k: int = 5) -> list[str]:
        return self.index.get_replacements(code, top_k=top_k)
