"""
src/ontology/rule_engine.py
===========================
Phase 2 -- Canonical ontology rule classes.

These are the *real* ontology-backed rules used by :class:`OntologyEngine`. They
operate over an :class:`OntologyIndex` (hierarchy + maps) and emit canonical
:class:`OntologyViolation` objects.

This module replaces the previously-broken ``from .rules import OntologyRule,
DemographicRule, ...`` imports (those classes never existed in ``rules.py``,
which only held the legacy ICD-prefix ``compute_s_ont`` fallback).

Rules are intentionally hierarchy-aware: a forbidden/required concept is matched
not only directly but also through its ontology descendants, so that e.g. a
"normal pregnancy" code is recognized as a pregnancy concept, and "type 2
diabetes" satisfies a "diabetes" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .records import ClinicalRecord, OntologyViolation

if TYPE_CHECKING:  # avoid import cost / cycles at runtime
    from .index import OntologyIndex


def _normalize_sex(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"m", "male", "man"}:
        return "M"
    if text in {"f", "female", "woman"}:
        return "F"
    return str(value).strip().upper() or None


class OntologyRule:
    """Base class for ontology rules.

    Subclasses implement :meth:`check`, returning a list of violations for a
    record given an ontology index.
    """

    rule_id: str

    def check(
        self,
        record: ClinicalRecord,
        index: "OntologyIndex",
    ) -> list[OntologyViolation]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class DemographicRule(OntologyRule):
    """Flag codes that are incompatible with the patient's sex.

    ``sex_to_forbidden_codes`` maps a normalized sex (``"M"``/``"F"``) to a set of
    forbidden concept ids. A record code triggers the rule if it equals a
    forbidden code or is a descendant of one.
    """

    rule_id: str
    sex_to_forbidden_codes: dict[str, set[str]] = field(default_factory=dict)
    severity: float = 1.0

    def check(
        self, record: ClinicalRecord, index: "OntologyIndex"
    ) -> list[OntologyViolation]:
        sex = _normalize_sex(record.sex)
        if sex is None:
            return []
        forbidden = self.sex_to_forbidden_codes.get(sex)
        if not forbidden:
            return []
        forbidden_set = set(forbidden)

        offending: list[str] = []
        for code in record.codes:
            if code in forbidden_set:
                offending.append(code)
                continue
            # descendant match: is any forbidden concept an ancestor of `code`?
            if index.get_ancestors(code) & forbidden_set:
                offending.append(code)

        # de-duplicate, preserve order
        seen: set[str] = set()
        unique = [c for c in offending if not (c in seen or seen.add(c))]
        if not unique:
            return []
        return [
            OntologyViolation(
                rule_id=self.rule_id,
                kind="demographic_mismatch",
                message=f"Sex-incompatible code(s) for sex={sex}: {unique}",
                codes=tuple(unique),
                severity=self.severity,
            )
        ]


@dataclass
class RequiredCodesRule(OntologyRule):
    """Flag a code whose required supporting diagnosis is absent.

    Uses ``index.required_diagnoses_for_code``: a code is satisfied if any of its
    required diagnoses (or a descendant of one) is present in the record.
    """

    rule_id: str
    severity: float = 1.0

    def check(
        self, record: ClinicalRecord, index: "OntologyIndex"
    ) -> list[OntologyViolation]:
        code_set = set(record.codes)
        violations: list[OntologyViolation] = []
        reported: set[str] = set()

        for code in record.codes:
            if code in reported:
                continue
            required = index.required_diagnoses_for_code.get(code)
            if not required:
                continue
            satisfied = False
            for req in required:
                if req in code_set or (code_set & index.get_descendants(req)):
                    satisfied = True
                    break
            if not satisfied:
                reported.add(code)
                violations.append(
                    OntologyViolation(
                        rule_id=self.rule_id,
                        kind="missing_required_code",
                        message=(
                            f"Code {code} requires one of {sorted(required)} "
                            "(or a descendant), none present."
                        ),
                        codes=(code,),
                        severity=self.severity,
                    )
                )
        return violations


@dataclass
class MutualExclusionRule(OntologyRule):
    """Flag mutually-exclusive concept pairs co-occurring in a record.

    Uses ``index.mutually_exclusive_pairs``. A pair fires if both members are
    present (directly or via a descendant).
    """

    rule_id: str
    severity: float = 1.0

    def _present(
        self, concept: str, code_set: set[str], index: "OntologyIndex"
    ) -> bool:
        if concept in code_set:
            return True
        return bool(code_set & index.get_descendants(concept))

    def check(
        self, record: ClinicalRecord, index: "OntologyIndex"
    ) -> list[OntologyViolation]:
        code_set = set(record.codes)
        violations: list[OntologyViolation] = []
        for a, b in sorted(index.mutually_exclusive_pairs):
            if self._present(a, code_set, index) and self._present(b, code_set, index):
                violations.append(
                    OntologyViolation(
                        rule_id=self.rule_id,
                        kind="mutual_exclusion",
                        message=f"Mutually exclusive concepts co-occur: ({a}, {b}).",
                        codes=(a, b),
                        severity=self.severity,
                    )
                )
        return violations


__all__ = [
    "OntologyRule",
    "DemographicRule",
    "RequiredCodesRule",
    "MutualExclusionRule",
]
