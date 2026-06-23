"""
src/ontology/records.py
=======================
Phase 2 -- Canonical ontology data model.

This module is the single source of truth for the ontology layer's core data
types. Other modules (``engine``, ``rule_engine``, ``loader``, ``distance``) and
later phases (scoring, counterfactual) should import these names from here (or
from the ``src.ontology`` package surface), not redefine them.

Note on the legacy violation type
----------------------------------
``src/ontology/rules.py`` contains a *different*, legacy ``OntologyViolation``
used only by the hard-coded ICD-prefix ``compute_s_ont`` scoring fallback. That
one carries token-weight fields; the canonical one here carries engine/rule
fields. They are intentionally separate and live in separate modules. New code
should use *this* canonical type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ClinicalRecord:
    """A single patient record in ontology-friendly form.

    Attributes
    ----------
    record_id:
        Stable identifier for the record (often ``f"{subject_id}_{hadm_id}"``).
    codes:
        Ordered tuple of clinical code tokens (diagnoses, procedures, meds).
    sex:
        Patient sex, e.g. ``"M"`` / ``"F"`` (kept as the canonical field name for
        backward compatibility with existing tests). See :pyattr:`gender`.
    age_group:
        Optional age bucket (e.g. ``"65-79"``).
    subject_id, hadm_id:
        Optional source identifiers, useful for non-circular split protocols.
    metadata:
        Free-form, non-model-critical extras.
    """

    record_id: Optional[str] = None
    codes: tuple[str, ...] = ()
    sex: Optional[str] = None
    age_group: Optional[str] = None
    subject_id: Optional[str] = None
    hadm_id: Optional[str] = None
    # Raw model-visible source tokens (e.g. ``DX_10_O80``) BEFORE ontology
    # mapping. Used by rules that must anchor on the source diagnosis-code family
    # to avoid many-to-many ICD->SNOMED crosswalk noise. Model-visible only;
    # never answer-key columns.
    source_tokens: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gender(self) -> Optional[str]:
        """Alias for :pyattr:`sex` (some datasets/callers use 'gender')."""
        return self.sex

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        code_keys: tuple[str, ...] = ("codes", "sequence_tokens", "tokens", "sequence"),
        sex_keys: tuple[str, ...] = ("sex", "gender"),
        age_keys: tuple[str, ...] = ("age_group", "age"),
    ) -> ClinicalRecord:
        """Build a :class:`ClinicalRecord` from a dict-like row.

        Only model-safe fields are read; this never ingests answer-key columns.
        """

        def first(keys: tuple[str, ...]) -> Any:
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            return None

        raw_codes = first(code_keys)
        if isinstance(raw_codes, (list, tuple)):
            codes = tuple(str(c) for c in raw_codes)
        elif raw_codes is None:
            codes = ()
        else:
            codes = (str(raw_codes),)

        subject = row.get("subject_id")
        hadm = row.get("hadm_id")
        rid = row.get("record_id") or row.get("source_row_id")
        if rid is None and subject is not None and hadm is not None:
            rid = f"{subject}_{hadm}"

        return cls(
            record_id=None if rid is None else str(rid),
            codes=codes,
            sex=(None if first(sex_keys) is None else str(first(sex_keys))),
            age_group=(None if first(age_keys) is None else str(first(age_keys))),
            subject_id=None if subject is None else str(subject),
            hadm_id=None if hadm is None else str(hadm),
        )


@dataclass(frozen=True)
class OntologyConcept:
    """A single ontology concept with local hierarchy context."""

    concept_id: str
    preferred_term: str = ""
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    domain: Optional[str] = None


@dataclass(frozen=True)
class OntologyViolation:
    """Canonical ontology-rule violation emitted by the rule engine."""

    rule_id: str
    kind: str
    message: str
    codes: tuple[str, ...] = ()
    severity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind,
            "message": self.message,
            "codes": list(self.codes),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class OntologyRuleResult:
    """Aggregated result of running rules over one record."""

    record_id: Optional[str]
    violations: tuple[OntologyViolation, ...] = ()

    @property
    def score(self) -> float:
        """Total severity across violations (the ``Sont`` ingredient)."""
        return float(sum(v.severity for v in self.violations))

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def kinds(self) -> list[str]:
        return [v.kind for v in self.violations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "score": self.score,
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
        }


__all__ = [
    "ClinicalRecord",
    "OntologyConcept",
    "OntologyViolation",
    "OntologyRuleResult",
]
