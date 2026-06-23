"""
src/ontology/types.py
=====================
Backward-compatibility shim. The canonical data model now lives in
``src/ontology/records.py``. This module re-exports the canonical types so that
existing imports (e.g. ``from .types import ClinicalRecord, OntologyViolation``)
keep working.
"""

from __future__ import annotations

from .records import (
    ClinicalRecord,
    OntologyConcept,
    OntologyRuleResult,
    OntologyViolation,
)

__all__ = [
    "ClinicalRecord",
    "OntologyConcept",
    "OntologyViolation",
    "OntologyRuleResult",
]
