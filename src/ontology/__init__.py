"""
src.ontology -- canonical ontology subsystem (Phase 2).

Public surface:

Data model (``records``):
    ClinicalRecord, OntologyConcept, OntologyViolation, OntologyRuleResult
Index / engine:
    OntologyIndex, OntologyEngine
Rules (``rule_engine``):
    OntologyRule, DemographicRule, RequiredCodesRule, MutualExclusionRule
Loading (``loader``):
    load_ontology_index, load_ontology_engine
Distance (``distance``):
    shortest_path_distance, ancestor_distance, neighborhood

Legacy fallback (``rules``): the hard-coded ICD-prefix scoring path remains
available as ``compute_s_ont`` / ``extract_tokens_from_row`` /
``infer_sequence_column`` for backward compatibility with the existing scoring
module. It is NOT the canonical ontology path and should be treated as a labelled
fallback only.
"""

from __future__ import annotations

from .distance import ancestor_distance, neighborhood, shortest_path_distance
from .engine import OntologyEngine
from .index import OntologyIndex
from .loader import load_ontology_engine, load_ontology_index
from .records import (
    ClinicalRecord,
    OntologyConcept,
    OntologyRuleResult,
    OntologyViolation,
)
from .rule_engine import (
    DemographicRule,
    MutualExclusionRule,
    OntologyRule,
    RequiredCodesRule,
)

# Legacy ICD-prefix fallback (kept for backward compatibility).
from .rules import compute_s_ont, extract_tokens_from_row, infer_sequence_column

__all__ = [
    # data model
    "ClinicalRecord",
    "OntologyConcept",
    "OntologyViolation",
    "OntologyRuleResult",
    # index / engine
    "OntologyIndex",
    "OntologyEngine",
    # rules
    "OntologyRule",
    "DemographicRule",
    "RequiredCodesRule",
    "MutualExclusionRule",
    # loader
    "load_ontology_index",
    "load_ontology_engine",
    # distance
    "shortest_path_distance",
    "ancestor_distance",
    "neighborhood",
    # legacy fallback
    "compute_s_ont",
    "extract_tokens_from_row",
    "infer_sequence_column",
]
