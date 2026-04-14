"""
src/ontology/loader.py
=======================
Load real SNOMED CT / RxNorm data into a fully populated OntologyEngine.

Reads:
  * ``snomed_hierarchy.json`` -- parent/child adjacency (from parse_snomed.py)
  * ``snomed_terms.json``     -- concept ID -> preferred term (from build_umls_maps.py)

Returns an :class:`OntologyEngine` with :class:`DemographicRule`,
:class:`RequiredCodesRule`, and :class:`MutualExclusionRule` populated
with real clinical constraints.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from .engine import OntologyEngine
from .index import OntologyIndex
from .rules import DemographicRule, MutualExclusionRule, RequiredCodesRule

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Well-known SNOMED CT concept IDs
# ---------------------------------------------------------------------------

# Pregnancy, childbirth, and the puerperium (parent concept)
_PREGNANCY_ROOT = "77386006"

# Diabetes mellitus
_DIABETES_ROOTS = ["73211009", "44054006", "46635009"]  # DM, DM type 2, DM type 1

# Malignant neoplastic disease (cancer)
_CANCER_ROOTS = ["363346000", "55342001", "86049000"]  # malignant neoplasm (various)

# Atrial fibrillation and coagulation disorders
_AF_AND_COAG = ["49436004", "64779008", "439127006"]  # AF, coag disorder, disorder of coag

# Type 1 diabetes (for mutual exclusion with type 2)
_DM_TYPE1 = "46635009"
_DM_TYPE2 = "44054006"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefix(concept_id: str) -> str:
    """Add SNOMED: prefix if not already present."""
    if concept_id.startswith("SNOMED:"):
        return concept_id
    return f"SNOMED:{concept_id}"


def _collect_descendants(
    concept_id: str,
    children_map: dict[str, list[str]],
    max_depth: int = 5,
) -> set[str]:
    """BFS to collect descendants up to *max_depth* levels deep."""
    descendants: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(concept_id, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth > max_depth:
            continue
        for child in children_map.get(current, []):
            if child not in descendants:
                descendants.add(child)
                queue.append((child, depth + 1))

    return descendants


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_ontology_index(
    data_dir: Path,
    *,
    hierarchy_filename: str = "snomed_hierarchy.json",
    terms_filename: str = "snomed_terms.json",
) -> OntologyIndex:
    """Load SNOMED hierarchy and terms into an :class:`OntologyIndex`.

    Parameters
    ----------
    data_dir:
        Directory containing the JSON files (typically ``ontologies/umls_maps``
        or ``ontologies/processed``).
    hierarchy_filename:
        Name of the hierarchy JSON file.
    terms_filename:
        Name of the SNOMED-terms JSON file.
    """
    # --- hierarchy ---
    hierarchy_path = data_dir / hierarchy_filename
    raw_parents: dict[str, list[str]] = {}
    raw_children: dict[str, list[str]] = {}

    if hierarchy_path.exists():
        log.info("Loading hierarchy from %s", hierarchy_path.name)
        hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        raw_parents = hierarchy.get("parents", {})
        raw_children = hierarchy.get("children", {})
    else:
        log.warning("Hierarchy file not found: %s", hierarchy_path)

    # Prefix all keys and values with SNOMED:
    prefixed_parents: dict[str, list[str]] = {
        _prefix(k): [_prefix(v) for v in vs]
        for k, vs in raw_parents.items()
    }
    prefixed_children: dict[str, list[str]] = {
        _prefix(k): [_prefix(v) for v in vs]
        for k, vs in raw_children.items()
    }

    # --- terms ---
    terms_path = data_dir / terms_filename
    prefixed_terms: dict[str, str] = {}

    if terms_path.exists():
        log.info("Loading terms from %s", terms_path.name)
        raw_terms: dict[str, str] = json.loads(terms_path.read_text(encoding="utf-8"))
        prefixed_terms = {_prefix(k): v for k, v in raw_terms.items()}
    else:
        log.warning("Terms file not found: %s", terms_path)

    log.info(
        "OntologyIndex: %d parents, %d children, %d terms",
        len(prefixed_parents), len(prefixed_children), len(prefixed_terms),
    )

    return OntologyIndex(
        preferred_terms=prefixed_terms,
        parents=prefixed_parents,
        children=prefixed_children,
    )


def load_ontology_engine(
    data_dir: Path,
    *,
    hierarchy_filename: str = "snomed_hierarchy.json",
    terms_filename: str = "snomed_terms.json",
) -> OntologyEngine:
    """Load a fully populated :class:`OntologyEngine` with real clinical rules.

    Parameters
    ----------
    data_dir:
        Directory containing the JSON files.
    """
    index = load_ontology_index(
        data_dir,
        hierarchy_filename=hierarchy_filename,
        terms_filename=terms_filename,
    )

    # --- Collect pregnancy-related codes for demographic rule ---
    # Use raw (unprefixed) children map for descendant collection,
    # then prefix the results.
    hierarchy_path = data_dir / hierarchy_filename
    raw_children: dict[str, list[str]] = {}
    if hierarchy_path.exists():
        hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        raw_children = hierarchy.get("children", {})

    pregnancy_descendants = _collect_descendants(_PREGNANCY_ROOT, raw_children)
    pregnancy_descendants.add(_PREGNANCY_ROOT)
    pregnancy_codes = {_prefix(c) for c in pregnancy_descendants}

    log.info("Pregnancy-related codes for demographic rule: %d", len(pregnancy_codes))

    # --- Demographic rule: pregnancy codes forbidden for male patients ---
    demographic_rule = DemographicRule(
        rule_id="sex_pregnancy_check",
        sex_to_forbidden_codes={
            "M": pregnancy_codes,
        },
    )

    # --- Required-diagnosis rules ---
    # These encode real clinical constraints:
    # 1. Insulin requires a diabetes diagnosis
    # 2. Antineoplastic agents require a cancer diagnosis
    # 3. Anticoagulants require AF or coagulation disorder
    diabetes_codes = [_prefix(c) for c in _DIABETES_ROOTS]
    cancer_codes = [_prefix(c) for c in _CANCER_ROOTS]
    af_coag_codes = [_prefix(c) for c in _AF_AND_COAG]

    index.required_diagnoses_for_code.update({
        "RXNORM:5856":   diabetes_codes,    # insulin (RxCUI 5856)
        "RXNORM:6809":   diabetes_codes,    # metformin (RxCUI 6809)
        "RXNORM:224905": cancer_codes,      # cyclophosphamide (RxCUI 224905)
        "RXNORM:11289":  af_coag_codes,     # warfarin (RxCUI 11289)
        "RXNORM:67108":  af_coag_codes,     # enoxaparin (RxCUI 67108)
    })

    required_rule = RequiredCodesRule(rule_id="required_diagnosis_support")

    # --- Mutual exclusion: type-1 and type-2 diabetes ---
    index.mutually_exclusive_pairs.add(
        (_prefix(_DM_TYPE1), _prefix(_DM_TYPE2))
    )

    exclusion_rule = MutualExclusionRule(rule_id="mutual_exclusion")

    engine = OntologyEngine(
        index=index,
        rules=[demographic_rule, required_rule, exclusion_rule],
    )

    log.info(
        "OntologyEngine loaded: %d rules, %d required-diagnosis entries, "
        "%d mutual-exclusion pairs",
        len(engine.rules),
        len(index.required_diagnoses_for_code),
        len(index.mutually_exclusive_pairs),
    )
    return engine
