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
from pathlib import Path
from typing import Any

from .engine import OntologyEngine
from .index import OntologyIndex

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefix(concept_id: str) -> str:
    """Add SNOMED: prefix if not already present."""
    if concept_id.startswith("SNOMED:"):
        return concept_id
    return f"SNOMED:{concept_id}"


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
        _prefix(k): [_prefix(v) for v in vs] for k, vs in raw_parents.items()
    }
    prefixed_children: dict[str, list[str]] = {
        _prefix(k): [_prefix(v) for v in vs] for k, vs in raw_children.items()
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

    # --- optional crosswalk maps (loaded only if present) ---
    icd_to_snomed: dict[str, list[str]] = {}
    for fname in ("icd9_to_snomed.json", "icd10_to_snomed.json"):
        fpath = data_dir / fname
        if fpath.exists():
            raw_map: dict[str, list[str]] = json.loads(
                fpath.read_text(encoding="utf-8")
            )
            for icd, snomed_ids in raw_map.items():
                icd_to_snomed.setdefault(icd, [])
                for sid in snomed_ids:
                    icd_to_snomed[icd].append(_prefix(sid))
        else:
            log.warning("Crosswalk file not found: %s", fpath)

    drug_to_rxcui: dict[str, str] = {}
    drug_path = data_dir / "drugname_to_rxcui.json"
    if drug_path.exists():
        raw_drug: dict[str, Any] = json.loads(drug_path.read_text(encoding="utf-8"))
        drug_to_rxcui = {str(k): str(v) for k, v in raw_drug.items()}
    else:
        log.warning("Drug map file not found: %s", drug_path)

    log.info(
        "OntologyIndex: %d parents, %d children, %d terms, %d icd-maps, %d drug-maps",
        len(prefixed_parents),
        len(prefixed_children),
        len(prefixed_terms),
        len(icd_to_snomed),
        len(drug_to_rxcui),
    )

    return OntologyIndex(
        preferred_terms=prefixed_terms,
        parents=prefixed_parents,
        children=prefixed_children,
        icd_to_snomed=icd_to_snomed,
        drug_to_rxcui=drug_to_rxcui,
    )


def load_ontology_engine(
    data_dir: Path,
    *,
    hierarchy_filename: str = "snomed_hierarchy.json",
    terms_filename: str = "snomed_terms.json",
) -> OntologyEngine:
    """Load an :class:`OntologyEngine` with the Phase 3b curated rule packs.

    The rule packs (:mod:`src.ontology.rule_packs` /
    :mod:`src.ontology.rule_loader`) provide high-precision, auditable
    demographic, medication-required-context, and diabetes-type mutual-exclusion
    rules built from the ICD->SNOMED crosswalk in the loaded index. When the
    crosswalk/hierarchy is absent (e.g. tests on empty dirs), each concept group
    degrades gracefully to its seeded SNOMED roots only.

    Parameters
    ----------
    data_dir:
        Directory containing the ontology JSON files.
    """
    from .rule_loader import build_real_ontology_rules

    index = load_ontology_index(
        data_dir,
        hierarchy_filename=hierarchy_filename,
        terms_filename=terms_filename,
    )

    rules, manifest = build_real_ontology_rules(index)
    engine = OntologyEngine(index=index, rules=rules)

    log.info(
        "OntologyEngine loaded with %d curated rule packs: %s",
        len(rules),
        ", ".join(m["rule_id"] for m in manifest),
    )
    return engine
