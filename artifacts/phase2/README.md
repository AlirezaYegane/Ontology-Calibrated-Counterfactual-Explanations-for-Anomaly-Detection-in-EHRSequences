# Phase 2 — Real Ontology Integration

## Status: CODE-COMPLETE, DATA-BLOCKED

The ontology *code* is now consolidated, canonical, and tested. The ontology
*data* (licensed SNOMED CT / RxNorm / UMLS) is **not present**, so real coverage
cannot be — and was not — computed. No ontology data was fabricated.

## What was done
- Fixed the broken `src.ontology` import surface. `tests/test_ontology.py` now
  collects and passes (previously: `ImportError: cannot import name 'ClinicalRecord'`).
- Consolidated into one canonical subsystem:
  - `records.py` — `ClinicalRecord`, `OntologyConcept`, `OntologyViolation`, `OntologyRuleResult`.
  - `index.py` — `OntologyIndex` + transitive `get_ancestors`/`get_descendants` + crosswalk lookups.
  - `rule_engine.py` — `OntologyRule`, `DemographicRule`, `RequiredCodesRule`, `MutualExclusionRule` (hierarchy/descendant aware).
  - `engine.py`, `loader.py`, `distance.py` wired together.
  - `rules.py` kept **as a labelled legacy ICD-prefix fallback**, untouched.
- Added `scripts/build_ontology_coverage_report.py` — emits real coverage when
  assets exist, otherwise a truthful `blocked_missing_real_ontology_assets` report.
- Added synthetic fixtures under `tests/fixtures/ontology/` (clearly not licensed data).

## Coverage
See `coverage_report.json`. Current status: **`blocked_missing_real_ontology_assets`**.
Missing: `snomed_hierarchy.json`, `snomed_terms.json`, `icd9_to_snomed.json`,
`icd10_to_snomed.json`, `drugname_to_rxcui.json`.

## Is it safe to build ontology-backed anomaly v2 now?
**No.** The engine is ready, but real concept sets (sex-restricted concepts,
drug→indication, disjoint diagnosis pairs) require the licensed data. Anomaly-v2
injectors stay guarded until Phase 2b loads real assets.

## Honesty note
Do not claim "real ontology integration" in the paper yet. The correct current
claim is: *"the ontology engine is implemented and validated on synthetic
fixtures; real SNOMED/RxNorm integration is pending data acquisition (Phase 2b)."*

See `docs/ontology_integration_status.md` for the full status.
