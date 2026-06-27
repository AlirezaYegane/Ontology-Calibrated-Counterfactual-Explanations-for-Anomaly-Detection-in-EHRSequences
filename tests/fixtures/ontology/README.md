# Synthetic ontology test fixtures

**These files are hand-built, SYNTHETIC test fixtures. They are NOT the licensed
SNOMED CT / RxNorm / UMLS distributions.**

They contain a tiny subset of concept identifiers arranged into a small hierarchy
purely so the ontology loader, index, distance, and rule-engine code can be unit
tested without any licensed data present. Coverage or clinical claims must never
be derived from these fixtures.

Files:
- `snomed_hierarchy.json` — `{"parents": {...}, "children": {...}}` adjacency.
- `snomed_terms.json` — concept id → preferred term.
- `icd9_to_snomed.json` — ICD-9 code → [SNOMED ids] (tiny synthetic crosswalk).
- `drugname_to_rxcui.json` — uppercased drug name → RxCUI (tiny synthetic map).
