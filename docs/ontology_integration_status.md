# Ontology Integration Status (Phase 2 / 2b / 2b-fix)

> **PHASE 2 CLOSED — Status A (Phase 2b-fix).** After enriching maps with the
> authoritative UMLS **MRMAP** SNOMED→ICD-10-CM map, an ICD-9→ICD-10(CUI)→SNOMED
> bridge, and conservative RxNorm IN/PIN ingredient matching, real coverage on
> MIMIC-IV now **meets both targets: diagnosis 0.8006 (≥0.80), medication 0.7767
> (≥0.70)**. ICD-10 diagnosis coverage is 0.92 (authoritative MRMAP). Coverage is
> real and not inflated (no fuzzy/title matching; ambiguous `INSULIN`/`DEXTROSE`
> left unmapped). Full detail: `artifacts/phase2b_fix/README.md`. Final map counts:
> ICD-9→SNOMED 10,254; ICD-10→SNOMED 30,783; RxNorm ingredient index 18,278.
>
> **Safe claim:** real SNOMED/RxNorm integration with diagnosis 0.80 / medication 0.78
> on MIMIC-IV via authoritative UMLS crosswalks.
> **Still not safe:** RxNorm drug-class reasoning (`rxnorm_classes` unavailable);
> complete ICD-9 coverage; treating unmapped administrative codes as anomalies.
>
> _History:_ Phase 2b reached only 0.51/0.54 (Status B); the sections below predate
> the fix and describe the earlier states.

**Bottom line (original Phase 2):** the ontology subsystem is a clean, canonical,
tested code layer. As of Phase 2 it was **not yet backed by real licensed ontology
data**. Status at Phase 2: **code-complete, data-blocked** (now superseded by 2b).

## 1. What was implemented

| Module | Role | State |
|---|---|---|
| `src/ontology/records.py` | Canonical data model: `ClinicalRecord`, `OntologyConcept`, `OntologyViolation`, `OntologyRuleResult` | New |
| `src/ontology/types.py` | Backward-compat shim re-exporting from `records.py` | Rewritten |
| `src/ontology/index.py` | `OntologyIndex` + transitive ancestors/descendants + crosswalk lookups | Extended |
| `src/ontology/rule_engine.py` | `OntologyRule`, `DemographicRule`, `RequiredCodesRule`, `MutualExclusionRule` (hierarchy-aware) | New |
| `src/ontology/engine.py` | `OntologyEngine` (scoring/checks/replacements) | Import fixed |
| `src/ontology/loader.py` | `load_ontology_index` / `load_ontology_engine` (+ crosswalk maps) | Import fixed, extended |
| `src/ontology/distance.py` | `shortest_path_distance`, `ancestor_distance`, `neighborhood` | New |
| `src/ontology/rules.py` | Legacy ICD-prefix `compute_s_ont` **fallback** | Untouched, labelled legacy |
| `src/ontology/__init__.py` | Canonical public surface | Rewritten |

### The two `OntologyViolation` types (intentional)
- `records.OntologyViolation` — **canonical** engine violation (`rule_id, kind, message, codes, severity`).
- `rules.OntologyViolation` — **legacy** token-weight violation, internal to the
  ICD-prefix `compute_s_ont` fallback only. New code should use the canonical one.

## 2. Broken import: fixed
Previously `engine.py` imported `OntologyRule` and `loader.py` imported
`DemographicRule/MutualExclusionRule/RequiredCodesRule` from `rules.py`, where
those classes never existed → `tests/test_ontology.py` failed at collection.
They now live in `rule_engine.py` and the imports point there. `test_ontology.py`
collects and passes (9/9).

## 3. Ontology assets found

| Asset | Present? |
|---|---|
| Raw UMLS MRCONSO / SNOMED RF2 / RxNorm | **No** |
| Processed `snomed_hierarchy.json`, `snomed_terms.json` | **No** |
| Processed `icd9_to_snomed.json`, `icd10_to_snomed.json` | **No** |
| Processed `drugname_to_rxcui.json` | **No** |

`ontologies/` and `ontologies/processed/` contain only `.gitkeep`. The parsers to
produce these (`scripts/parse_snomed.py`, `src/preprocessing/build_umls_maps.py`,
`scripts/parse_rxnorm.py`, `src/preprocessing/build_rxnorm_maps.py`) exist but
were never run because the licensed sources are absent.

## 4. Coverage: real or fixture-only?
**Neither real nor fixture-faked.** `scripts/build_ontology_coverage_report.py`
reports `status: blocked_missing_real_ontology_assets`. Real coverage numbers are
intentionally `null`. The code path that computes real coverage is implemented and
will run once assets exist; it is exercised in tests only against synthetic fixtures.

## 5. Token format reminder (for Phase 2b mapping)
Processed EHR tokens are raw ICD/drug-name, e.g. `DX_9_4019`, `DX_10_E785`,
`PROC_9_xxxx`, `MED_ACETAMINOPHEN`. Mapping to SNOMED/RxNorm requires
normalization (e.g. `DX_9_4019` → ICD-9 `401.9`/`4019`; `MED_ACETAMINOPHEN` →
`ACETAMINOPHEN` → RxCUI). The coverage script implements a first-pass
normalization; it should be hardened in Phase 2b against the real crosswalks.

## 6. What remains (Phase 2b / Phase 3)
- **Phase 2b (data):** acquire UMLS/SNOMED/RxNorm, run parsers into
  `ontologies/processed/`, re-run the coverage report for real numbers, and tune
  token→code normalization until coverage meets thresholds (diagnosis ≥ 0.80,
  medication ≥ 0.70).
- **Then:** implement ontology-backed anomaly-v2 injectors (Phase 1b finish) using
  real concept sets, and wire the canonical engine into `src/scoring/ontology_aware.py`
  to replace the ICD-prefix fallback.
- **Phase 3:** non-circular detector + calibrated scoring on the v2 benchmark.

## 7. Is it safe to build anomaly-v2 with ontology-backed rules now?
**No** — the engine is ready but the real concept data is not. Keep the v2
injectors guarded until Phase 2b loads real assets.
