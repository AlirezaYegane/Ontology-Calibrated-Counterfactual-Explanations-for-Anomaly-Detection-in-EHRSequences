# Phase 2b — Real Ontology Asset Population, Parsing, and Coverage

## Status: B — Partially complete real ontology integration

Real licensed ontology assets were found, validated, extracted, and parsed into
real crosswalk maps. Coverage is now **real** (not faked) but **below target**.

## Raw assets (validated)
All six expected zips are structurally valid (central-directory readable):
ICD-9-CM, ICD-10-CM (descriptions + tabular/index), RxNorm full, SNOMED CT US,
UMLS 2026AA metathesaurus-full (5.8 GB). One stray `login.html` (4.6 KB) under
`umls/2026AA/` was detected and ignored (not a zip). The stray root-level
`ontologies/raw/icd10cm/` zips were moved into `ontologies/raw/icd/icd10cm/`.
See `raw_asset_inventory.{json,md}`.

## Parsed → `ontologies/processed/`
| File | Entries |
|---|---|
| `snomed_hierarchy.json` | 641,727 IS-A edges (parents 386,109 / children 135,836) |
| `snomed_terms.json` | 386,064 |
| `icd9_to_snomed.json` | 8,818 |
| `icd10_to_snomed.json` | 11,339 |
| `drugname_to_rxcui.json` | 178,692 |
| `rxnorm_terms.json` | 128,678 |
| `rxnorm_classes.json` | **EMPTY — unavailable** (needs RXNREL/RXNSAT/ATC, not extracted) |
| `ontology_asset_manifest.json` | provenance + counts + sha1 heads |

## Real coverage (mimiciv_val.pkl, 30,000 rows)
| Metric | Coverage | Target | Met? |
|---|---:|---:|:--:|
| Diagnosis | **0.5052** | 0.80 | ❌ |
| Medication | **0.5400** | 0.70 | ❌ |

(Procedures not mapped in Phase 2.) See `coverage_report.json`, `unmapped_codes.csv`.

## Why coverage is below target (honest diagnosis)
- **Diagnosis (0.51):** the UMLS *shared-CUI* crosswalk only links ~8.8k ICD-9 and
  ~11.3k ICD-10 codes to SNOMED. Top unmapped tokens are Z/V/E/F **administrative &
  history codes** (`Z87891`, `Z20822`, `V1582`, `F17210`, …) that genuinely lack
  SNOMED disease concepts. Token normalization was verified correct (`E78.5`,
  `I50.9` map). **Fix:** add the UMLS **MRMAP** SNOMED↔ICD map (148 MB, already in
  the zip) and/or MRREL relations.
- **Medication (0.54):** MIMIC drug tokens are formulation/brand/descriptor strings
  (`MED_0_9_SODIUM_CHLORIDE`, `MED_HYDROMORPHONE_DILAUDID`,
  `MED_OXYCODONE_IMMEDIATE_RELEASE`) and bare `MED_INSULIN`, not clean RxNorm
  ingredient names. **Fix:** ingredient-level (IN/PIN) longest-match + strip
  concentration/route/form descriptors.

## Honesty note
Coverage is real and below target. Do **not** claim full ontology integration.
Correct current claim: *"real SNOMED/RxNorm crosswalks are integrated; diagnosis
coverage ≈ 0.51 and medication ≈ 0.54 on MIMIC-IV, limited by UMLS crosswalk
completeness and MIMIC drug-name formatting; improvements identified."*

## Next step
Improve crosswalk completeness (MRMAP) and medication ingredient matching to lift
coverage toward targets, then proceed to Phase 1b ontology-backed anomaly-v2. Do
not advance to Phase 3 / H200 yet.
