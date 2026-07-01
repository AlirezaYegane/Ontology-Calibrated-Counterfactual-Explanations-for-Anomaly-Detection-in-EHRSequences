# Phase 7 -- External Validation Status

**Status:** `external_validation_blocked_schema_mismatch`  |  dataset: eICU

eICU uses APACHE / body-system tokens (e.g. EICU_APACHE2_DX:*, EICU_BODYSYS:*), NOT ICD-10/ICD-9 or RxNorm. The ontology scorer + Phase 3b rule packs are keyed on ICD->SNOMED / drug->RxNorm mappings, which do not cover APACHE codes -> ~0 tokens map and ~0 rules fire. External validation would require (a) an APACHE->ICD/SNOMED crosswalk and (b) applying the benchmark-v2 anomaly injectors to eICU. Both are out of Phase 7 scope and are documented as future work.

## Empirical schema check
- sampled records: 500, unique tokens: 0
- example tokens: []
- ontology-mappable tokens: **0** / total ontology codes mapped: **0**
- records where the ontology fired: **0**
- schema compatible with ontology: **False**

**Recommended paper scope:** MIMIC-IV benchmark-v2 only; external validation = future work (pending APACHE crosswalk).