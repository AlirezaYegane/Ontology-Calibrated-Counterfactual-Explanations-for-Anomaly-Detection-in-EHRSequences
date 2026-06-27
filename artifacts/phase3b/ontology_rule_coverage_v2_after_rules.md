# Phase 3b -- Real Ontology Rule Coverage (after rule packs)

**Split:** `test`  |  **records:** 6307  
**Normal false-positive rate:** 0.1304  
**Families still uncovered (<5%):** none

## Violation rate / mean S_ont by class

| class | n | violation_rate | mean_S_ont | rules fired |
|---|---:|---:|---:|---|
| normal | 4931 | 0.1304 | 0.0695 | medication_required_context=338, diabetes_type_exclusion=331, sex_restricted_concepts=8 |
| demographic_incompatibility | 272 | 1.0 | 1.0662 | sex_restricted_concepts=272, medication_required_context=18, diabetes_type_exclusion=18 |
| medication_indication_mismatch | 875 | 0.5006 | 0.2731 | medication_required_context=284, diabetes_type_exclusion=190, sex_restricted_concepts=2 |
| forbidden_cooccurrence | 229 | 1.0 | 0.5066 | diabetes_type_exclusion=229, medication_required_context=3 |

## Rule manifest

| rule_id | type | severity | sizes | limitations |
|---|---|---:|---|---|
| sex_restricted_concepts | sex_restriction | 1.0 | n_female_icd_families=82, n_male_icd_families=31, n_female_clean_snomed_concepts=110, n_male_clean_snomed_concepts=0 | Fires on the presence of an opposite-sex diagnosis-code FAMILY in the source tokens; this ... |
| medication_required_context | required_context | 0.5 | n_drugs=9, n_anticoag_context_concepts=1693, n_hypothyroid_context_concepts=196, n_diabetes_context_concepts=732 | WEAK signal (severity 0.5): drugs have secondary/off-label uses and coding is incomplete, ... |
| diabetes_type_exclusion | group_mutual_exclusion | 0.5 | n_type1_concepts=24, n_type2_concepts=16 | MEDIUM severity (0.5): real EHRs sometimes co-code both types during diagnostic transition... |