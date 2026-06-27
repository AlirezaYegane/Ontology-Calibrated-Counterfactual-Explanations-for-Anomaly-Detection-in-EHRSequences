# Residual Unmapped Analysis (Phase 2b-fix)

## Diagnosis unmapped (by occurrence-weighted category)

- `icd9_disease_numeric`: 39936 occ
- `icd9_status_history(V)`: 8819 occ
- `icd10_injury_poisoning`: 6620 occ
- `icd9_external_cause(E)`: 5163 occ
- `icd10_status_history_screening`: 3336 occ
- `icd10_external_cause`: 2725 occ
- `icd10_disease`: 2453 occ
- `icd10_symptoms_signs`: 482 occ

### Top unmapped diagnosis tokens

| token | count | category |
|---|---:|---|
| DX_9_25000 | 2344 | icd9_disease_numeric |
| DX_9_311 | 1961 | icd9_disease_numeric |
| DX_9_40390 | 1197 | icd9_disease_numeric |
| DX_9_49390 | 1061 | icd9_disease_numeric |
| DX_9_V4582 | 891 | icd9_status_history(V) |
| DX_9_496 | 822 | icd9_disease_numeric |
| DX_9_V4581 | 729 | icd9_status_history(V) |
| DX_9_41400 | 676 | icd9_disease_numeric |
| DX_10_Y92230 | 590 | icd10_external_cause |
| DX_9_V1251 | 579 | icd9_status_history(V) |
| DX_9_60000 | 568 | icd9_disease_numeric |
| DX_9_40391 | 488 | icd9_disease_numeric |
| DX_9_E8497 | 460 | icd9_external_cause(E) |
| DX_9_71590 | 433 | icd9_disease_numeric |
| DX_9_4168 | 411 | icd9_disease_numeric |

## Medication unmapped (by occurrence-weighted category)

- `iv_fluid_electrolyte`: 96790 occ
- `brand_combo_or_missing_synonym`: 86867 occ
- `insulin_class_ambiguous`: 49082 occ
- `biologic_vaccine`: 15048 occ

### Top unmapped medication tokens

| token | count | category |
|---|---:|---|
| MED_INSULIN | 46651 | insulin_class_ambiguous |
| MED_5_DEXTROSE | 20275 | iv_fluid_electrolyte |
| MED_BAG | 19418 | iv_fluid_electrolyte |
| MED_SENNA | 18840 | brand_combo_or_missing_synonym |
| MED_ISO_OSMOTIC_DEXTROSE | 16759 | iv_fluid_electrolyte |
| MED_LACTATED_RINGERS | 13953 | iv_fluid_electrolyte |
| MED_DEXTROSE_50 | 8850 | iv_fluid_electrolyte |
| MED_VIAL | 8537 | brand_combo_or_missing_synonym |
| MED_POLYETHYLENE_GLYCOL | 7570 | brand_combo_or_missing_synonym |
| MED_SW | 6933 | brand_combo_or_missing_synonym |
| MED_NS | 4994 | brand_combo_or_missing_synonym |
| MED_INFLUENZA_VACCINE_QUADRIVALENT | 4833 | biologic_vaccine |
| MED_NEUTRA_PHOS | 4696 | brand_combo_or_missing_synonym |
| MED_D5_1_2NS | 4282 | iv_fluid_electrolyte |
| MED_MULTIVITAMINS | 4161 | brand_combo_or_missing_synonym |

## Interpretation

- **Diagnosis (0.80, target met).** ICD-10 coverage is 0.92 via the authoritative SNOMED->ICD-10-CM MRMAP. Residual unmapped diagnosis is dominated by ICD-9 disease codes that neither share a SNOMED CUI nor bridge through an MRMAP-mapped ICD-10 sibling, plus ICD-9/10 status/history (V/Z) and external-cause (E/V-Y) codes that have no SNOMED disease concept. None of these are mapped to spurious concepts.
- **Medication (0.78, target met).** Mapped via exact RxNorm drug-name + conservative longest-span IN/PIN ingredient match. Residual unmapped is dominated by IV fluids/electrolytes, vaccines/biologics, brand/combo strings, and bare `INSULIN` (correctly skipped as ambiguous: RxNorm has no bare 'insulin' ingredient, only specific insulins). No fragment was force-mapped.