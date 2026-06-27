# Phase 2b-fix — Close Real Ontology Coverage Gaps

## Status: A — Phase 2 fully closed (both targets met with real ontology data)

| Metric | Before (2b) | After (2b-fix) | Target | Met? |
|---|---:|---:|---:|:--:|
| Diagnosis coverage | 0.5052 | **0.8006** | 0.80 | ✅ |
| Medication coverage | 0.5400 | **0.7767** | 0.70 | ✅ |

Computed on `data/processed/mimiciv_val.pkl` (30,000 rows) from real UMLS 2026AA /
SNOMED CT US / RxNorm assets. No fabricated or fuzzy mappings.

## What changed
1. **ICD-10 → SNOMED via MRMAP** (`META/MRMAP.RRF`, `SNOMEDCT_US`→ICD-10-CM,
   `TOTYPE=SDUI`, excluding `REL=XR`). ICD-10 map 11,339 → **30,783** codes;
   ICD-10 diagnosis coverage 0.53 → **0.92**.
2. **ICD-9 → SNOMED bridge** through CUI-linked ICD-10 siblings + MRMAP
   (authoritative UMLS synonymy, no fuzzy matching). ICD-9 map 8,818 → **10,254**;
   recovered high-frequency disease codes (272.4, 414.01, 401.9, …).
3. **RxNorm ingredient matching** — built an IN/PIN ingredient index (18,278) and a
   conservative longest-span matcher: maps `MED_0_9_SODIUM_CHLORIDE`→`SODIUM_CHLORIDE`,
   `MED_ACETAMINOPHEN_IV`→`ACETAMINOPHEN`, `MED_TRAMADOL_ULTRAM`→`TRAMADOL`,
   `MED_HYDROMORPHONE_DILAUDID`→`HYDROMORPHONE`, while **skipping** ambiguous
   `MED_INSULIN` / `MED_5_DEXTROSE`.

## Anti-inflation safeguards (coverage is defensible, not inflated)
- No ICD title-string or fuzzy-similarity matches counted as ontology mappings.
- Only authoritative UMLS CUI / MRMAP links used.
- Bare `INSULIN`/`DEXTROSE` left **unmapped** (no clean RxNorm ingredient).
- Single-word electrolyte fragments (`SODIUM`, `CHLORIDE`, …) never matched alone.
- Targets were **not** changed (0.80 / 0.70).

## Residual (see `residual_unmapped_analysis.{md,json}`)
- **Diagnosis residual:** remaining ICD-9 disease codes lacking any SNOMED CUI/bridge,
  plus V/Z status-history and E/V-Y external-cause codes that have no SNOMED disease
  concept (correctly not mapped).
- **Medication residual:** IV fluids/electrolytes, vaccines/biologics, brand/combo
  strings, and ambiguous `INSULIN` (correctly skipped).

## Artifacts
`coverage_report.json`, `unmapped_codes.csv`, `crosswalk_sanity.md`,
`icd_map_delta.json`, `icd9_bridge_delta.json`, `rxnorm_medication_delta.json`,
`residual_unmapped_analysis.{md,json}`, `mrmap_extraction_status.json`,
`phase2b_fix_summary.json`.

## Ontology claims now safe / not safe
- **Safe:** "real SNOMED/RxNorm integration with diagnosis coverage 0.80 (ICD-10 0.92)
  and medication coverage 0.78 on MIMIC-IV, via authoritative UMLS crosswalks."
- **Not safe:** RxNorm drug-class reasoning (`rxnorm_classes` unavailable); complete
  ICD-9 coverage; any claim that unmapped administrative codes are clinically anomalous.

## Next step
Phase 2 is closed. Proceed to **Phase 1b**: ontology-backed anomaly-v2 injectors using
the real engine + maps. Do not advance to Phase 3 / H200 yet.
