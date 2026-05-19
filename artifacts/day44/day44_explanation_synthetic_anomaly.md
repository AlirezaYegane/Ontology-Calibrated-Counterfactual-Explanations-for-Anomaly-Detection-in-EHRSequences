# Day 44 High-Risk Synthetic Anomaly Explanation

## Record

- Record identifier: `synthetic_anomaly`
- Source artifact: `artifacts\day36_repair_ready\repair_ready_scores.csv`
- Row index: `55318`
- Label: `1`
- Anomaly type / variant: `demographic_conflict`
- Interface risk band: **high**

## Score Components

- Sdet / detector score: 1.000000
- Sont / ontology score: n/a
- Sgen / generative score: 0.000000  _(diagnostic only)_
- Scal / calibrated score: n/a

## Human-Readable Interpretation

- No calibrated score was available, so the detector score 1.0000 is used as the visible primary anomaly signal. This places the record in the high band.
- No ontology score was available for this row.
- The generative score is shown as a diagnostic auxiliary signal only; it should not be over-interpreted as the main evidence source.
- No explicit counterfactual edit list was found in this artifact row.

## Record Preview

### `sequence_tokens`
- `DX_9_99649`
- `DX_9_73381`
- `DX_9_E8788`
- `DX_9_4019`
- `DX_9_53081`
- `DX_9_V4975`
- `DX_9_V220`
- `PROC_9_7855`
- `MED_POTASSIUM_CHL_20_MEQ_1000_ML_D5_1_2_NS`
- `MED_OXYCODONE_IMMEDIATE_RELEASE`
- `MED_DOCUSATE_SODIUM`
- `MED_TRAZODONE`
- `MED_SENNA`
- `MED_ENOXAPARIN_SODIUM`
- `MED_SODIUM_CHLORIDE_0_9_FLUSH`
- `MED_MICONAZOLE_POWDER_2`
- `MED_ACETAMINOPHEN`
- `MED_FUROSEMIDE`
- `MED_METOPROLOL_SUCCINATE_XL`
- `MED_OMEPRAZOLE`
- `MED_LISINOPRIL`
- `MED_MULTIVITAMINS`
- `MED_DIAZEPAM`
- `MED_OXYCODONE_IMMEDIATE_RELEASE`
- `MED_OXYCODONE_IMMEDIATE_RELEASE`

## Paper-Facing Note

This Day 44 interface is designed for reproducible qualitative inspection. It exposes the score components and available counterfactual evidence without inventing missing values. In the paper, these outputs can support case-study examples, explanation audits, and demo screenshots.
