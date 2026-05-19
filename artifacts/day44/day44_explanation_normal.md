# Day 44 Low-Risk Normal Record Explanation

## Record

- Record identifier: `normal`
- Source artifact: `artifacts\day36_repair_ready\repair_ready_scores.csv`
- Row index: `0`
- Label: `0`
- Anomaly type / variant: `not specified`
- Interface risk band: **low**

## Score Components

- Sdet / detector score: 0.327263
- Sont / ontology score: n/a
- Sgen / generative score: 0.000000  _(diagnostic only)_
- Scal / calibrated score: n/a

## Human-Readable Interpretation

- No calibrated score was available, so the detector score 0.3273 is used as the visible primary anomaly signal. This places the record in the low band.
- No ontology score was available for this row.
- The generative score is shown as a diagnostic auxiliary signal only; it should not be over-interpreted as the main evidence source.
- No explicit counterfactual edit list was found in this artifact row.

## Record Preview

### `sequence_tokens`
- `DX_10_I509`
- `DX_10_K7200`
- `DX_10_R570`
- `DX_10_N179`
- `DX_10_I2510`
- `DX_10_Z9861`
- `DX_10_I255`
- `DX_10_E785`
- `DX_10_E118`
- `DX_10_E669`
- `DX_10_I129`
- `DX_10_N189`
- `DX_10_J449`
- `DX_10_R7989`
- `DX_10_I4891`
- `DX_10_Z515`
- `DX_10_Z66`
- `DX_10_I447`
- `DX_10_I340`
- `DX_10_I272`
- `DX_10_I071`
- `MED_HEPARIN`
- `MED_0_9_SODIUM_CHLORIDE`
- `MED_NOREPINEPHRINE`
- `MED_INFLUENZA_VACCINE_QUADRIVALENT`

## Paper-Facing Note

This Day 44 interface is designed for reproducible qualitative inspection. It exposes the score components and available counterfactual evidence without inventing missing values. In the paper, these outputs can support case-study examples, explanation audits, and demo screenshots.
