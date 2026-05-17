# Day 41 — Ablation Study Results

- Generated at: `2026-05-17T05:34:40+00:00`
- Input scores: `artifacts\day40\variant_scores.csv`
- Rows evaluated: **71503**
- Label column: `label`
- Evaluated variants: **9**
- Best variant: **no_generative**
- Best ROC-AUC: **0.8010**
- Best Average Precision: **0.7334**

## Main Comparative Table

| variant | roc_auc | average_precision | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| no_generative | 0.8010 | 0.7334 | 0.6201 | 0.6201 | 0.6201 |
| Svae_norm | 0.8010 | 0.7337 | 0.6208 | 0.6208 | 0.6208 |
| vae_replacement_slot | 0.8010 | 0.7332 | 0.6198 | 0.6198 | 0.6198 |
| full_model_conservative | 0.8010 | 0.7332 | 0.6196 | 0.6196 | 0.6196 |
| Sdet_norm | 0.8002 | 0.7332 | 0.6196 | 0.6196 | 0.6196 |
| no_ontology | 0.8002 | 0.7332 | 0.6196 | 0.6196 | 0.6196 |
| detector_only | 0.8002 | 0.7332 | 0.6196 | 0.6196 | 0.6196 |
| Sont_norm | 0.6578 | 0.4756 | 0.3051 | 0.6112 | 0.4070 |
| ontology_only | 0.6578 | 0.4756 | 0.3051 | 0.6112 | 0.4070 |

## Interpretation

- Full model minus no-ontology ROC-AUC delta: 0.0008.
- Full model minus no-generative ROC-AUC delta: -0.0000.
- The full ontology-calibrated variant is not the top-ranked setting by ROC-AUC; report this as an honest ablation finding.

## Scientific Reporting Note

This Day 41 run uses the Day 40 ablation score artifact directly. Raw diffusion `Sgen` should be interpreted conservatively because the earlier Day 34 evaluation found weak standalone generative separation.

## Generated Artifacts

- `artifacts/day41/day41_ablation_results.csv`
- `artifacts/day41/day41_ablation_summary.json`
- `artifacts/day41/day41_ablation_tables.tex`
