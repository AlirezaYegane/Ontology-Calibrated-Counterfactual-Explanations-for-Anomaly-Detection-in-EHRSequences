# Day 40 — Ablation Framework Report

- Generated at: `2026-05-17T05:22:11+00:00`
- Input scores: `artifacts/day36_repair_ready/repair_ready_scores.csv`
- Rows evaluated: **71503**
- Positive/anomaly rows: **16442**
- Negative/normal rows: **55061**

## Column Resolution

- Label column: `label` (direct_label_column)
- Detector score column: `Sdet`
- Ontology score column: `Sont_curated_raw`
- Generative score column: `Sgen`
- VAE score column: `None`
- True VAE score used: **False**
- VAE proxy used: **True**

## Warnings

- No true VAE score column was found. VAE replacement can only run as a proxy slot.

## Main Ablation Results

| Variant | ROC-AUC | AP | F1 | Precision | Recall | P@5% | Effect Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_generative | 0.8010 | 0.7334 | 0.6763 | 0.9216 | 0.5342 | 1.0000 | 1.4651 |
| no_ontology | 0.8002 | 0.7332 | 0.6751 | 0.9053 | 0.5382 | 0.9997 | 1.4884 |
| detector_only | 0.8002 | 0.7332 | 0.6751 | 0.9053 | 0.5382 | 0.9997 | 1.4884 |
| vae_replacement_slot | 0.8010 | 0.7332 | 0.6763 | 0.9216 | 0.5342 | 1.0000 | 1.4618 |
| full_model_conservative | 0.8010 | 0.7332 | 0.6763 | 0.9216 | 0.5342 | 1.0000 | 1.4603 |
| ontology_only | 0.6578 | 0.5185 | 0.4438 | 0.5611 | 0.3670 | 0.9348 | 0.7358 |
| generative_only | 0.5000 | 0.1250 | 0.3739 | 0.2299 | 1.0000 | 0.0000 | nan |

## Scientific Interpretation

- `full_model_conservative` is the current paper-safe full system score. It gives high weight to detector and ontology, and only low weight to generative Sgen.
- `no_ontology` tests whether ontology violations are actually contributing beyond statistical scoring.
- `no_generative` tests whether the system remains strong without diffusion/Sgen.
- `detector_only`, `ontology_only`, and `generative_only` are single-component baselines.
- `vae_replacement_slot` is the framework slot for replacing diffusion with a simpler VAE-style generative score. If no real VAE score column exists, this row is marked as proxy and must not be claimed as a trained VAE baseline.

## Paper Use

Use this table as the Day 40 ablation-framework artifact. Day 41 should run the same framework on the final selected evaluation split and, if possible, replace the VAE proxy with a trained VAE reconstruction score.

