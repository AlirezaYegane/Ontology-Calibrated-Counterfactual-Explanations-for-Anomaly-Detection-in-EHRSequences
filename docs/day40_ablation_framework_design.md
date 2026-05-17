# Day 40 — Ablation Framework Design

## Goal
Day 40 implements the ablation framework required for a paper-style evaluation of the ontology-calibrated counterfactual explanation system.

The goal is not to make a new claim yet. The goal is to make the comparison framework reproducible.

## Variants

| Variant | Formula | Purpose |
|---|---|---|
| full_model_conservative | 0.55 Sdet + 0.40 Sont + 0.05 Sgen | Current full system with conservative Sgen weighting |
| no_ontology | 0.90 Sdet + 0.10 Sgen | Measures what is lost when ontology reasoning is removed |
| no_generative | 0.60 Sdet + 0.40 Sont | Measures whether the system remains strong without diffusion/Sgen |
| detector_only | Sdet | Statistical detector baseline |
| ontology_only | Sont | Rule/ontology-only baseline |
| generative_only | Sgen | Diffusion/generative-only baseline |
| vae_replacement_slot | 0.55 Sdet + 0.40 Sont + 0.05 Svae | Framework slot for replacing diffusion with a simpler VAE score |

## Why conservative Sgen?
Earlier generative evaluation showed that the current raw diffusion Sgen proxy is weak as a standalone discriminator. Therefore, Day 40 avoids overclaiming and keeps Sgen as a low-weight auxiliary signal.

## Metrics
The script reports:

- ROC-AUC
- Average Precision
- best-F1 threshold
- precision / recall / F1
- precision at top 1%, 5%, and 10%
- mean score for normal vs anomaly rows
- effect size

## Outputs
The framework writes:

- `artifacts/day40/ablation_results.csv`
- `artifacts/day40/variant_scores.csv`
- `artifacts/day40/day40_ablation_summary.json`
- `artifacts/day40/day40_ablation_report.md`

## Paper usage
The result can become the ablation table in the paper after Day 41 reruns it on the final selected evaluation split.
