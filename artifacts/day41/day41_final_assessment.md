# Day 41 — Final Ablation Assessment

## Final interpretation

The Day 41 ablation study shows that the strongest configuration in this benchmark is `no_generative`, with `full_model_conservative` performing extremely close behind.

This suggests that the current detector and ontology-calibrated components provide the main anomaly-discrimination signal, while the current diffusion-based `Sgen` proxy does not add measurable value and may slightly dilute performance.

## Paper-ready wording

The ablation results indicate that ontology-aware and detector-based signals are the most reliable contributors to anomaly detection in the current implementation. The conservative full model performs competitively, but the no-generative variant achieves the highest ROC-AUC. This is consistent with the earlier generative evaluation, where the raw diffusion-based surprise score showed weak standalone separation. We therefore treat the generative component as a promising but currently limited module, requiring a stronger surprise-score formulation before it can be claimed as a major source of performance gain.

## Recommended claim strength

Use a conservative claim:

> The proposed framework benefits primarily from calibrated detector and ontology-based scoring. The current diffusion-derived surprise score is useful for system completeness and future counterfactual generation, but its standalone discriminative value is limited in the present benchmark.

Avoid this claim:

> The full diffusion-based ontology-calibrated model clearly outperforms all ablations.

## Key result summary

| Variant | Interpretation |
|---|---|
| `no_generative` | Best overall in this run |
| `full_model_conservative` | Very close, but slightly below no-generative |
| `detector_only` / `no_ontology` | Strong baseline, detector dominates |
| `ontology_only` | Weaker alone, but non-random signal |
| `generative_only` / `Sgen` | Not strong enough as standalone anomaly signal |

## Next implication
The next research step should improve the generative surprise definition rather than merely increasing the weight of `Sgen`.
