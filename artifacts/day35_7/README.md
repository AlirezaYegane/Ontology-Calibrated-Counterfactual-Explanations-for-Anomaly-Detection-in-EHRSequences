# Day 35.7 — Strict Curated Sont Ablation

## Status
Complete.

## Goal
Refine Day 35.6 by removing noisy weak rules and evaluating stricter ontology-rule variants.

## Best mode
`strong_weighted`

## Best mode held-out metrics
- ROC-AUC: 0.7982
- AP: 0.7312
- F1: 0.6759
- Precision: 0.9039
- Recall: 0.5397

## Interpretation
This artifact separates global ranking performance from rule-level high-precision behavior.

The publishable result should compare:
- detector-only
- detector + strict curated ontology score
- rule-level precision/recall for interpretability

No `label`, `source`, or `anomaly_type` is used during scoring.

## Scientific conclusion from Day 35.7
The strict curated ontology scorer is not a strong standalone global ranker, but it provides a high-precision interpretable subset.

Key held-out results:

- Detector-only AP: 0.730970
- Strict Sont-only AP: 0.406654
- Detector + strict Sont AP: 0.731192
- Detector + strong-weighted Sont AP: 0.731201

The global ranking gain is small, so the paper should not overclaim a large performance improvement.

The strongest publishable contribution is interpretability:

- strict/no-noisy rule flags have precision around 0.9328
- pregnancy_male_specific_conflict has precision 1.0000
- isolated_pregnancy_signal has precision 0.9603

Paper-safe claim:
Curated ontology rules provide a high-precision explanatory layer that complements the supervised detector, even though the global ranking improvement is modest.
