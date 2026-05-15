# Day 35.6 — Curated Independent Ontology/Rule Scorer

## Status
Complete.

## Scientific purpose
Day 35.5 showed that purely data-mined co-occurrence rules were too weak. Day 35.6 introduces curated content-only clinical rules.

## No-leakage policy
Scoring uses only `sequence_tokens`.

The scorer does not use:

- `label`
- `source`
- `anomaly_type`

Those fields are used only after scoring for evaluation and breakdown.

## Curated rule families
- isolated pregnancy/delivery signal
- pregnancy + male-specific code contradiction
- obstetric intervention without pregnancy diagnosis
- chemotherapy without cancer diagnosis
- weak insulin/diabetes indication mismatch
- weak anticoagulation/cardiovascular indication mismatch

## Best calibrated weights
- `w_det`: 0.7143
- `w_ont`: 0.2857
- `w_gen`: 0.0000

## Held-out metrics

| Signal | ROC-AUC | AP | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Detector only | 0.7983 | 0.7310 | 0.6748 | 0.9045 | 0.5381 |
| Curated Sont only | 0.6591 | 0.4745 | 0.4444 | 0.5640 | 0.3666 |
| Sdet + curated Sont | 0.7993 | 0.7315 | 0.6754 | 0.8897 | 0.5443 |
| Oracle proxy reference | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Rule-flag quality on held-out split
- flagged count: 16539
- flagged precision: 0.3057
- flagged recall: 0.6150

## Interpretation
For publication, the main comparison is detector-only vs `Sdet + curated independent Sont`.

The oracle proxy remains only an upper-bound reference.

## Scientific conclusion from Day 35.6
The curated content-only ontology/rule scorer is no longer random.

Held-out results:

- Detector only ROC-AUC: 0.7983
- Detector only AP: 0.7310
- Curated Sont only ROC-AUC: 0.6591
- Curated Sont only AP: 0.4745
- Sdet + curated Sont ROC-AUC: 0.7993
- Sdet + curated Sont AP: 0.7315

This shows that curated ontology-style rules provide a weak but real independent signal. However, the global gain over the detector is small, and rule-level false positives remain high.

The main noisy rules are:
- anticoag_or_antiplatelet_without_cardiovascular_dx
- diabetes_med_without_diabetes_dx

Next step:
Run a stricter rule ablation that keeps only high-confidence curated rules and reports per-rule precision.
