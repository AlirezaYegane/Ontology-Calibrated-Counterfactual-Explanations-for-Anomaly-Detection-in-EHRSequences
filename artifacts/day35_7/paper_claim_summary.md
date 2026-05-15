# Day 35.7 — Paper-Oriented Scientific Interpretation

## Main scientific result

The supervised detector remains the strongest global anomaly ranking signal.

However, strict curated ontology rules provide a high-precision interpretable anomaly subset.

## Held-out global ranking comparison

| Signal | ROC-AUC | AP | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Detector only | 0.7983 | 0.7310 | 0.6748 | 0.9045 | 0.5381 |
| Strict Sont only | 0.6230 | 0.4067 | 0.3961 | 0.9328 | 0.2514 |
| Detector + strict Sont | 0.7982 | 0.7312 | 0.6757 | 0.9042 | 0.5394 |
| Detector + strong-weighted Sont | 0.7982 | 0.7312 | 0.6759 | 0.9039 | 0.5397 |

## Rule-level interpretability

The strongest curated rules are:

| Rule | Precision | Interpretation |
|---|---:|---|
| pregnancy_male_specific_conflict | 1.0000 | very high-confidence coded demographic/clinical contradiction |
| isolated_pregnancy_signal | 0.9603 | strong signal for injected demographic-conflict anomalies |

## Paper-safe interpretation

The calibrated ontology component does not substantially improve global anomaly ranking over the supervised detector. The improvement in AP and F1 is small.

The stronger contribution is interpretability: strict curated rules identify a smaller set of anomalies with high precision, especially demographic-conflict anomalies, without using labels, source, or anomaly_type during scoring.

## Safe claim

Strict curated ontology rules are best interpreted as a high-precision explanatory layer rather than as a replacement for the learned detector.

## Limitation

Because the current supervised dataset contains only sequence tokens and not demographic metadata such as sex or age, demographic contradictions must be inferred from coded clinical content rather than patient attributes. Future work should integrate real demographic metadata and richer clinical ontology mappings.
