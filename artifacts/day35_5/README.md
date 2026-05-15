# Day 35.5 — Independent Ontology/Rule Scorer

## Status
Complete.

## Scientific purpose
Day 35 used an ontology proxy derived from `anomaly_type`, which is useful as an oracle-style integration check but not a publishable independent ontology score.

Day 35.5 replaces that with an independent content-only `Sont_independent` scorer.

## No-leakage rule
The scorer is learned from normal training records only and scores validation records using:

- `sequence_tokens`
- learned medication/procedure → expected diagnosis rules

The scorer does **not** use the following during scoring:

- `label`
- `anomaly_type`
- `source`

Those fields are used only after scoring for evaluation and breakdown.

## Best calibrated weights from calibration split
- `w_det`: 1.0000
- `w_ont`: 0.0000
- `w_gen`: 0.0000

## Held-out evaluation metrics

| Signal | ROC-AUC | Average Precision | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Detector only | 0.7983 | 0.7310 | 0.6748 | 0.9045 | 0.5381 |
| Independent Sont only | 0.5081 | 0.2373 | 0.3739 | 0.2299 | 1.0000 |
| Sdet + independent Sont | 0.7983 | 0.7310 | 0.6748 | 0.9045 | 0.5381 |
| Oracle Sont proxy reference | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Interpretation
The publishable comparison is `Detector only` versus `Sdet + independent Sont`.

The oracle proxy is retained only as an upper-bound/sanity-check reference and must not be presented as independent ontology performance.

## Output files
- `artifacts/day35_5/val_independent_sont_scores.csv`
- `artifacts/day35_5/learned_indication_rules.json`
- `artifacts/day35_5/learned_indication_rules.csv`
- `artifacts/day35_5/day35_5_scientific_summary.json`
- `artifacts/day35_5/paper_ready_metrics.csv`

## Scientific conclusion from Day 35.5
The independent content-only ontology/rule scorer did not improve performance over the supervised detector.

Held-out comparison:

- Detector only ROC-AUC: 0.7983
- Detector only AP: 0.7310
- Independent Sont ROC-AUC: 0.5081
- Independent Sont AP: 0.2373
- Calibrated independent Scal ROC-AUC: 0.7983
- Calibrated independent Scal AP: 0.7310

The calibration grid assigned zero weight to the independent Sont signal:

- w_det = 1.0
- w_ont = 0.0
- w_gen = 0.0

Interpretation:
The simple data-mined medication/procedure-to-diagnosis co-occurrence rules are not strong enough to provide a useful independent ontology score. This is scientifically useful because it rules out a weak approximation and motivates the next step: a curated independent ontology/rule scorer based on explicit clinical anomaly constraints rather than label-derived proxies.
