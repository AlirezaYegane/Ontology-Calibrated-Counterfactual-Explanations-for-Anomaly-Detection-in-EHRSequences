# Day 35 — Detector + Diffusion + Ontology Scoring Integration

## Status
Complete.

## Goal
Integrate detector score `Sdet`, ontology score `Sont`, and generative surprise `Sgen` into a calibrated anomaly score `Scal`.

## Scoring formula
`Scal = w_det * Sdet + w_ont * Sont + w_gen * Sgen`

## Main design decision
Day 34 showed that the current diffusion `Sgen` proxy is close to random, so `Sgen` is kept as a low-weight diagnostic signal for this milestone.

## Important honesty note
`Sdet` is a real model score from the recovered Day 20 supervised detector.

`Sont` in this Day 35 artifact is a synthetic-benchmark proxy derived from the injected anomaly family (`anomaly_type`). It is useful for testing the calibrated scoring machinery, but it should not be overclaimed as a full independent clinical ontology checker yet.

## Best calibrated weights
- `w_det`: 0.0000
- `w_ont`: 1.0000
- `w_gen`: 0.0000

## Metrics

| Signal | ROC-AUC | Average Precision | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Detector only | 0.8002 | 0.7332 | 0.6754 | 0.9045 | 0.5389 |
| Sont proxy only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Sgen only | 0.5000 | 0.2299 | 0.3739 | 0.2299 | 1.0000 |
| Calibrated Scal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Output files
- `artifacts/day35/calibrated_scores.csv`
- `artifacts/day35/day35_weight_search.csv`
- `artifacts/day35/day35_calibrated_scoring_summary.json`
- `artifacts/day35/README.md`

## Next step
Day 36 should implement the counterfactual generator using the calibrated score as the candidate-ranking objective.

## Scientific caveat
The perfect `Sont proxy` and `Scal` metrics should be interpreted carefully.

In this Day 35 milestone, `Sont_proxy` is derived from the injected synthetic anomaly family (`anomaly_type`). Therefore, it behaves like a benchmark-side ontology oracle/proxy rather than a fully independent ontology-rule engine.

The most scientifically meaningful standalone model result remains:

- Detector-only ROC-AUC: 0.8002
- Detector-only Average Precision: 0.7332
- Detector-only F1: 0.6754

The main value of Day 35 is that the calibrated scoring machinery now works end-to-end:

`Sdet + Sont + Sgen -> Scal`

For future/reporting use:
- use `Sdet` as the real learned anomaly score,
- use `Sont_proxy` as a controlled synthetic benchmark signal,
- keep `Sgen` diagnostic until a stronger generative surprise definition is implemented,
- implement an independent ontology-rule checker before claiming clinical ontology-score performance.
