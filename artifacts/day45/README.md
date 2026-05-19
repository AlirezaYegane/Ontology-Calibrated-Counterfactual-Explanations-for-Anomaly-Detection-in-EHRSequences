# Day 45 — Comprehensive Test Set Evaluation

## Status

Complete.

## Goal

Evaluate available Day 41 score variants on the held-out/test score table using ROC-AUC, Average Precision, conservative threshold selection, and false-positive / false-negative behavior.

## Evaluated variants

| Variant | ROC-AUC | AP | Threshold | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| detector_only | 0.800219 | 0.733243 | 0.485000 | 0.801756 | 0.566476 | 0.663887 | 0.041826 | 0.433524 |
| no_ontology | 0.800219 | 0.733243 | 0.436500 | 0.801756 | 0.566476 | 0.663887 | 0.041826 | 0.433524 |
| generative_only | 0.500000 | 0.229948 | 0.000000 | 0.229948 | 1.000000 | 0.373916 | 1.000000 | 0.000000 |

## Best-performing variant

- Best variant by ROC-AUC/AP/F1 ranking: `detector_only`
- ROC-AUC: `0.800219`
- Average Precision: `0.733243`
- Selected threshold: `0.485000`
- Precision: `0.801756`
- Recall: `0.566476`
- F1: `0.663887`
- False-positive rate: `0.041826`


## Scientific caveat

The `detector_only` and `no_ontology` variants produce identical ROC-AUC, Average Precision, precision, recall, F1, FPR, and FNR after threshold calibration. This indicates that, in the available Day 41 wide-format score artifact, `no_ontology` preserves the same sample ranking as `detector_only`, likely through monotonic score scaling. Therefore, this evaluation supports detector-driven discrimination but should not be over-interpreted as proving that ontology information is ineffective.

The `generative_only` variant has no useful discrimination in this artifact. Its ROC-AUC of 0.500000 and FPR of 1.000000 arise because the available generative-only scores are constant at zero. This is useful negative evidence for the ablation section: the current generative-only score artifact does not provide independent anomaly ranking signal.

## Recommended paper wording

The detector-based score achieved ROC-AUC 0.8002 and Average Precision 0.7332 on the Day 45 held-out/test-style evaluation table. Under a conservative threshold selected with precision and false-positive constraints, it reached precision 0.8018, recall 0.5665, and FPR 0.0418. The no-ontology score showed identical ranking behaviour in this artifact, while the generative-only score was non-discriminative due to constant zero-valued scores.

## Paper-ready interpretation

Day 45 converts the ablation score artifacts into a held-out/test-style evaluation package. The outputs include threshold sensitivity tables, ROC/PR curve points, false-positive previews, false-negative previews, and per-variant metric summaries. These artifacts support the evaluation section of the paper by showing not only discrimination performance, but also the operating threshold behavior and error profile.

## Artifacts

- `day45_variant_summary.csv`
- `detector_only/day45_test_set_metrics.json`
- `no_ontology/day45_test_set_metrics.json`
- `generative_only/day45_test_set_metrics.json`
- each variant folder includes threshold sensitivity, ROC/PR points, FP/FN previews, and README.
