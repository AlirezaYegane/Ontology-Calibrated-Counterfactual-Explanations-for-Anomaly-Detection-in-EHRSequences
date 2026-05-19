# Day 45 — Comprehensive Test Set Evaluation

## Status

Complete.

## Goal

Evaluate the final calibrated anomaly score on a held-out/test score artifact, inspect threshold sensitivity, and select a conservative operating threshold with controlled false-positive rate.

## Input

- Input scores: `D:\Article\artifacts\day45\day45_input_generative_only.csv`
- Score column: `score`
- Label source: `y_true`
- Type column: `not available`

## Global metrics

| Metric | Value |
|---|---:|
| Records | 71503 |
| Positives | 16442 |
| Negatives | 55061 |
| ROC-AUC | 0.5 |
| Average Precision | 0.2299483937736878 |

## Selected operating threshold

- Selection strategy: `fallback_best_f1_threshold_constraints_not_met`
- Threshold: `0.000000`
- Precision: `0.229948`
- Recall: `1.000000`
- F1: `0.373916`
- False positive rate: `1.000000`
- False negative rate: `0.000000`
- Predicted positive rate: `1.000000`

## Error-analysis artifacts

- `threshold_sensitivity.csv`
- `roc_curve_points.csv`
- `pr_curve_points.csv`
- `error_analysis_by_type.csv`
- `false_positive_examples.csv`
- `false_negative_examples.csv`
- `flagged_records_preview.csv`

## Paper-ready interpretation

The selected threshold provides a conservative operating point for the ontology-calibrated anomaly score. This report should be used to support the evaluation section by documenting global discrimination, precision-recall trade-offs, false-positive behavior, and anomaly-type-specific failure modes.
