# Day 45 — Comprehensive Test Set Evaluation

## Status

Complete.

## Goal

Evaluate the final calibrated anomaly score on a held-out/test score artifact, inspect threshold sensitivity, and select a conservative operating threshold with controlled false-positive rate.

## Input

- Input scores: `D:\Article\artifacts\day45\day45_input_no_ontology.csv`
- Score column: `score`
- Label source: `y_true`
- Type column: `not available`

## Global metrics

| Metric | Value |
|---|---:|
| Records | 71503 |
| Positives | 16442 |
| Negatives | 55061 |
| ROC-AUC | 0.8002188893877784 |
| Average Precision | 0.7332429325476619 |

## Selected operating threshold

- Selection strategy: `conservative_threshold_with_fpr_and_precision_constraints`
- Threshold: `0.436500`
- Precision: `0.801756`
- Recall: `0.566476`
- F1: `0.663887`
- False positive rate: `0.041826`
- False negative rate: `0.433524`
- Predicted positive rate: `0.162469`

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
