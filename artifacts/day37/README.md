# Day 37 — Counterfactual Evaluation

## Status
Complete.

## Purpose
Day 37 evaluates whether generated counterfactuals reduce ontology/repair scores, remain sparse, and resolve ontology-related inconsistencies.

## Input
- Source file: `artifacts/day36/counterfactuals.csv`

## Main Metrics

- Rows evaluated: **16442**
- Percentage with reduced score: **0.9998175404452013**
- Mean score reduction: **1.7183128573166282**
- Median score reduction: **1.5**
- Mean edit count: **1.0084539593723392**
- One-or-two-edit rate: **0.9998175404452013**
- Ontology fully resolved rate: **0.9956817905364311**

## Files

- `counterfactual_eval_records.csv`
- `counterfactual_eval_by_type.csv`
- `day37_counterfactual_eval_summary.json`

## Scientific Note
These metrics support the paper's explanation-efficacy analysis. They should be reported honestly: strong score reduction and sparse edits support the method; weak or uneven results should be treated as failure-mode evidence rather than hidden.

## Methodological clarification
The Day 36 output stores ontology repair quality through before/after violation-score columns rather than a full post-counterfactual calibrated anomaly score.

Therefore, Day 37 evaluates counterfactual repair efficacy using:

- before: `original_violation_score`
- after: `counterfactual_violation_score`

This is a conservative evaluation choice. It directly measures whether the proposed edit reduces ontology inconsistency while remaining sparse. Full calibrated-score improvement can be added later when post-counterfactual detector and diffusion scores are recomputed.
