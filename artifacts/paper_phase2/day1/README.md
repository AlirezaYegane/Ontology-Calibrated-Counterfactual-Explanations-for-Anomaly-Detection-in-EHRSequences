# Paper Phase 2 — Operational Day 1

## Status
Complete.

## Purpose
Implement and smoke-evaluate the B0 token-rarity and B1 statistical-relational baselines on the canonical benchmark-v2 validation split.

This is an untuned validation diagnostic, not final benchmark evidence.

## Canonical data
- clean train: 20,570 normal records
- validation: 3,123 records
- validation normals: 2,513
- validation anomalies: 610
- test split accessed: no

## Initial validation results

| Score | PR-AUC | ROC-AUC |
|---|---:|---:|
| B0 max token surprisal | 0.2491 | 0.6021 |
| B1 top-k relation | 0.2450 | 0.6005 |
| B1 worst relation | 0.2435 | 0.5991 |
| B0 rare-code fraction | 0.2218 | 0.5714 |
| B1 q90 relation | 0.1954 | 0.4998 |
| B1 confidence anomaly | 0.1788 | 0.4587 |
| B1 mean relation | 0.1682 | 0.4363 |
| B1 lift anomaly | 0.1648 | 0.4362 |
| B0 mean negative-log-frequency | 0.1621 | 0.4271 |
| B1 NPMI anomaly | 0.1573 | 0.4095 |

## Preliminary interpretation
The initial B0 and B1 configurations contain modest anomaly-discrimination signal.

The strongest untuned score is B0 maximum token surprisal. B1 top-k and worst-relation aggregation are very close behind.

The large difference between B1 worst/top-k aggregation and B1 mean aggregation suggests that relational anomaly signal may be concentrated in a small number of unusually incompatible relations within otherwise ordinary records.

No superiority claim is made from this run.

## Scientific safeguards
- statistics estimated from clean training records only
- validation used for diagnostics only
- test split not accessed
- ontology information not used by B0/B1
- hidden evaluation metadata not used as input
- no per-record scores persisted

## Next step
Day 2 will perform validation-only hyperparameter and aggregation selection, including the required sequence-length and candidate-pair-count bias audit. The chosen configuration will then be frozen before any benchmark-v2 test evaluation.
