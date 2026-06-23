# RF2 — Benchmark Triviality Diagnostic

**Input:** `data\processed\benchmark_v2\val.pkl`  
**Records:** 3123 (normal=2513, anomaly=610)  
**Label column:** `label` | **Type column:** `anomaly_type` | **Sequence column:** `model_visible_sequence`

**Detector reference ROC-AUC:** 0.8002 (`artifacts\day45\detector_only\day45_test_set_metrics.json`, trained on the same synthetic scheme)

## Overall trivial-baseline results

| Signal | ROC-AUC | Avg Precision | Discriminative power | Direction |
|---|---:|---:|---:|---|
| contains_pregnancy_or_sex_specific_token | 0.6127 | 0.3147 | 0.6127 | higher_is_anomalous |
| sequence_length | 0.6091 | 0.2484 | 0.6091 | higher_is_anomalous |
| diagnosis_token_count | 0.6081 | 0.2634 | 0.6081 | higher_is_anomalous |
| medication_token_count | 0.5990 | 0.2434 | 0.5990 | higher_is_anomalous |
| token_namespace_entropy | 0.5893 | 0.2392 | 0.5893 | higher_is_anomalous |
| procedure_token_count | 0.5538 | 0.2279 | 0.5538 | higher_is_anomalous |
| rare_token_count | 0.5357 | 0.2153 | 0.5357 | higher_is_anomalous |
| rare_token_presence | 0.5317 | 0.2132 | 0.5317 | higher_is_anomalous |
| unknown_or_unmapped_token_count | 0.5000 | 0.2010 | 0.5000 | higher_is_anomalous |

## Per-anomaly-type best trivial signal

| Anomaly type | Best trivial signal | Discriminative power |
|---|---|---:|
| demographic_incompatibility | contains_pregnancy_or_sex_specific_token | 0.9462 |
| forbidden_cooccurrence | diagnosis_token_count | 0.7522 |
| medication_indication_mismatch | sequence_length | 0.6653 |

## Verdict

- **Are current labels likely artifact-driven?** No (not strongly) (severity: moderate).
- **Strongest trivial baseline:** `contains_pregnancy_or_sex_specific_token` (discriminative power 0.6127).
- **Most suspicious anomaly type:** `demographic_incompatibility` — recoverable by `contains_pregnancy_or_sex_specific_token` at discriminative power 0.9462.
- **Do trivial signals rival the detector (~0.8002188893877784)?** No.
- **Is benchmark redesign mandatory?** Not strictly, but recommended.

## What this means for A* readiness

Trivial baselines do not fully explain the labels, which is encouraging, but the supervised-on-synthetic training still creates circularity. A non-circular protocol (Phase 1b/3) remains required before A* claims.
