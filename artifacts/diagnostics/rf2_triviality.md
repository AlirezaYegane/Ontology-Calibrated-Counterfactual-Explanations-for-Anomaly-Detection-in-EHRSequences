# RF2 — Benchmark Triviality Diagnostic

**Input:** `data\processed\mimiciv_val_synth_anomaly.pkl`  
**Records:** 71602 (normal=55061, anomaly=16541)  
**Label column:** `is_synthetic_anomaly` | **Type column:** `anomaly_type` | **Sequence column:** `codes`

**Detector reference ROC-AUC:** 0.8002 (`artifacts\day45\detector_only\day45_test_set_metrics.json`, trained on the same synthetic scheme)

## Overall trivial-baseline results

| Signal | ROC-AUC | Avg Precision | Discriminative power | Direction |
|---|---:|---:|---:|---|
| contains_pregnancy_or_sex_specific_token | 0.7023 | 0.4473 | 0.7023 | higher_is_anomalous |
| rare_token_count | 0.4595 | 0.2006 | 0.5405 | lower_is_anomalous |
| rare_token_presence | 0.4595 | 0.2006 | 0.5405 | lower_is_anomalous |
| token_namespace_entropy | 0.4946 | 0.2286 | 0.5054 | lower_is_anomalous |
| procedure_token_count | 0.4981 | 0.2293 | 0.5019 | lower_is_anomalous |
| diagnosis_token_count | 0.4982 | 0.2313 | 0.5018 | lower_is_anomalous |
| sequence_length | 0.4988 | 0.2299 | 0.5012 | lower_is_anomalous |
| medication_token_count | 0.4991 | 0.2301 | 0.5009 | lower_is_anomalous |
| unknown_or_unmapped_token_count | 0.5000 | 0.2310 | 0.5000 | higher_is_anomalous |

## Per-anomaly-type best trivial signal

| Anomaly type | Best trivial signal | Discriminative power |
|---|---|---:|
| demographic_conflict | contains_pregnancy_or_sex_specific_token | 0.9372 |
| medication_mismatch | sequence_length | 0.7187 |
| missing_diagnosis | diagnosis_token_count | 0.5591 |

## Verdict

- **Are current labels likely artifact-driven?** YES (severity: high).
- **Strongest trivial baseline:** `contains_pregnancy_or_sex_specific_token` (discriminative power 0.7023).
- **Most suspicious anomaly type:** `demographic_conflict` — recoverable by `contains_pregnancy_or_sex_specific_token` at discriminative power 0.9372.
- **Do trivial signals rival the detector (~0.8002188893877784)?** No.
- **Is benchmark redesign mandatory?** YES — proceed to Phase 1b anomaly v2 before any further modeling.

## What this means for A* readiness

The current synthetic anomalies are recoverable by label-free trivial signals at a level comparable to the trained detector. Any detection or ontology-calibration result on this benchmark is therefore **not A*-defensible**: a reviewer can attribute the performance to injection artifacts (e.g., changes in sequence length or token-type counts) rather than clinical anomaly reasoning. Benchmark redesign (Phase 1b: anomaly v2 + leave-type-out protocol) is a prerequisite for any headline claim.
