# Phase 6 -- Detector Evaluation

**run:** `phase6_detector_full_gpu` | **evidence:** full_gpu | device cuda

- test ROC-AUC: **0.4525** CI [0.4353, 0.4693]
- test AP: 0.1904 CI [0.1792, 0.2035]
- test F1 (val-threshold 2.935288): 0.3608 (P 0.2205 / R 0.9927)

## By anomaly family
| family | n | ROC-AUC | AP |
|---|---:|---:|---:|
| demographic_incompatibility | 5203 | 0.4064 | 0.0438 |
| forbidden_cooccurrence | 5160 | 0.5729 | 0.0549 |
| medication_indication_mismatch | 5806 | 0.4354 | 0.1233 |

> Smoke-scale results are diagnostic. The unsupervised next-token detector has limited signal on relational benchmark-v2 anomalies; ontology_only_real remains the strongest variant (see combined_eval).