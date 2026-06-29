# Phase 6 -- Detector Evaluation

**run:** `phase6_detector_smoke_seed42` | **evidence:** smoke | device cpu

- test ROC-AUC: **0.426** CI [0.4087, 0.4414]
- test AP: 0.181 CI [0.1708, 0.1926]
- test F1 (val-threshold 3.615161): 0.3582 (P 0.2182 / R 1.0)

## By anomaly family
| family | n | ROC-AUC | AP |
|---|---:|---:|---:|
| demographic_incompatibility | 5203 | 0.576 | 0.0617 |
| forbidden_cooccurrence | 5160 | 0.5162 | 0.048 |
| medication_indication_mismatch | 5806 | 0.3557 | 0.1082 |

> Smoke-scale results are diagnostic. The unsupervised next-token detector has limited signal on relational benchmark-v2 anomalies; ontology_only_real remains the strongest variant (see combined_eval).