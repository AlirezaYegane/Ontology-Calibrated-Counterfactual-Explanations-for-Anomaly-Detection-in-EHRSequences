# Phase 4 -- Counterfactual Repair Evaluation

**Split:** `test` (v2 (non-circular)) | **detector:** disabled

- repair attempted: **30** (ontology-flagged: 24, unflagged/detection-gap: 6)
- repair success (valid): **20** (rate over all **0.6667**; **over flagged 0.8333**)
- mean ΔS_ont: **0.55** | mean ΔS_cal: 0.1252 | mean ΔS_det: None (diagnostic-only)
- mean edits: **1.25** | median edits: 1.0 | mean ontology distance: 0.0
- edit ops: {'remove': 23, 'add': 3}
- failure reasons: {'no_ontology_violation_to_repair': 6, 'no_score_reducing_edit_found': 3, 'empty_record': 1}

## Success by anomaly type
| anomaly_type | n | success | rate |
|---|---:|---:|---:|
| demographic_incompatibility | 5 | 1 | 0.2 |
| forbidden_cooccurrence | 10 | 10 | 1.0 |
| medication_indication_mismatch | 15 | 9 | 0.6 |

## Success by rule type (violation kind before repair)
| rule_kind | n | success | rate |
|---|---:|---:|---:|
| demographic_mismatch | 5 | 1 | 0.2 |
| missing_required_code | 8 | 8 | 1.0 |
| mutual_exclusion | 11 | 11 | 1.0 |
| none | 6 | 0 | 0.0 |

> S_det is smoke-scale and diagnostic-only; it does NOT drive repair and ΔS_det must not be read as repair success.
>
> Generator received model-visible rows only; label/anomaly_type used for selection+bucketing only.