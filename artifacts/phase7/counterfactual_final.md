# Phase 7 -- Counterfactual Final (benchmark-v2 test)

- attempted: **1376** (ontology-flagged 939)
- repair success among flagged: **0.8999** (overall 0.6141)
- mean delta S_ont: **0.643787** | mean edits 1.243787 (median 1) | edit ops {'remove': 936, 'add': 119}

## Success by rule type
| rule_kind | n | success | rate |
|---|---:|---:|---:|
| demographic_mismatch | 274 | 180 | 0.6569 |
| missing_required_code | 283 | 283 | 1.0 |
| mutual_exclusion | 437 | 437 | 1.0 |
| none | 437 | 0 | 0.0 |

## Edit-strategy ablation (on a capped sample of 300 ontology-flagged anomalies)
| strategy | success_rate | mean_edits |
|---|---:|---:|
| remove_only | 0.8933 | 1.328 |
| replace_only | 0.47 | 1.284 |
| add_context_allowed | 0.9133 | 1.212 |
| full_policy | 0.9133 | 1.212 |

> generator read model-visible rows only; no hidden/audit metadata.