# Results

All numbers are on benchmark-v2 test (n = 6,307; anomaly rate 21.8%), threshold selected on
validation. Sources are the aggregate JSON/CSV under `artifacts/phase7/`.

## Main results

| Variant | ROC-AUC | 95% CI | AP | F1 |
|---|---:|---|---:|---:|
| **ontology_only_real** (main) | **0.7881** | [0.7743, 0.8015] | 0.5422 | 0.6349 |
| legacy_baseline | 0.7358 | [0.7197, 0.7511] | 0.5429 | 0.5928 |
| detector_only_full | 0.4525 | [0.4357, 0.4702] | 0.1904 | 0.3608 |
| combined_real_without_sgen | 0.7036 | [0.6866, 0.7201] | 0.4039 | 0.4596 |
| Sgen (diagnostic, excluded) | 0.4868 | — | — | — |

The real ontology scorer is the strongest variant. The detector alone is *below chance*.
The combined score sits between the two — worse than ontology-only.

## Statistical tests (paired bootstrap, ROC-AUC)

| Comparison | Δ ROC-AUC | 95% CI | p | Significant |
|---|---:|---|---:|---|
| ontology_only_real − legacy_baseline | +0.0524 | [0.0325, 0.0718] | ≈ 0 | yes |
| ontology_only_real − detector_only_full | +0.3356 | [0.3141, 0.3581] | ≈ 0 | yes |
| combined − ontology_only_real | −0.0845 | [−0.0963, −0.0732] | ≈ 0 | yes |
| combined − detector_only_full | +0.2511 | [0.2364, 0.2665] | ≈ 0 | yes |
| combined − legacy_baseline | −0.0322 | [−0.0567, −0.0103] | 0.004 | yes |

Two conclusions carry the paper: **real ontology significantly beats legacy** (+0.052), and
**adding the detector significantly hurts** (−0.085). Both intervals exclude zero.

## Ablation summary

Ontology rule-family ablation (ROC-AUC):

| Variant | ROC-AUC | Normal FP rate |
|---|---:|---:|
| real_ontology_rules_full | 0.7881 | 0.130 |
| legacy_icd_prefix_rules | 0.7358 | — |
| demographic_rules_only | 0.5988 | 0.0016 |
| medication_rules_only | 0.5711 | 0.0635 |
| forbidden_cooccurrence_rules_only | 0.6252 | 0.0671 |
| ontology_disabled | degenerate (chance) | — |

No single rule family reaches the full 0.7881; the three families **synergize**. Score-component
ablation confirms `S_ont` (0.7881) > `S_ont + S_det` (0.7036) > `S_det` (0.4525), with
`S_ont + S_det + S_gen` excluded (`w_gen = 0`).

## Counterfactual results

| Metric | Value |
|---|---:|
| Test anomalies attempted | 1,376 |
| Ontology-flagged anomalies | 939 |
| Repair success among flagged | 0.8999 |
| Repair success overall | 0.6141 |
| Mean ΔS_ont | 0.644 |
| Median edits | 1 |

By rule type: missing-required-context 100%, mutual-exclusion 100%, demographic mismatch
65.7% (edit-budget-limited on records with many obstetric codes). Edit operations are
dominated by removals (936 remove, 119 add). The edit-strategy ablation shows add-context /
full policy best (0.9133 on the capped sample), remove-only strong (0.8933), replace-only
weak (0.47) — neighbors of a sex-restricted code stay sex-restricted. The 437
non-flagged anomalies are detection (coverage) gaps, not repair failures.

## External validation status

`external_validation_blocked_schema_mismatch`. Of 500 sampled eICU records, 0 tokens map to
the ICD/SNOMED/RxNorm ontology and 0 rules fire, because eICU uses APACHE/body-system tokens
(e.g. `EICU_APACHE2_DX:*`, `EICU_BODYSYS:*`). External validation would require an
APACHE→ICD/SNOMED crosswalk plus applying the benchmark-v2 injectors to eICU — both out of
scope and documented as future work.
