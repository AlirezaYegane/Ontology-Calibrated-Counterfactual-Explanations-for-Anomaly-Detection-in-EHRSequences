# Phase 7 — Final Evaluation, Ablations, Statistics, Tables

## Status: `phase7_partial_external_validation_blocked`
Everything is complete on **benchmark-v2** (the final benchmark) — main results,
ablations, paired statistical tests, counterfactual results, tables/figures, and
final claim decisions. The **only** blocked item is **external validation**: eICU
exists but its APACHE/body-system tokens do not map to the ICD/SNOMED/RxNorm
ontology (0/500 records fire), so it is honestly blocked (schema mismatch) and
documented as future work.

## Final main method
**Ontology-centered: `S_main = S_ont`** (real Phase 3b rule packs). The detector is a
diagnostic/negative result; **Sgen is excluded** (`w_gen = 0`).
Score equation: `S_cal = (w_det·S_det + w_ont·S_ont′)/(w_det + w_ont)`, `w_gen = 0`;
recommended main = `S_ont` (detector is non-additive).

## Main results (benchmark-v2 test, n=6,307)
| variant | ROC-AUC | 95% CI | AP | F1 |
|---|---:|---|---:|---:|
| **ontology_only_real** | **0.7881** | [0.774, 0.802] | 0.542 | 0.635 |
| legacy_baseline (pure legacy rules) | 0.7358 | [0.720, 0.751] | 0.543 | 0.593 |
| detector_only_full | 0.4525 | [0.436, 0.470] | 0.190 | 0.361 |
| combined_real_without_sgen | 0.7036 | [0.687, 0.720] | 0.404 | 0.460 |
| Sgen (diagnostic, EXCLUDED) | 0.4868 | — | — | — |

## Statistical tests (paired bootstrap ROC-AUC, p≈0 unless noted)
- **ontology_only_real − legacy = +0.052** [0.033, 0.072] → real **significantly beats** legacy.
- **combined − ontology_only = −0.085** [−0.096, −0.073] → adding the detector **significantly hurts**.
- ontology_only − detector = +0.336; combined − legacy = −0.032 (p=0.004).

## Ablations
**Ontology rules:** full 0.7881 > forbidden-only 0.625 > demographic-only 0.599 >
medication-only 0.571 > disabled (chance). No single family reaches the full score —
the three synergize. Normal FP: full 0.13.
**Score components:** S_ont 0.788, S_det 0.453, S_ont+S_det 0.704; Sgen row EXCLUDED.

## Counterfactual (benchmark-v2 test, 1,376 anomalies; 939 ontology-flagged)
**89.99% valid repair among flagged** (61.4% overall — the rest are ontology
detection gaps), mean ΔS_ont 0.644, **median 1 edit**, 936 remove / 119 add.
Per rule: medication 100%, forbidden 100%, demographic 65.7%.
Edit-strategy ablation: remove_only 89.3%, replace_only 47.0%, add_context 91.3%,
full 91.3%.

## External validation
`external_validation_blocked_schema_mismatch` — eICU uses APACHE codes; 0 tokens map
to the ontology. Needs an APACHE→ICD/SNOMED crosswalk + eICU anomaly injection
(future work). Recommended scope: **MIMIC-IV-only paper**.

## Final claims (see `final_claims_decision.json`)
- supported_now: non-circular benchmark, real ontology integration, **real ontology >
  legacy**, leakage-free counterfactual, **effective counterfactual repair**, reproducible.
- unsupported: detector improves detection; combined > ontology-only.
- removed_from_core: Sgen.
- future_work: clinical validation, external generalization.

## What Phase 8 should write
An **ontology-centered, MIMIC-IV-only** paper: real ontology rules are the main
anomaly-ranking signal (significantly > legacy); leakage-free ontology counterfactual
repair (~90% for flagged); the detector is an honest negative result; Sgen removed.

## What Phase 8 must NOT overclaim
Detector value, Sgen, external validation, clinical validity, or SOTA. Report the
detector as a negative result; keep external/clinical validation as future work.
