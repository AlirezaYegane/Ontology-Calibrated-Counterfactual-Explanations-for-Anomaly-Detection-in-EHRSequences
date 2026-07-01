# Phase 7 — Final Evaluation Status

**Status: `phase7_partial_external_validation_blocked`.** Final aggregate evidence on
benchmark-v2 is complete; external validation is honestly blocked (eICU schema
mismatch). Not paper writing yet — this is the evidence package for Phase 8.

## 1. Final evaluation dataset
benchmark-v2 (non-circular, MIMIC-IV): train 20,570 (normal-only) / val 3,123 / test
6,307; subject-overlap 0. The old circular benchmark is **not** used as final evidence.

## 2. Final main method
Ontology-centered anomaly ranking: `S_main = S_ont` (real Phase 3b rule packs:
demographic sex-restriction, medication required-context, diabetes-type exclusion).
The unsupervised detector is a **diagnostic/negative result**; Sgen is excluded.

## 3. Final score equation
`S_cal = (w_det·S_det + w_ont·S_ont′)/(w_det + w_ont)` with **`w_gen = 0`**.
Because the detector is non-additive, the recommended paper main score is `S_ont`.

## 4. Why Sgen is excluded
Phase 5 gate = `remove_from_core`: on benchmark-v2 Sgen ROC-AUC 0.4868 (below chance)
and it significantly harms the combined score; the generator is mode-collapsed.

## 5. Detector result & interpretation
Full-scale unsupervised next-token detector (Phase 6, 25 epochs, 20,570 normals):
test ROC-AUC **0.4525** (below chance). The v2 anomalies are **relational** (gender
flip, indication removal, mutual exclusion) and carry little next-token surprise, so a
language-model detector cannot separate them. Honest negative result.

## 6. Ontology result & interpretation
`ontology_only_real` ROC-AUC **0.7881** (CI [0.774, 0.802]) — the strongest variant,
carrying the discriminative signal. Ablation: the three rule families synergize (no
single family exceeds ~0.63 alone).

## 7. Legacy comparison
Real ontology (0.7881) **significantly beats** the legacy ICD-prefix rules (0.7358);
paired bootstrap +0.052, CI [0.033, 0.072], p≈0.

## 8. Counterfactual result
Leakage-free ontology counterfactual repair: **89.99% valid among ontology-flagged**
anomalies (939/1,376 flagged), median 1 edit, mean ΔS_ont 0.644. Effective for
medication/forbidden (100%), weaker for demographic (65.7%, edit-budget-limited).

## 9. External validation status
`external_validation_blocked_schema_mismatch`: eICU uses APACHE/body-system tokens; 0
of 500 sampled records map to the ICD/SNOMED/RxNorm ontology. Requires an
APACHE→ICD/SNOMED crosswalk + eICU anomaly injection (future work).

## 10. Final supported claims
Non-circular benchmark; real ontology integration; **real ontology > legacy**;
leakage-free counterfactual; **effective counterfactual repair for flagged anomalies**;
reproducible infrastructure.

## 11. Final unsupported / removed claims
Detector improves detection — **unsupported**. Combined > ontology-only —
**unsupported** (significantly worse). Sgen — **removed_from_core**. Clinical &
external validation — **future_work**.

## 12. What Phase 8 should write
An **ontology-centered, MIMIC-IV-only** paper: real ontology rules are the main
anomaly-ranking signal (significantly above legacy), plus leakage-free ontology
counterfactual repair; the detector is an honest negative result; Sgen removed.

## 13. What Phase 8 must not overclaim
No detector value, no Sgen, no external validation, no clinical validity, no SOTA.
Keep the negative results visible.
