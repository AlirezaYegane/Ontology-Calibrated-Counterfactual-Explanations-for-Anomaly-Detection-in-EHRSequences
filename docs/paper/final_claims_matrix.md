# Final Claims Matrix (Phase 8)

**Authoritative source:** [`artifacts/phase7/final_claims_decision.json`](../../artifacts/phase7/final_claims_decision.json).
This table is the single, final ledger of what the paper may and may not claim. It matches
the Phase 7 evidence exactly. Unsupported claims are **not** softened into supported ones.

All evidence is on **benchmark-v2 (non-circular, MIMIC-IV)**. The old circular benchmark is
**not** used as final evidence.

| Claim | Decision | Evidence |
|---|---|---|
| benchmark-v2 is leakage-controlled | **supported** | Strongest label-free trivial signal 0.6127 < 0.80 gate; subject-level splits (overlap 0); model-visible / audit / hidden-eval field separation (`artifacts/phase1b/`, `artifacts/diagnostics/rf2_triviality_v2.json`) |
| real ontology integration works | **supported** | Real UMLS 2026AA / SNOMED CT / RxNorm parsed; MIMIC-IV coverage diagnosis 0.80, medication 0.78; canonical engine wired into scoring (`artifacts/phase2b_fix/coverage_report.json`) |
| real ontology beats legacy | **supported** | ontology_only_real 0.7881 vs legacy 0.7358; paired bootstrap +0.0524, CI [0.033, 0.072], p ≈ 0 (`artifacts/phase7/final_stat_tests.json`) |
| detector improves anomaly detection | **unsupported** | Full-scale unsupervised detector (25 epochs, 20,570 normals) test ROC-AUC 0.4525 — below chance; anomalies are relational, not next-token-surprising (`artifacts/phase6/`, `artifacts/phase7/final_evaluation.json`) |
| combined detector+ontology beats ontology-only | **unsupported** | combined 0.7036 < ontology-only 0.7881; paired bootstrap −0.0845, CI [−0.096, −0.073], p ≈ 0 → adding the detector significantly hurts |
| Sgen / diffusion improves detection | **removed from core** | Phase 5 gate = remove_from_core: Sgen ROC-AUC 0.4868 (below chance), harms combined, mode-collapsed; w_gen = 0 (`artifacts/phase5/phase5_summary.json`) |
| counterfactual repair is leakage-free | **supported** | Generator reads only model-visible sequence + demographics + ontology + scorer; leakage tests pass; leaky legacy logic deleted (`tests/test_phase4_counterfactual_leakage.py`) |
| repair effective for ontology-flagged anomalies | **supported** | 89.99% valid repair among 939 flagged anomalies; mean ΔS_ont 0.644; median 1 edit (`artifacts/phase7/counterfactual_final.json`) |
| external validation | **future work / blocked** | eICU uses APACHE/body-system tokens; 0/500 records map to the ICD/SNOMED/RxNorm ontology → external_validation_blocked_schema_mismatch (`artifacts/phase7/external_validation_status.json`) |
| clinical validity | **future work** | No clinician validation; validity = ontology-violation resolution, not clinical ground truth |
| reproducibility | **supported** | Config system + resumable training + experiment index + deterministic seeds + aggregate evaluation scripts; licensed data / checkpoints git-ignored (`artifacts/phase6/`, `artifacts/phase7/`) |

## Buckets

- **supported (6):** leakage-controlled benchmark-v2; real ontology integration; real
  ontology > legacy; leakage-free counterfactual; effective repair for flagged anomalies;
  reproducibility.
- **unsupported (2):** detector improves detection; combined > ontology-only.
- **removed from core (1):** Sgen / diffusion.
- **future work (2):** external validation (blocked by schema mismatch); clinical validity.

## Honesty constraints (restated)

1. Do not upgrade an unsupported claim to supported.
2. Report the below-chance detector and the near-random Sgen as **negative results**, not
   as omissions.
3. Keep the scope explicit: MIMIC-IV benchmark-v2, synthetic relational anomalies,
   ontology-based (not clinician-verified) validity.

See also: [`final_contribution_statement.md`](final_contribution_statement.md),
[`claims_matrix.md`](claims_matrix.md) (full historical ledger).
