# Contribution Matrix (Phase 0)

> **Phase 8 finalization note.** This is the *original Phase 0* planning matrix, kept for
> history. The **final, evidence-based** contribution statement is
> [`final_contribution_statement.md`](final_contribution_statement.md) and the final claim
> ledger is [`final_claims_matrix.md`](final_claims_matrix.md). Outcome vs. this plan: K1
> (benchmark) and K2 (ontology triage) **landed and are supported**; K3 (leakage-free
> ontology counterfactual) **landed and is supported**; K4 (detector backbone) became a
> **negative result** (below chance, non-additive); K5 (diffusion/`Sgen`) was **removed
> from the core**; K6 (clinician study) and K7 (eICU external validation) remain **future
> work** — eICU is blocked by an APACHE↔ICD/SNOMED schema mismatch.

**Purpose.** Map candidate contributions to a realistic paper plan, with current
readiness and the single biggest blocker for each. Readiness is a blunt estimate, not
a promise.

## Candidate contributions

| ID | Candidate contribution | Evidence today | Readiness | Biggest blocker | Paper assignment |
|---|---|---|---|---|---|
| K1 | A non-circular synthetic anomaly **benchmark** for EHR sequences (with trivial-baseline controls and leave-type-out protocol) | Injection code + data exist but circular | 30% | Phase 1 anomaly v2 + protocol | **Main paper** (contribution 1) |
| K2 | **Ontology-grounded interpretable anomaly triage** (real SNOMED/RxNorm rules as a high-precision explanatory layer) | Hardcoded ICD-prefix rules; real engine disconnected | 25% | Phase 2 real ontology integration | **Main paper** (contribution 2) |
| K3 | **Leakage-free, ontology-constrained counterfactual repair**, validated by an independent scorer | Circular metadata-driven edits | 15% | Phase 4 rebuild without answer-key | **Main paper** (contribution 3, the centerpiece) |
| K4 | **Non-circular detector + calibrated scoring** with confidence intervals and significance tests | Supervised, circular; Δ≈0.001 ablation, no CIs | 35% | Phase 3 unsupervised detector + stats | **Main paper** (method backbone) |
| K5 | **Diffusion-based generative surprise / counterfactual** | Sgen AUC 0.475; mode-collapsed sampler | 5% | Phase 5 decision gate | **Second paper only if gate passes**, else future work |
| K6 | **Clinician-validated explanation usefulness** | None | 0% | Phase 7 clinician study | Optional strengthener for main paper |
| K7 | **Cross-dataset (eICU) external validation** | eICU sequences exist, unused | 10% | Phase 2 mapping + Phase 7 | Strengthener for main paper |

## Recommended paper plan

- **Paper 1 (main, A* target — CHIL / ML4H proceedings / JBI/JAMIA):**
  K1 + K2 + K3 + K4. Story: *a non-circular EHR anomaly benchmark plus an
  ontology-grounded, leakage-free counterfactual explanation method.*
  Novelty anchor = **K3** (leakage-free ontology counterfactual repair).
- **Paper 2 (conditional — generative):** K5, **only if** the Phase 5 gate passes.
  Otherwise K5 is explicitly future work in Paper 1.
- **Optional strengtheners:** K6 (clinician κ study), K7 (eICU).

## Decision on diffusion (recorded in Phase 0)

Diffusion is **provisionally demoted to a diagnostic component** pending the Phase 5
decision gate. It is **not** part of the main-paper claim set. It is promoted back to a
contribution only if multi-sample `Sgen` reaches ROC-AUC ≥ 0.65 on the Phase 1b
benchmark **and** generated samples match the real length/vocabulary distribution
(JS ≤ 0.10, vocab coverage ≥ 60%). Otherwise it is cut to future work without sentiment.

## Minimum result needed before writing Paper 1

1. Trivial baselines underperform the detector by a clear margin on the v2 benchmark.
2. Real ontology loaded with a coverage report meeting thresholds.
3. Unsupervised detector + significance-tested ablation.
4. Counterfactual passes the no-leakage test and beats a random-edit control under an
   independent scorer.

Until these hold, Paper 1 is not writable at A* level.
