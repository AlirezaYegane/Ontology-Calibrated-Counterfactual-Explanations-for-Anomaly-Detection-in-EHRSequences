# Final Contribution Statement (Phase 8)

This project makes five defensible contributions, plus two transparent negative results.
Each contribution is scoped to the evidence actually in the repository (MIMIC-IV
benchmark-v2). Nothing here depends on the detector, on `Sgen`, on external validation, or
on clinician validation.

## Contributions

1. **A leakage-controlled MIMIC-IV benchmark (benchmark-v2) for relational EHR anomalies.**
   Each anomaly is a violation of a *relationship* between fields while every individual
   model-visible token stays common, so a single token-presence feature cannot recover the
   label (strongest trivial signal 0.6127 < 0.80). Subject-level splits with zero overlap;
   strict model-visible / audit / hidden-eval separation.

2. **A real UMLS / SNOMED CT / RxNorm ontology scoring engine.** ICD-9/10 → SNOMED and
   drug → RxNorm crosswalks from authoritative UMLS mappings, with MIMIC-IV coverage of
   0.80 (diagnosis) and 0.78 (medication), driving three auditable rule families
   (demographic incompatibility, medication-indication context, forbidden co-occurrence).

3. **Evidence that real ontology scoring outperforms legacy ICD-prefix rules.**
   ontology_only_real 0.7881 vs legacy 0.7358; paired-bootstrap +0.052 ROC-AUC, p ≈ 0.

4. **A leakage-free, ontology-guided counterfactual repair method.** Minimal
   (median 1 edit), ontology-valid edits, validated by an independent scorer, that never
   read the corruption answer key; 89.99% valid repair among ontology-flagged anomalies.

5. **A transparent negative result for the detector and generative components.** A
   full-scale unsupervised sequence detector is below chance (0.4525) and *non-additive*
   (combining it lowers ROC-AUC by 0.085, p ≈ 0), and diffusion-based generative surprise
   (`Sgen`) is near-random and mode-collapsed and is removed from the core (`w_gen = 0`).
   These results clarify *why* the anomalies are relational rather than next-token-surprising.

## What is deliberately **not** claimed

- external validation / cross-dataset generalization (eICU schema mismatch — blocked);
- clinical deployment readiness;
- clinician-validated repairs;
- state-of-the-art deep anomaly detection;
- any improvement from diffusion / `Sgen`;
- any improvement from adding the detector to the ontology score.

## Why the negative results strengthen the paper

The project set out expecting the detector and the generative term to carry the signal.
Instead the evidence showed the discriminative signal is *relational* and is captured by
the ontology rules, not by next-token surprise. Reporting this plainly — with confidence
intervals and paired significance tests — is what makes the ontology-centered claim
credible rather than convenient.

See also: [`final_claims_matrix.md`](final_claims_matrix.md).
