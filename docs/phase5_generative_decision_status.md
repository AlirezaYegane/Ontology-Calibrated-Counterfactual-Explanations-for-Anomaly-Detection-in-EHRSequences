# Phase 5 — Generative / Sgen Decision Status

**Status: `phase5_remove_sgen_from_core`.** The diffusion generative-surprise `Sgen`
is removed from the core method. It scores below chance on the non-circular
benchmark-v2 and statistically-credibly *harms* the combined score. `w_gen` stays 0.0.

## Decision gate
`src/evaluation/generative_gate.py` (strict): `keep_main` requires ALL of — Sgen
ROC-AUC ≥ 0.55, CI clears chance, adds signal beyond ontology+detector, improves the
combined score, no leakage, valid protocol, no mode collapse. `Sgen` meets **none** of
the keep-main conditions and is mode-collapsed → **`remove_from_core`**.

## Evidence (benchmark-v2, diagnostic-only)
- Sgen ROC-AUC **0.4868** (CI [0.4633, 0.5109]); anomalies score *lower* than normals.
- corr(Sgen, S_ont) = **−0.07** (no shared signal with the ontology).
- Combined **without** Sgen 0.6545 → **with** Sgen 0.637; paired ΔAUC **−0.0175**,
  CI [−0.027, −0.009], p ≈ 0 (Sgen significantly degrades the combined score).
- Generation mode-collapsed (127/4587 unique tokens; length 254 vs 47).
- Checkpoint trained on **old circular-era data**; loads only via a compatibility shim
  (architecture drift) → diagnostic-only, not paper evidence.

## 1. Is Sgen kept in the main method?
**No.** Removed from the core; diagnostic/appendix-only.

## 2. Is the diffusion/generative model diagnostic-only?
**Yes.** Old-data, mode-collapsed, not loadable without a shim. Diagnostic-only.

## 3. Is any generative result paper-claimable?
**No.** `final_paper_evidence_claimable = false`.

## 4. Score equation
`S_cal = (w_det·S_det + w_ont·S_ont′) / (w_det + w_ont)` with **`w_gen = 0`**. `Sgen`
enters only if explicitly supplied with `w_gen > 0` — never by default.

## 5. Can Phase 6 proceed?
**Yes — without Sgen in the core.** Full-scale training + experiment infrastructure may
proceed. Generative retraining (benchmark-v2 clean-normal-only + anti-collapse) is a
separate optional future phase; revisit Sgen only if it later clears the gate. No H200
unless Phase 6 explicitly requires it.

## Claim impact
- **C7** Sgen separates anomalies → confirmed removed from core (below chance, harms
  combined). Negative result / future work.
- **C8** realistic diffusion generation → cut (mode collapse).
