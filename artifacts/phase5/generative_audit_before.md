# Phase 5 — Generative / Diffusion Audit (BEFORE)

**Verdict (blunt): the diffusion / `Sgen` component does not work and is not
paper-claimable.** It is mode-collapsed, scores at/below chance on the
non-circular benchmark-v2, was trained on old circular-era data, and its
checkpoint no longer loads cleanly into the current model code. It is already
excluded from the calibrated score (`w_gen = 0`) and should stay out of the core.

## 1. What generative/diffusion code exists?
- `src/models/diffusion.py` (continuous-embedding Transformer diffusion;
  `surprise_score` = midpoint non-pad denoising error), `diffusion_legacy_day33.py`.
- `src/training/{train_diffusion.py, diffusion_data_utils.py, diffusion_training_utils.py, build_diffusion_data.py}`.
- `src/evaluation/evaluate_day34_generative.py`.
- `scripts/{day34_sgen_timestep_sweep, smoke_day29_diffusion, summarize_day31/32/34_*}.py`.
- Checkpoints under `outputs/diffusion/day3x/` (day31–33) + `artifacts/day27,day31` diffusion tensors.

## 2. What data does it train on?
The latest checkpoint (`outputs/diffusion/day33_ontology_regularized_fixed/`) was
trained on `artifacts/day27/mimiciv_val_diffusion.pt` — the **old `mimiciv_val.pkl`**
(circular-era val data, 12k records, vocab 47010 from the Day-20 supervised model).
**Not** benchmark-v2; **not** an explicit clean-normal-only split.

## 3. What score does it output?
Per-record scalar `Sgen` = mean squared denoising error at the midpoint timestep
over non-pad positions (`DiffusionModel.surprise_score`). Stochastic (Gaussian
noise per call).

## 4. Is `Sgen` used anywhere?
Only as a **diagnostic** term in `compute_calibrated_score` (it is admitted to
`S_cal` *only* if an `s_gen` is supplied AND `w_gen > 0`). No Phase 3/3b/4 pipeline
supplies it or sets `w_gen > 0`. The counterfactual/scoring paths never use it.

## 5. Is `w_gen` nonzero anywhere?
**No.** `ScoreWeights.w_gen = 0.0` everywhere; grep finds no override.

## 6. Previous metrics
`artifacts/day34/day34_assessment.json`: **Sgen ROC-AUC 0.4749** (below random) on the
old circular benchmark.

## 7. Mode collapse?
**Yes, severe** (`artifacts/day34/day34_generative_eval_summary.json`):
generated unique tokens **127** vs real **4587**; generated length mean **254**
(≈max_len) vs real **47**; marginal JS **0.27**; top-1000 token Jaccard **0.127**.
The model emits near-max-length sequences from a tiny token set.

## 8. Leakage?
The generative path itself does not read anomaly labels for scoring (surprise is
unsupervised). However, its *training/eval provenance is invalid for the current
benchmark*: trained on old circular-era data, evaluated previously on the circular
benchmark. No answer-key leakage, but no valid-benchmark evidence either.

## 9. Feasible to repair now?
- **Checkpoint load:** the Day-33 checkpoint does NOT load into the current
  `diffusion.py` (architecture drift): module renames (`encoder.`→`denoiser.`,
  `pos_embedding`→`position_embedding`, `norm`/`out`→`output_head.0/.1`) AND a genuine
  shape change in the timestep-embedding MLP (checkpoint 128-wide vs current 512-wide).
  A **minimal compatibility shim** (key remap + skip the drifted time-MLP) loads all
  token/transformer/output weights and lets us SCORE benchmark-v2 (diagnostic-only).
- **Real fix (mode collapse / valid benchmark):** requires retraining on benchmark-v2
  clean-normal-only with anti-collapse measures — **large training, out of Phase-5 scope.**

## 10. What a defensible generative claim would require
Retrain on benchmark-v2 clean-normal-only; fix mode collapse (discrete/latent
diffusion or length/entropy regularization); demonstrate Sgen ROC-AUC > 0.50 on
benchmark-v2 with a CI clearing chance AND an ablation showing it improves
`combined_real`. None of this holds today.

## Fresh benchmark-v2 diagnostic (this phase, compat-shim load)
Scoring the real trained weights (minus the un-loadable time-MLP) over benchmark-v2
test (n=6307): **Sgen ROC-AUC 0.4866, AP 0.20**, score std 0.039 (near-constant);
mean normal 0.596 vs anomaly 0.593 (anomalies score *lower* — wrong direction).
→ **No usable anomaly signal on the valid benchmark.**
