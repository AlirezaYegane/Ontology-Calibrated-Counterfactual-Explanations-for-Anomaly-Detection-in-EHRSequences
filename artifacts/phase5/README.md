# Phase 5 — Generative / Diffusion / Sgen Decision Gate

## Decision: `remove_from_core`  ·  Status: `phase5_remove_sgen_from_core`

The diffusion generative-surprise `Sgen` is **removed from the core method**. On the
non-circular benchmark-v2 it scores **below chance** and **harms** the combined score.
`w_gen` stays **0.0**; `Sgen` is diagnostic/appendix-only. This is a decision gate, and
the evidence points one way.

> Everything here is **diagnostic-only**: the only available diffusion checkpoint was
> trained on **old circular-era data** and is **mode-collapsed**; it loads into the
> current code only via a compatibility shim. Not paper evidence.

## Evidence (benchmark-v2 test)
**Sgen alone:** ROC-AUC **0.4868** (CI [0.4633, 0.5109]), AP 0.198, score std 0.04;
mean surprise normal **0.5915** vs anomaly **0.5876** (anomalies score *lower* — wrong
direction). Per family: demographic 0.469, medication 0.479, forbidden 0.539. It does
not correlate with the ontology signal (corr(Sgen, S_ont) = **−0.07**).

**Ablation — does Sgen help the combined score?** No; it hurts, credibly:

| variant | ROC-AUC | 95% CI |
|---|---:|---|
| ontology_only_real | **0.807** | [0.791, 0.827] |
| detector_only (smoke) | 0.427 | [0.402, 0.453] |
| combined_real **without** Sgen | 0.6545 | [0.629, 0.684] |
| combined_real **with** Sgen | **0.637** | [0.613, 0.662] |
| legacy_baseline | 0.6367 | [0.614, 0.662] |

Sgen ΔROC-AUC (with − without) = **−0.0175**, paired bootstrap CI **[−0.027, −0.009]**,
**p ≈ 0** → adding Sgen significantly *degrades* the combined score.

## Mode collapse (why generation is broken)
Generated unique tokens **127** vs real **4587**; generated length mean **254** (≈max)
vs real **47**; marginal JS **0.27**. The model emits near-max-length sequences from a
tiny token set.

## What was built
- `src/evaluation/generative_gate.py` — formal, strict gate
  (`keep_main` / `diagnostic_only` / `remove_from_core` / `blocked_no_valid_model`).
- `scripts/run_phase5_generative_eval.py` — compatibility-shim checkpoint loader +
  Sgen scoring on benchmark-v2 + correlations + gate; **blocked report** if no model.
- `scripts/run_phase5_score_ablation.py` — with/without-Sgen ablation + paired test.
- Tests: `tests/test_phase5_{generative_gate,sgen_scoring_defaults,generative_eval}.py`.

### Minimal repair (Part C, in scope)
Only a **compatibility-shim load** (key remap `encoder.→denoiser.` etc. + skip the
arch-drifted time-MLP) + deterministic multi-noise-sample scoring + min-max
normalization. **No retraining, no architecture redesign.** Mode collapse is *not*
fixed (that needs large retraining = future work).

## Scoring defaults (Part E)
`w_gen = 0.0` by default (`ScoreWeights`). The core equation stays
`S_cal = (w_det·S_det + w_ont·S_ont′) / (w_det + w_ont)`. `Sgen` enters `S_cal` only if
an `s_gen` is supplied **and** `w_gen > 0` — never by default. Diffusion code is
retained but isolated as diagnostic/legacy (not deleted; Phase-2/3 tests still import
the model module).

## Claim impact
- **C7** (Sgen separates anomalies): confirmed **removed from core** — below chance,
  harms the combined score. Negative result / future work.
- **C8** (realistic diffusion generation): **cut** — severe mode collapse.

## Next
**Phase 6** (full-scale training + experiment infrastructure) may proceed **without
Sgen in the core**. Generative retraining (benchmark-v2 clean-normal-only + anti-collapse)
is a separate optional future phase; only revisit Sgen if it later clears the gate.
**No H200 unless Phase 6 explicitly requires it.**
