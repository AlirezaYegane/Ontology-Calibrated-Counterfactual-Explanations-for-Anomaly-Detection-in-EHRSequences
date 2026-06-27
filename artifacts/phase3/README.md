# Phase 3 — Detector and Scoring Rebuild

> **Phase 3b update:** the at-chance ontology finding below was the *sparse-rules*
> baseline. After authoring curated rule packs (Phase 3b, `artifacts/phase3b/`),
> `ontology_only_real` rose to **ROC-AUC 0.7881 (CI [0.774, 0.802])** — a credible
> standalone ranking signal. `combined_real` (0.658) beats `detector_only` (0.425)
> but trails `ontology_only` (smoke-detector dilution). See
> [artifacts/phase3b/README.md](../phase3b/README.md).

## Status: `fully_closed_with_valid_benchmark_v2`

Phase 3 is closed. The non-circular **benchmark-v2** (Phase 1b) exists and passes the
triviality gate (strongest trivial signal **0.6127 < 0.80**); the real-ontology
scorer, the unsupervised detector, calibration, and statistics all run **end-to-end**
on it, producing **valid non-circular test-set detection metrics with bootstrap CIs**
(threshold selected on val, applied to test). The full test suite passes (200).

> **Scope honesty.** This is an ontology-backed **SYNTHETIC** benchmark, not
> real-world external validation. The detector is **smoke-scale** (tiny GRU, 2 epochs,
> 4,000 train normals); its metrics are **preliminary non-circular evidence, NOT final
> paper SOTA**. `final_paper_evidence_claimable = false`. Full-scale detector training
> is deferred to a later experiment phase.

## Results on benchmark-v2 test (n=6,307; anomaly rate 0.218)
| Variant | ROC-AUC | 95% CI | AP |
|---|---:|---|---:|
| detector_only (smoke) | 0.4247 | [0.407, 0.440] | 0.180 |
| ontology_only_real | 0.5013 | [0.500, 0.503] | 0.219 |
| combined_real_ontology | 0.4255 | [0.408, 0.441] | 0.181 |
| combined_legacy_ontology | 0.6625 | [0.644, 0.679] | 0.446 |

### Honest reading (do not overclaim)
- **Smoke detector is at/below chance (0.42).** A 2-epoch tiny GRU on 4k normals has
  not learned the normal distribution, and several v2 anomalies carry no next-token
  signal by construction (gender-flip leaves the sequence identical; indication-removal
  makes it shorter / more predictable). Expected at smoke scale; **not** a final result.
- **Real ontology is at chance (0.50).** Diagnostic
  (`ontology_rule_coverage_v2.json`): the engine's default ruleset fires on ~1.5% of
  demographic anomalies, **0%** of forbidden-cooccurrence, 0.11% of
  medication-indication-mismatch — barely above the 0.10% normal rate. Token→SNOMED/
  RxNorm mapping works (222–446 codes/record); the gap is **ruleset coverage**, not data
  or wiring. **No ontology ranking gain on v2.**
- **combined_real ≈ detector_only.** With S_ont≈0, the calibrated score is dominated by
  the weak detector; its val-selected threshold collapses to ~0 (recall 1.0, precision =
  base rate).
- **combined_legacy (0.66)** is the only above-chance variant, but it is the **legacy
  handcrafted ICD-prefix scorer** (over-fires; documented circular). It is retained as
  the historical comparison and is **not claimable as clean evidence**.

## What was built / fixed this phase
- Real ontology-aware scoring wired into the eval (`src/scoring/ontology_aware.py`),
  explicit equation `S_cal = (w_det·S_det + w_ont·S_ont′ + w_gen·S_gen′)/Σactive`,
  `w_gen=0` (Sgen diagnostic-only).
- Unsupervised next-token GRU detector trained on **normal-only** v2 train split.
  **OOM fix:** `anomaly_scores` now mini-batches (it was building a `[N, max_len,
  vocab]` ≈ 59 GB logits tensor over a whole split); per-record scores are
  batch-invariant (padding masked, normalised by non-pad length).
- Val-only threshold calibration + bootstrap-CI statistics, both exercised on v2.

## Artifacts
- `artifacts/phase3/phase3_scoring_eval.{json,md}`, `score_variant_table.csv`
- `artifacts/phase3/ontology_rule_coverage_v2.json` (explains the 0.50)
- `artifacts/phase3/detector_unsup_v2/` (smoke checkpoint + vocab + train metrics)
- `artifacts/phase1b/` (benchmark-v2 description + triviality verdict)

## Claim impact
- **C2b** non-circular benchmark: `cut_unless_fixed → supported_now` (scoped, synthetic).
- **C4** ontology ranking gain: **still not supported** — tested non-circularly, found
  at chance (ruleset coverage gap). No improvement claimed.
- **C6** scoring formulation: explicit/configurable calibrated score now exercised on a
  non-circular benchmark with CIs (stays reframed `partially_supported`).
- **C13** demographic metadata: `partially_supported → supported_now` (scoped) — v2
  flips the **real** gender field, no token inference.

## Next
Phase 3 closure confirmed pending user sign-off. Next is **Phase 4** (leakage-free
counterfactual repair). Separately, a later experiment phase must (a) train the detector
at full scale and (b) author ontology rules covering the v2 anomaly families so the real
scorer can actually fire. **Do not proceed to Phase 4/5/H200 yet.**
