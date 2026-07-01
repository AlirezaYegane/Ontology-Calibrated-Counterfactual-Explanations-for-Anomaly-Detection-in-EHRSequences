# Claims Matrix — Scientific Contract (Phase 0)

**Purpose.** Single source of truth for what this project may claim. Every claim is
mapped to its current evidence and a status label. No claim may appear in the paper
unless it is `supported_now` (with a backing artifact) or has been upgraded by a later
phase. This document is intentionally strict; it exists for scientific honesty, not
marketing.

**Generated:** Phase 0 (Day 51 recovery track). **Repo state audited:** Day 50.

## Phase 7 final decisions (authoritative; see `artifacts/phase7/final_claims_decision.json`)
On benchmark-v2 (final benchmark), with paired-bootstrap significance:
- **supported_now:** non-circular benchmark (C2b); real ontology integration (C5);
  **real ontology > legacy** (C4: 0.7881 vs 0.7358, +0.052 p≈0); leakage-free
  counterfactual (C10); **effective counterfactual repair** (89.99% among flagged);
  reproducible infrastructure.
- **unsupported:** detector improves detection (0.4525, below chance); combined >
  ontology-only (−0.085, p≈0, significantly worse).
- **removed_from_core:** Sgen (C7).
- **future_work:** clinical validation (C11); external generalization (eICU schema
  mismatch — blocked).
- **Final main method:** ontology-centered `S_main = S_ont`; detector = negative result.


## Status labels

| Label | Meaning |
|---|---|
| `supported_now` | Backed by code + artifact in the repo today; safe to claim. |
| `partially_supported` | Some evidence exists but it is weak, narrow, or caveated. |
| `needs_evidence` | Plausible but not yet demonstrated; requires a specific future phase. |
| `cut_unless_fixed` | Currently contradicted by the repo's own artifacts; must be removed unless a named phase repairs it. |
| `future_work_only` | Out of scope for the current evidence; may appear only as future work. |

## Master claim table

| # | Claim | Status | Evidence (for / against) | Required fix / phase |
|---|---|---|---|---|
| C1 | Real MIMIC-IV preprocessing into ICU code sequences | `supported_now` | `data/processed/mimiciv_*.pkl` (~380k train admissions, vocab ~47k); `src/preprocessing/extract_mimiciv.py`; split manifest `data/processed/mimiciv_split_manifest.json` | Verify subject-level splits (Phase 1/3) |
| C2 | A synthetic anomaly benchmark exists | `partially_supported` | `src/preprocessing/anomaly_injection.py`; `data/processed/mimiciv_*_synth_anomaly.pkl` (val: 71,602 rows; 16,541 anomalies) | Benchmark validity is in doubt (see C2b); Phase 1 |
| C2b | The synthetic anomaly benchmark measures real anomaly reasoning (not injection artifacts) | `supported_now` (scoped, synthetic) | FOR (Phase 1b): `data/processed/benchmark_v2/` — anomalies are **relational** (gender-flip injects no token; indication-removal; mutual-exclusion partner), every model-visible token stays common. Triviality `artifacts/diagnostics/rf2_triviality_v2.json`: strongest label-free trivial signal **0.6127 < 0.80** (v1 was 0.94). Subject-level splits (overlap 0); strict model_visible/audit/hidden_eval separation. SCOPE: ontology-backed **synthetic** benchmark, not real-world external validation; demographic per-type triviality 0.95 is a no-token **selection effect** (sequence identical to source normal), documented in `artifacts/phase1b/`. | Done (Phase 1b). Real-world external validation = future work |
| C3 | Supervised detector achieves ROC-AUC ≈ 0.80 on the synthetic benchmark | `supported_now` (as a number) / `partially_supported` (as a scientific result) | `artifacts/day45/detector_only/day45_test_set_metrics.json` (ROC-AUC 0.8002, AP 0.733) | Number is real; its meaning is limited by C2b. Re-evaluate non-circularly in Phase 3 |
| C4 | Ontology-calibrated anomaly scoring (ontology improves ranking) | `partially_supported` (Phase 3b) | FOR (Phase 3b on benchmark-v2, `artifacts/phase3b/phase3b_scoring_eval.json`): after authoring curated rule packs, `ontology_only_real` ROC-AUC **0.7881** (95% CI [0.7743, 0.8016], clears chance), AP 0.542; coverage 100% demographic / 100% forbidden / 50% medication vs 13% normal FP; rules use only model-visible codes+gender (no leakage, `test_phase3b_rule_leakage.py`). `combined_real` (0.658) > `detector_only` (0.425). AGAINST/scope: `combined_real` (0.658) < `ontology_only` (0.788) — the smoke-scale detector (0.425, weight 0.7) dilutes the combination, so "the *calibrated combination* beats a strong detector" is NOT yet shown. Earlier Phase-3 at-chance (0.50) was a ruleset gap, now closed. | Keep scoped: ontology has demonstrated **standalone** discriminative value (beyond interpretability). **Phase 6 update:** a FULL-SCALE unsupervised detector was trained on benchmark-v2 (`artifacts/phase6/`) and is STILL below chance (≈0.46) — the v2 anomalies are relational, not next-token-surprising — so the combination still does not beat ontology-only. The method is **ontology-centered**; no further "combination beats strong detector" claim is pursued |
| C5 | Real SNOMED CT / RxNorm ontology integration | `supported_now` (scoped) | FOR (Phase 2b-fix): real UMLS 2026AA / SNOMED CT US / RxNorm parsed into `ontologies/processed/`; canonical engine + maps loader-verified (ICD-9 `428.0`→SNOMED CHF). Real MIMIC-IV coverage **diagnosis 0.8006, medication 0.7767** (both targets met) via authoritative MRMAP + CUI bridge + RxNorm ingredient matching. AGAINST/scope: ICD-10 diag 0.92 but ICD-9 lower; `rxnorm_classes` unavailable; live `ontology_aware.py` scoring still uses the legacy ICD-prefix fallback (canonical engine not yet wired into scoring). | Wire canonical engine into `src/scoring/ontology_aware.py` (Phase 3); keep claim scoped to coverage achieved |
| C6 | Anomaly decomposition into detector / statistical / ontology components (`Sdet = Sgen + Sont`) | `partially_supported` (reframed) | Phase 3: scoring is now an **explicit, configurable** normalized weighted score (`compute_calibrated_score` / `OntologyAwareScorer`, `w_gen=0` default, `Sgen` diagnostic-only, no hidden weights), now **exercised end-to-end on the non-circular benchmark-v2 with bootstrap CIs** (`artifacts/phase3/phase3_scoring_eval.json`; threshold selected on val, applied to test). The proposal's additive identity `Sdet=Sgen+Sont` remains **dropped** (not realized); the defensible framing is a transparent calibrated combination. | Keep reframed; `Sgen` term gated on Phase 5. Combination is now validated to RUN on v2 (Phase 3); a *useful* combination awaits ontology-rule coverage (C4) |
| C7 | Diffusion-based generative surprise `Sgen` separates anomalies | `cut` (Phase 5 gate: remove_from_core) | AGAINST (Phase 5 on benchmark-v2, `artifacts/phase5/`): Sgen ROC-AUC **0.4868** (CI [0.4633, 0.5109], below chance), anomalies score *lower* than normals, corr(Sgen,S_ont) **−0.07**. Adding Sgen **harms** the combined score (0.6545→0.637; paired ΔAUC −0.0175, CI [−0.027,−0.009], p≈0). Confirms the old 0.475 on valid data. | **Removed from the core** (`w_gen=0`). Reframe as a negative result / future work. Revisit only if a retrained generative variant clears the gate |
| C8 | Diffusion model generates realistic EHR sequences | `cut` (Phase 5) | AGAINST: severe mode collapse — generated length ~254 vs real ~47; **127 of 4,587** tokens used; JS 0.27. The only checkpoint is old-data and not loadable without a compat shim. | Cut. Future work: discrete/latent diffusion + retrain on benchmark-v2 clean-normal-only |
| C9 | Diffusion-based counterfactual generation | `cut_unless_fixed` | AGAINST (unchanged after Phase 4): the Phase 4 counterfactual is **ontology/edit-based**, not diffusion; the diffusion model is never invoked. The defensible counterfactual is the ontology one (C10), not a diffusion one. | Phase 5 gate (only if a generative variant passes); otherwise keep cut |
| C10 | Ontology-guided counterfactual repair (minimal, plausible edits) | `supported_now` (scoped) | FOR (Phase 4, `artifacts/phase4/`): rewritten leakage-free generator (`src/explanations/counterfactual.py`) reads only model-visible sequence + demographics + real ontology + scorer feedback (leakage tests pass; misleading answer keys ignored). On benchmark-v2: **90.5% valid repair among ontology-flagged anomalies** (62.1% overall), mean ΔS_ont 0.644, **median 1 edit**; validated by an independent scorer; forbidden/medication 100%, demographic 67.5%. The old circular answer-key logic + its leaky test/script were deleted. AGAINST/scope: validity = ontology-violation resolution, NOT clinician-verified correctness (C11); demographic gender-flips with many obstetric codes exceed a small edit budget. | Done (leakage-free, minimal, ontology-grounded). Clinical validity beyond rule-plausibility = C11/Phase 7 |
| C11 | Clinical plausibility of explanations | `needs_evidence` | No clinician validation performed (acknowledged in `artifacts/day42/day42_gap_register.csv`, row G4) | Phase 7 clinician review; until then frame as preliminary |
| C12 | A*-paper-level novelty (as currently framed) | `cut_unless_fixed` | The headline novelties (C5–C10) are unsupported or circular today | Phases 1–4 must pass; novelty re-anchored on leakage-free ontology counterfactuals + non-circular benchmark |
| C13 | Demographic metadata available for sex/age anomaly reasoning | `supported_now` (scoped) | FOR (Phase 1b): benchmark-v2 `demographic_incompatibility` flips the **real `gender` field** (no token inference); `gender`/`age_group` are the only demographic model-visible fields, with subject-level splits. AGAINST (residual): the historical supervised pipeline still inferred sex from tokens — not used in the v2 path. | Done for the benchmark. Detector use of demographics at full scale = later experiment phase |
| C14 | Reproducible, script-generated experiment artifacts | `partially_supported` | FOR: disciplined per-day artifacts + tests. AGAINST: absolute `D:\Article\...` paths baked into artifacts; some metrics computed off pre-baked CSVs, not the live pipeline | Phase 6 portability; Phase 7 end-to-end regeneration |

## Summary buckets

### Safe to claim now (`supported_now`)
- C1 Real MIMIC-IV preprocessing.
- C2b **Non-circular synthetic benchmark (benchmark-v2)** — relational anomalies; strongest trivial signal 0.6127 < 0.80; subject-level splits; leakage guards (Phase 1b). Scoped: synthetic, not external validation.
- C3 Supervised detector reaches ROC-AUC ≈ 0.80 **on the old circular synthetic benchmark** (number only). On the non-circular benchmark-v2 a **smoke-scale unsupervised** detector is at chance (0.42); full-scale training deferred.
- C5 Real SNOMED/RxNorm integration **(scoped, Phase 2b-fix)** — diagnosis coverage 0.80, medication 0.78 on MIMIC-IV via authoritative UMLS crosswalks. Canonical engine is now wired into and exercised by the Phase 3 eval (mapping works); but its default ruleset does not yet fire on the v2 anomaly families (see C4).
- C13 **Demographic metadata** — benchmark-v2 flips the real `gender` field (no token inference). Scoped to the benchmark.
- C10 **Leakage-free ontology counterfactual repair** (Phase 4) — reads only model-visible + ontology; 90.5% valid repair among ontology-flagged anomalies, median 1 edit. Scoped: ontology-violation resolution, not clinician-verified.

### Requires further fixes (`partially_supported` / `needs_evidence`)
- C2 benchmark (v1) validity → superseded by C2b benchmark-v2.
- C4 ontology **ranking** value → Phase 3b made the ontology a credible **standalone** ranker (0.79, CI clears chance); the calibrated *combination* beating a strong detector awaits full-scale training.
- C6 defensible scoring formulation → explicit/configurable, exercised on v2 with CIs; ontology term now carries real signal (Phase 3b).
- C11 clinical plausibility → Phase 7.
- C14 reproducibility hardening → Phase 6/7.

### Must be removed unless repaired (`cut_unless_fixed`)
- C7 Sgen separation, C8 realistic generation,
  C9 diffusion counterfactual, C12 A* novelty framing.
- (C2b non-circular benchmark **resolved to `supported_now` (scoped)** in Phase 1b. C5 real ontology integration **resolved to `supported_now` (scoped)** in Phase 2b-fix. C4 ontology ranking **upgraded to `partially_supported`** in Phase 3b — real standalone signal 0.79. **C10 ontology counterfactual repair resolved to `supported_now` (scoped)** in Phase 4 — leakage-free, 90.5% valid among flagged.)
- Note: the proposal's additive identity `Sdet=Sgen+Sont` (formerly tracked under C6) remains **dropped** — the defensible framing is the transparent calibrated combination.

### Future work only (`future_work_only`)
- High-frequency physiological time series (HiRID).
- Multi-site deployment / prospective clinical use.
- Generative surprise (`Sgen`) and diffusion generation/repair — **Phase 5 gate returned `remove_from_core`** (Sgen below chance + harms combined + mode collapse). Now a documented **negative result**; only revisit if a retrained generative variant (benchmark-v2 clean-normal-only + anti-collapse) later clears the gate.

## What an A* reviewer attacks first
1. "Your anomalies are detectable by sequence length / token counts — show trivial baselines." → **Addressed (Phase 1b):** benchmark-v2 strongest trivial signal 0.6127 < 0.80.
2. "You call it *ontology*-calibrated but there is no ontology in the repo." → **Addressed (Phase 2/2b):** real UMLS/SNOMED/RxNorm, coverage 0.80/0.78.
3. "Your counterfactual reads the corruption note / answer-key columns." → **Addressed (Phase 4):** counterfactual rewritten leakage-free (reads only model-visible + ontology); leakage tests pass, misleading answer keys ignored; old leaky logic deleted. 90.5% valid repair among ontology-flagged anomalies, median 1 edit.
4. "Your generative surprise is worse than random; why keep it?" → **Addressed (Phase 5 gate = remove_from_core):** Sgen is below chance on benchmark-v2 (0.4868) and significantly *harms* the combined score (paired ΔAUC −0.0175, p≈0); it is removed from the core (`w_gen=0`) and reported as a negative result. We do not keep it.
5. "Your ablation gain is ~0.001 AUC with no significance test." → **Addressed (Phase 3 → 3b):** on benchmark-v2 the real ontology went from at-chance (0.50, sparse rules) to a credible standalone ranker (**0.79, CI [0.774, 0.802]**) after auditable rule authoring, reported with bootstrap CIs. Honest caveat: the calibrated *combination* (0.66) still trails ontology-alone because the detector is smoke-scale.
