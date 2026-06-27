# Phase 3 — Detector & Scoring Status

**Status: `fully_closed_with_valid_benchmark_v2`** (closed Phase 3) **+
`phase3b_success_rule_signal_supported`** (Phase 3b rule authoring). The pipeline
runs end-to-end on the non-circular benchmark-v2 with bootstrap CIs. **Phase 3b**
then authored curated ontology rule packs, lifting the real ontology scorer from
**chance (0.5013) to ROC-AUC 0.7881** (CI [0.7743, 0.8016]) — a real, credible
standalone ranking signal with no label leakage. Still smoke-scale ⇒
`final_paper_evidence_claimable = false`; full-scale detector training deferred.

> **Nuance (do not overclaim):** `combined_real_ontology` (0.658) is *below*
> `ontology_only_real` (0.788) because the smoke-scale detector (0.425, below
> chance, weight 0.7) dilutes the strong ontology term. The ontology is a strong
> **standalone** ranker; the calibrated *combination* is not the best variant until
> a full-scale detector replaces the smoke one. See `artifacts/phase3b/`.

## Scoring path
| Mode | Source of `S_ont` | Use |
|---|---|---|
| `real` | canonical `OntologyEngine` over SNOMED/RxNorm codes mapped from EHR tokens | **primary** |
| `legacy` | ICD-prefix `compute_s_ont` | historical comparison only (documented circular) |
| `disabled` | forced 0 | ablation |

- Explicit, configurable equation `S_cal = (w_det·S_det + w_ont·S_ont′ + w_gen·S_gen′)/
  Σ(active weights)`; `w_gen = 0` (Sgen diagnostic-only, ROC-AUC was 0.475).
- Real mode never silently falls back to legacy.

## Detector path
- **Primary:** `UnsupervisedSequenceDetector` — next-token GRU trained on **normal
  sequences only** (benchmark-v2 train split is clean-normal-only). Ran on v2 at
  **smoke scale** (embed/hidden 64, 1 layer, 2 epochs, 4,000 normals). Full-scale run
  deferred.
- **OOM fix:** `anomaly_scores` now mini-batches; previously it padded an entire split
  into one `[N, max_len, vocab]` logits tensor (≈ 59 GB) and crashed. Per-record scores
  are batch-invariant because padding is masked and NLL is normalised by non-pad length.
- **Secondary:** Day-20 supervised classifier, relabelled `supervised_synthetic_baseline`
  (circular upper bound; not used as evidence).

## Eval result on benchmark-v2 test (n=6,307, anomaly rate 0.218)
| Variant | ROC-AUC | 95% CI | AP |
|---|---:|---|---:|
| detector_only (smoke) | 0.4247 | [0.407, 0.440] | 0.180 |
| ontology_only_real | 0.5013 | [0.500, 0.503] | 0.219 |
| combined_real_ontology | 0.4255 | [0.408, 0.441] | 0.181 |
| combined_legacy_ontology | 0.6625 | [0.644, 0.679] | 0.446 |

**Reading (Phase 3, sparse default rules).** Smoke detector at/below chance; the real
ontology scorer was **at chance (0.50)** because its default ruleset didn't encode the
v2 relationships (fired on ~1.5% demographic / 0% forbidden / 0.11% medication vs 0.10%
normals — `artifacts/phase3/ontology_rule_coverage_v2.json`).

## Phase 3b update — after ontology rule authoring (`artifacts/phase3b/`)
Curated rule packs (`src/ontology/rule_packs.py`, `rule_loader.py`) raised the real
ontology coverage to **100% of demographic, 100% of forbidden, 50% of medication**
anomalies vs **13% of normals**, lifting the standalone ranker:

| variant | ROC-AUC (Phase 3) | ROC-AUC (Phase 3b) | 95% CI (3b) |
|---|---:|---:|---|
| detector_only (smoke) | 0.4247 | 0.4247 | [0.407, 0.440] |
| **ontology_only_real** | 0.5013 | **0.7881** | [0.774, 0.802] |
| combined_real_ontology | 0.4255 | 0.6581 | [0.640, 0.676] |
| combined_legacy_ontology | 0.6625 | 0.6625 | [0.644, 0.679] |

`ontology_only_real` clears chance with a comfortable CI; `combined_real` now beats
`detector_only` but trails `ontology_only` (smoke detector dilution). The rules use
**only** model-visible codes + gender (source-token ICD families + RxNorm + SNOMED);
no labels/hidden/audit metadata (`tests/test_phase3b_rule_leakage.py`).

## Calibration & statistics
- Validation-only threshold selection (best-F1 / fixed-recall / fixed-precision);
  test-set application is a separate call. No test-set threshold tuning.
- Bootstrap CIs + paired bootstrap difference with two-sided p-value. All variant AUCs
  above are reported with bootstrap CIs.

## Claim impact
- **C2b** non-circular benchmark → `supported_now` (scoped, synthetic): triviality gate
  0.6127 < 0.80, subject-level splits, leakage guards, tests.
- **C4** ontology ranking gain → upgraded to `partially_supported` (Phase 3b): the real
  ontology is now a **credible standalone ranker** on the non-circular benchmark
  (`ontology_only_real` 0.79, CI clears chance, no leakage). NOT yet claimable as "the
  calibrated combination beats a strong detector" — the combination trails ontology-only
  because the detector is smoke-scale. Full-scale detector needed before that headline.
- **C6** scoring formulation → stays reframed `partially_supported`; now exercised on a
  non-circular benchmark with explicit equation + CIs.
- **C13** demographic metadata → `supported_now` (scoped): v2 flips the real `gender`
  field, not token-inferred sex.

## Next
Phase 4 (leakage-free counterfactual repair). Later experiment phase: full-scale detector
training + ontology-rule authoring for the v2 anomaly families, then re-evaluate. No H200.
