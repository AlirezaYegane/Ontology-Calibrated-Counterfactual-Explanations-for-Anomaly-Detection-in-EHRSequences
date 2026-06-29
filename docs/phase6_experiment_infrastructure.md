# Phase 6 — Experiment Infrastructure

Reproducible, resumable, data-safe full-scale training + evaluation for the
**Sgen-free** ontology-calibrated core (benchmark-v2 + real ontology scorer +
unsupervised detector + combined score without Sgen + leakage-free counterfactuals).

This phase builds infrastructure; final tables/ablations belong to Phase 7.

## Components
| Piece | File |
|---|---|
| Config (dataclass + YAML) | `src/experiments/config.py`, `configs/phase6_detector_{smoke,full,h200}.yaml` |
| Training (resume + val early-stop) | `src/training/train_detector_unsup.py::train_detector_full`, `scripts/run_phase6_train_detector.py` |
| Detector eval | `scripts/run_phase6_evaluate_detector.py` |
| Combined eval (Sgen-free) | `scripts/run_phase6_combined_eval.py` |
| Shared eval helpers | `src/experiments/eval_common.py` |
| Experiment index | `src/experiments/tracking.py` → `artifacts/phase6/experiment_index.{json,md}` |
| Pipelines | `scripts/run_phase6_{smoke,full_local,full_gpu}.py` |

## How to run
**Smoke (CPU, fast, used by tests):**
```
python scripts/run_phase6_smoke.py
```
**Full local (CPU, slow):**
```
python scripts/run_phase6_full_local.py
```
**Full GPU / H200 (CUDA if available; falls back to CPU, never hard-fails):**
```
python scripts/run_phase6_full_gpu.py
python scripts/run_phase6_full_gpu.py --config configs/phase6_detector_h200.yaml
```
Each pipeline trains → evaluates the detector → runs the Sgen-free combined eval →
updates the experiment index.

## Reproducing a run
A run is identified by `run_id` (default `<experiment_name>_seed<seed>`). Re-running
the same config **resumes** from `checkpoints/last.pt` (model + optimizer + epoch +
best-metric state). Determinism: fixed `seed`, seeded shuffles, `set_seed`. The
`config.json` snapshot + `train_summary.json` capture the exact settings.

## What is ignored vs committed
**Committed (aggregate, no PHI):** `train_metrics.jsonl`, `train_summary.json`,
`detector_eval.{json,md,csv}`, `combined_eval.{json,md,csv}`, `experiment_index.*`,
`phase6_summary.json`.
**Git-ignored (MIMIC-derived / heavy):**
`artifacts/phase6/runs/*/checkpoints/` (`.pt` + vocab), `.../vocab/`, `.../ignored/`
(per-record scores), and any `*.pt|*.pkl|*.parquet|per_record*` under `artifacts/phase6`.

## Safety invariants
- Detector trains on **normal-only** benchmark-v2 train sequences (no anomaly labels).
- Thresholds / model selection use **validation only**; test is never tuned.
- `anomaly_score` mini-batches (no giant-tensor OOM).
- **Sgen stays disabled**: `ExperimentConfig` raises if `sgen_enabled` or `w_gen != 0`.

## Why Sgen is disabled
Phase 5 (`artifacts/phase5/`) returned `remove_from_core`: on benchmark-v2 Sgen is
below chance (ROC-AUC 0.4868) and statistically harms the combined score. `w_gen = 0`
in every Phase 6 config; the combined score is `S_cal = (w_det·S_det + w_ont·S_ont′)/
(w_det + w_ont)`.

## What Phase 7 should consume
- A trained detector checkpoint (under the ignored `checkpoints/`), regenerable from a
  config via `run_phase6_train_detector.py`.
- `combined_eval.json` variant table (detector_only / ontology_only_real /
  combined_real_without_sgen / legacy_baseline) with bootstrap CIs + paired diffs.
- `experiment_index.json` for run provenance (git commit, config, seed, device).

Phase 7 adds the final evaluation/ablation tables; Phase 6 makes those runs
reproducible.
