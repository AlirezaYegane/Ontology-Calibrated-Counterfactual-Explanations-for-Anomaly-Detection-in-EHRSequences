# Phase 6 — Full-Scale Training & Experiment Infrastructure

Reproducible, resumable, data-safe training + evaluation for the **Sgen-free**
ontology-calibrated core. See [docs/phase6_experiment_infrastructure.md](../../docs/phase6_experiment_infrastructure.md)
for the full how-to.

## Status: `phase6_infrastructure_complete_full_run_available`
Configs + training/eval pipelines + tracking exist; smoke validated; **one full GPU
detector run completed** (see `runs/phase6_detector_full_gpu/` and
`experiment_index.json`).

## Quick start
```
python scripts/run_phase6_smoke.py                 # CPU, fast (used by tests)
python scripts/run_phase6_full_gpu.py              # CUDA if available, else CPU
python scripts/run_phase6_full_gpu.py --config configs/phase6_detector_h200.yaml
```

## Headline finding (honest)
A **full-scale** unsupervised next-token detector (run `phase6_detector_full_gpu`: 25
epochs on CUDA, all 20,570 train normals) still does **not** beat chance on benchmark-v2
(test ROC-AUC **0.4525**, CI [0.435, 0.469]), because the v2 anomalies are **relational**
(gender-flip, indication-removal, mutual-exclusion) and carry little next-token surprise.

| variant (full GPU run) | ROC-AUC | 95% CI | AP |
|---|---:|---|---:|
| detector_only | 0.4525 | [0.435, 0.469] | 0.190 |
| **ontology_only_real** | **0.7881** | [0.774, 0.802] | 0.542 |
| combined_real_without_sgen | 0.7036 | [0.687, 0.720] | 0.404 |
| legacy_baseline | 0.6989 | [0.681, 0.715] | 0.427 |

**`ontology_only_real` is the strongest variant; the main claim stays ontology-centered.**
Adding the detector to the combined score **significantly hurts** vs ontology-only
(paired ΔAUC **−0.0845**, CI [−0.097, −0.074], **p≈0**). Real ontology significantly
beats the legacy baseline (+0.089, p≈0). This confirms Phase 3/3b at full scale: the
detector is below chance and non-additive.

## What is committed vs ignored
- **Committed:** per-run `train_metrics.jsonl`, `train_summary.json`,
  `detector_eval.*`, `combined_eval.*`; `experiment_index.*`; `phase6_summary.json`.
- **Ignored (MIMIC-derived/heavy):** `runs/*/checkpoints/`, `runs/*/vocab/`,
  `runs/*/ignored/`, and any `*.pt|*.pkl|*.parquet|per_record*`.

## Invariants
Sgen disabled everywhere (`w_gen=0`, config raises otherwise); detector trains on
normal-only train sequences; thresholds/selection use validation only; test never
tuned; mini-batched scoring (no OOM).

## Next
Phase 7 (final evaluation + ablations). The detector is reproducible from config; the
honest result that the detector is weak and the ontology carries the signal should be
the Phase 7 headline.
