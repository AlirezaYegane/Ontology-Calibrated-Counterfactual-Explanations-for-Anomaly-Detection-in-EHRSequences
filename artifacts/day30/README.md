# Day 30 — Diffusion Training Loop

## Status
Complete.

## Goal
Implement and verify the first working training loop for the Day 29 diffusion model.

## What was implemented
- `src/training/diffusion_training_utils.py`
- `src/training/train_diffusion.py`
- `tests/test_train_diffusion_smoke.py`

## Verified results
- Test suite: 4 passed
- Smoke training completed on GPU
- Global steps: 80
- First epoch loss: 1.0762
- Last epoch loss: 1.0021
- Best epoch: 2
- Best loss: 1.0021
- Loss decreased: true

## Training logic
- Load Day 27 diffusion-ready sequence tensor artifact
- Sample random diffusion timestep `t`
- Convert token ids to embedding-space clean representation
- Add Gaussian noise using cosine beta schedule
- Predict noise with the Day 29 Transformer-based diffusion model
- Optimize mask-aware MSE over non-pad positions
- Save per-step metrics and checkpoints

## Generated runtime artifacts
- `outputs/diffusion/day30_smoke/metrics.jsonl`
- `outputs/diffusion/day30_smoke/summary.json`
- `outputs/diffusion/day30_smoke/config_resolved.json`
- `outputs/diffusion/day30_smoke/best.pt`
- `outputs/diffusion/day30_smoke/last.pt`

## Next step
Day 31 will move from smoke training to fuller diffusion training with larger data coverage, longer runs, and sample-quality checks.
