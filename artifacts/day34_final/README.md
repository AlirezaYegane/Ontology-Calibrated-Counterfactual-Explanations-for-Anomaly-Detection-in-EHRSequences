# Day 34 — Final Generative Evaluation Assessment

## Status
Complete with negative Sgen finding.

## What was fixed
The initial Day 34 evaluation was not reliable because the Day 33 checkpoint did not match the current `DiffusionModel` class.  
This was fixed by adding a legacy-compatible Day 33 diffusion wrapper and loading the checkpoint strictly.

Final checkpoint load status:

- missing parameters: 0
- unexpected parameters: 0
- checkpoint/model alignment: resolved

## Main result
After strict checkpoint alignment, the diffusion-based Sgen proxy still did not meaningfully separate injected anomalies from normal validation records.

Best timestep sweep result:

| Metric | Value |
|---|---:|
| Best timestep | 12 |
| ROC-AUC | 0.5082 |
| Average Precision | 0.5050 |
| Mean Sgen gap anomaly - normal | 0.00058 |

## Interpretation
The result is close to random. This means the current denoising-error-based Sgen definition is not strong enough as an anomaly-discrimination signal for the current injected anomaly benchmark.

This is not a code failure anymore. The checkpoint is aligned and the evaluation is now reliable. The scientific finding is that the current diffusion Sgen proxy is weak.

## Decision for Day 35
Do not use raw Sgen as the dominant calibrated anomaly score.

For Day 35:

- Use ontology score `Sont` as a primary signal.
- Use detector/supervised anomaly score as a primary statistical signal.
- Keep diffusion `Sgen` as a low-weight auxiliary or diagnostic signal.
- Do not claim that diffusion Sgen alone separates anomalies.

A conservative Day 35 calibrated score should be closer to:

`Scal = w_det * Sdet + w_ont * Sont + w_gen * Sgen`

with `w_gen` initially small.

## Evidence
- `outputs/diffusion_eval/day34_generative_legacy_aligned/summary.json`
- `outputs/diffusion_eval/day34_sgen_sweep/sgen_timestep_sweep.csv`
- `outputs/diffusion_eval/day34_sgen_sweep/sgen_timestep_sweep_summary.json`
- `artifacts/day34_final/day34_final_assessment.json`
