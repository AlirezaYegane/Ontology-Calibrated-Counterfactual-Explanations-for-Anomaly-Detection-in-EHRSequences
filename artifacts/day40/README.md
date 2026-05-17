# Day 40 — Ablation Framework

## Status
Implemented.

## Goal
Build a reproducible ablation framework for evaluating which parts of the proposed system contribute to anomaly detection and explanation quality.

## Implemented variants
- Full model, conservative weighting
- No ontology
- No generative component
- Detector only
- Ontology only
- Generative only
- VAE replacement slot

## Important note
The vae_replacement_slot is a framework slot. If no real trained VAE score is present in the input file, the script uses a clearly marked proxy only to keep the pipeline runnable. This proxy must not be reported as a trained VAE baseline in the final paper.

## Main command
Run:

    python .\scripts\run_day40_ablation_framework.py --out_dir .\artifacts\day40 --allow_vae_proxy

## Outputs
- artifacts/day40/ablation_results.csv
- artifacts/day40/variant_scores.csv
- artifacts/day40/day40_ablation_summary.json
- artifacts/day40/day40_ablation_report.md

## Next step
Day 41 should run the ablation framework on the final selected evaluation dataset and, if possible, replace the VAE proxy slot with a trained VAE reconstruction score.
