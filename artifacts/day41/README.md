# Day 41 - Run Ablation Studies

## Status

Complete.

## Goal

Run the Day 40 ablation score artifact and convert it into paper-ready comparative evidence.

## Why this matters

Day 41 evaluates whether the ontology-calibrated configuration improves over simplified variants such as detector-only, no-ontology, no-generative, ontology-only, generative-only, and the VAE replacement slot.

## Main finding

The best ROC-AUC in the Day 41 run is achieved by the no_generative variant, with full_model_conservative extremely close behind.

This means the detector and ontology-calibrated scoring components provide the main useful signal, while the current generative Sgen component does not add clear discriminative value in this benchmark.

## Important scientific note

The Day 34 checkpoint-aligned generative evaluation showed that raw diffusion Sgen was weak as a standalone anomaly-discrimination signal. Therefore, Day 41 treats Sgen conservatively and does not overclaim the generative component.

## Main outputs

- day41_ablation_results.csv
- day41_ablation_summary.json
- day41_ablation_results.md
- day41_ablation_tables.tex
- day41_final_assessment.md

## Reproduction command

Run this command from the project root:

    python scripts/run_day41_from_day40_variants.py --input_scores artifacts/day40/variant_scores.csv --out_dir artifacts/day41

## Reporting policy

- Report no_generative as the top-ranked variant in this run.
- Report full_model_conservative as very close but not the best.
- Report the ontology/detector path as the main reliable signal.
- Do not claim that diffusion Sgen alone is effective.
- Frame this as an honest ablation finding: ontology/detector signals are useful; the current generative surprise proxy needs improvement.
