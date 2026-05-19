# Day 46 — Paper-Ready Ablation Evidence Pack

## Status
Complete.

## Goal
Convert ablation/variant score outputs into manuscript-ready evidence:
- metrics table
- bootstrap confidence intervals
- pairwise deltas
- score distribution summaries
- figures
- conservative paper-ready interpretation

## Why this matters
The project is not only an implementation exercise. It is intended to support a scientific article on ontology-calibrated counterfactual explanations for EHR anomaly detection.

Day 46 turns raw score columns into reproducible comparative evidence that can be used in the Results and Ablation Study sections of the paper.

## Input
- `artifacts/day41/day41_variant_scores.csv`

## Main outputs
- `artifacts/day46/day46_variant_metrics.csv`
- `artifacts/day46/day46_pairwise_deltas.csv`
- `artifacts/day46/day46_score_distribution_by_label.csv`
- `artifacts/day46/day46_paper_results_table.md`
- `artifacts/day46/day46_result_interpretation.md`
- `artifacts/day46/day46_roc_auc_by_variant.png`
- `artifacts/day46/day46_average_precision_by_variant.png`
- `artifacts/day46/day46_f1_by_variant.png`
- `docs/paper/day46_ablation_results_table.md`
- `docs/paper/day46_ablation_interpretation.md`

## Scientific interpretation policy
Use conservative language:
- report which variant performs best
- report confidence intervals
- report weak components honestly
- do not overclaim the generative component if `generative_only` is weak
- treat diffusion/generative score as auxiliary unless the ablation clearly supports it

## Re-run command

```powershell
python .\scripts\build_day46_paper_evidence_pack.py `
  --scores_csv .\artifacts\day41\day41_variant_scores.csv `
  --out_dir .\artifacts\day46 `
  --n_boot 200 `
  --seed 42
Validation
python -m pytest -q .\tests\test_day46_paper_evidence_pack.py
Main scientific finding

The detector-based score remains the dominant signal in this ablation pack.

The generative_only score is non-discriminative as a standalone component:

ROC-AUC: 0.5000
AP: 0.2299
Precision: 0.2299
Recall: 1.0000

This means the generative score should be described conservatively as an auxiliary/diagnostic signal unless future generative surprise definitions improve its separation.

The detector_only and no_ontology variants have nearly identical discrimination in this artifact, which means this specific ablation pack does not yet provide strong evidence that the current ontology term improves ROC-AUC/AP over the detector-only score.
