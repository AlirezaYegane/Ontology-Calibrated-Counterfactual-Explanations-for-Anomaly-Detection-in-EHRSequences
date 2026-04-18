# Day 26 - Baseline Completion

- Generated at: `2026-04-18T11:10:03+00:00`
- Status: **complete**
- Baseline ready for generative stage: **True**
- Operating profile: **high_precision_moderate_recall**

## Final Metrics

- ROC-AUC: **0.8002**
- Average Precision: **0.7332**
- F1: **0.6754**
- Precision: **0.9045**
- Recall: **0.5389**
- Threshold: **0.6343**
- Predicted positive rate: **0.1370**
- Hardest anomaly type: **missing_diagnosis**
- Easiest anomaly type: **demographic_conflict**

## Strengths

- Detector achieved ROC-AUC=0.8002 on the supervised anomaly evaluation set.
- Average Precision reached 0.7332, indicating meaningful anomaly ranking quality.
- Operational thresholding produced F1=0.6754.
- Strongest anomaly family at this stage: demographic_conflict.
- Current operating point is conservative: precision is stronger than recall.

## Limitations

- Hardest anomaly family remains: missing_diagnosis.
- Recall is still moderate, so some anomalies are likely being missed.
- This milestone is still pre-diffusion: generative surprise (Sgen) is not yet the main driver of the score.
- Baseline detector behavior is now understood well enough to serve as a stable pre-generative reference.

## Implications for Generative Stage

- Freeze this detector behavior as the baseline reference before Day 27 data preparation.
- Prioritize the hardest anomaly family during generative design and later counterfactual repair analysis.
- Carry current threshold and anomaly-type breakdown forward as the benchmark for post-diffusion comparison.
- Use Day 27 to prepare tensors/dataloaders cleanly; do not change baseline evaluation definitions during that transition.

## Supporting Evidence Paths

- `artifact_day20_summary`: `D:\Article\artifacts\day20\day20_supervised_eval_summary.json`
- `day20_eval_summary`: `None`
- `day20_breakdown_csv`: `None`
- `day25_scoring_dir`: `D:\Article\outputs\scoring\day25\run_ref_v2`

## Day 25 Scoring Files Detected

- `by_anomaly_type.csv`
- `scored_rows.csv`
- `summary.json`
- `top_examples_per_type.csv`

## Next Step

- Day 27 - prepare diffusion inputs, masks, batching, and storage artifacts.

