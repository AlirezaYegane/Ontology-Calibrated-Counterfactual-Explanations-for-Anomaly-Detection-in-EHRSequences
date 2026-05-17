# Day 34 — Generative Model Evaluation

## Status
Complete.

## Goal
Evaluate the ontology-regularized diffusion model after Day 33 using generative quality and Sgen separation metrics.

## What was evaluated
- Marginal code/token distribution similarity between real and generated records
- Co-occurrence similarity over frequent token pairs
- Record-level Jaccard similarity
- Sgen separation between normal records and injected anomalies

## Main metrics

| Metric | Value |
|---|---:|
| Real records | 1024 |
| Generated records | 32 |
| Generated source | model.sample |
| Marginal L1 distance | 1.1505 |
| Marginal JS divergence | 0.2726 |
| Marginal frequency correlation | 0.6887 |
| Top-1000 token Jaccard | 0.1270 |
| Co-occurrence pair Jaccard | 0.2104 |
| Co-occurrence frequency correlation | 0.3929 |
| Paired record Jaccard mean | 0.1173 |
| Sgen method | model_midpoint_nonpad_denoising_error |
| Sgen ROC-AUC | 0.4749 |
| Sgen Average Precision | 0.4841 |
| Sgen mean gap anomaly-normal | -0.0024 |

## Interpretation
Day 34 is an evaluation checkpoint rather than a training step. The generated distribution metrics tell us whether the diffusion model produces records that resemble the real validation distribution. The Sgen metrics tell us whether the model assigns higher generative surprise to injected anomalies than to normal records.

## Important note
If `generated_source` contains `fallback_resampled_real_records_NOT_FINAL_GENERATION`, then the model sampling API was not available through the common interfaces and the distribution metrics should be treated as a pipeline smoke check, not a final generative-quality result.

If `sgen_method` contains `proxy_token_frequency_nll_NOT_FINAL_DIFFUSION_SGEN`, then the Sgen separation is only a fallback diagnostic and should be replaced by model-based diffusion Sgen before reporting.

## Artifacts
- outputs/diffusion_eval/day34_generative/summary.json
- outputs/diffusion_eval/day34_generative/metrics_table.csv
- outputs/diffusion_eval/day34_generative/top_token_frequency_comparison.csv
- outputs/diffusion_eval/day34_generative/sgen_scores.csv, if anomaly data was available
- artifacts/day34/day34_generative_eval_summary.json
- artifacts/day34/README.md

## Next step
Day 35 should integrate the diffusion Sgen score with ontology score Sont to build the calibrated anomaly score:

Scal = w1 * Sgen + w2 * Sont

## Final assessment

The Day 34 evaluation pipeline completed successfully, but the current result should be treated as a diagnostic checkpoint rather than a final generative-quality result.

Key concerns:
- The checkpoint did not load cleanly: `missing=63 unexpected=57`.
- Generated records are almost always near `max_len` (`generated_length_mean ≈ 254` vs real mean ≈ 47).
- Generated token diversity is low (`127` unique generated tokens vs `4587` real unique tokens in the evaluated subset).
- `Sgen` does not separate injected anomalies from normal records (`ROC-AUC ≈ 0.475`, below random baseline).
- The mean Sgen gap is negative, meaning injected anomalies were not assigned higher generative surprise.

Conclusion:
Day 34 is complete as an evaluation/diagnostic pass, but the model is not ready for Day 35 calibrated scoring until checkpoint loading and sampling/Sgen alignment are fixed.
