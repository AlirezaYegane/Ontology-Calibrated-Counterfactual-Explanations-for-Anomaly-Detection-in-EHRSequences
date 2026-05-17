from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = json.loads(summary_path.read_text(encoding="utf-8"))

    dist = s.get("distribution_metrics", {})
    cooc = s.get("cooccurrence_metrics", {})
    rec = s.get("record_jaccard_metrics", {})
    sgen = s.get("sgen_separation") or {}

    readme = f"""# Day 34 — Generative Model Evaluation

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
| Real records | {s.get("num_real_records")} |
| Generated records | {s.get("num_generated_records")} |
| Generated source | {s.get("generated_source")} |
| Marginal L1 distance | {fmt(dist.get("marginal_l1_distance"))} |
| Marginal JS divergence | {fmt(dist.get("marginal_js_divergence"))} |
| Marginal frequency correlation | {fmt(dist.get("marginal_frequency_correlation"))} |
| Top-1000 token Jaccard | {fmt(dist.get("top_1000_token_jaccard"))} |
| Co-occurrence pair Jaccard | {fmt(cooc.get("pair_jaccard"))} |
| Co-occurrence frequency correlation | {fmt(cooc.get("pair_frequency_correlation"))} |
| Paired record Jaccard mean | {fmt(rec.get("paired_record_jaccard_mean"))} |
| Sgen method | {s.get("sgen_method") or "n/a"} |
| Sgen ROC-AUC | {fmt(sgen.get("roc_auc"))} |
| Sgen Average Precision | {fmt(sgen.get("average_precision"))} |
| Sgen mean gap anomaly-normal | {fmt(sgen.get("mean_gap_anomaly_minus_normal"))} |

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
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "day34_generative_eval_summary.json").write_text(
        json.dumps(s, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(readme)


if __name__ == "__main__":
    main()
