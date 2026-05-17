# Day 43 — Performance Profiling Report

## Status

Complete.

## Goal

Measure the runtime profile of the integrated anomaly-explanation pipeline and identify bottlenecks before formal evaluation and paper writing.

## Profiling Scope

- Input artifact: `artifacts/day36_repair_ready/repair_ready_scores.csv`
- Records profiled: **1000**
- Repeat count per stage: **5**
- Profiling mode: `artifact-level integrated pipeline profiling`

## Detected Columns

- `label`: `label`
- `detector`: `Sdet_raw`
- `sgen`: `Sgen`
- `sont`: `Sont_curated_raw`
- `scal`: `Scal_curated`
- `variant`: `None`
- `anomaly_type`: `anomaly_type`

## Component Runtime

| Stage | Mean total ms | Mean ms / record | Records / sec |
|---|---:|---:|---:|
| detector_score_loading_and_ranking | 0.0558 | 0.000056 | 17908310.25 |
| calibrated_score_computation | 0.1654 | 0.000165 | 6045949.05 |
| counterfactual_candidate_search | 90.8222 | 0.090822 | 11010.52 |
| explanation_generation | 74.6419 | 0.074642 | 13397.31 |

## Bottleneck Ranking

1. `counterfactual_candidate_search` — 54.82% of measured runtime
2. `explanation_generation` — 45.05% of measured runtime
3. `calibrated_score_computation` — 0.10% of measured runtime
4. `detector_score_loading_and_ranking` — 0.03% of measured runtime

## Optimization Findings

- Vectorized score computation speedup over row-wise scoring: **74.75x**
- Cached lookup speedup over repeated uncached lookup: **14.29x**

## Interpretation

The profiling result should be interpreted as an engineering runtime audit of the current integrated evaluation artifacts. If this report is generated from score/case-study artifacts rather than live GPU inference, it measures downstream scoring, ranking, counterfactual-search glue code, and explanation rendering rather than raw neural model training time.

## Paper-Ready Note

For the manuscript, this artifact supports the computational-efficiency paragraph: the pipeline should report latency per component, identify the dominant stage, and describe practical optimizations such as vectorized scoring and cached ontology lookups.

## Generated Files

- `summary_json`: `artifacts/day43/day43_profile_summary.json`
- `component_timings_csv`: `artifacts/day43/day43_component_timings.csv`
- `bottleneck_report_md`: `artifacts/day43/day43_bottleneck_report.md`
- `raw_cprofile`: `artifacts/day43/day43_raw_profile.prof`
- `cprofile_top_txt`: `artifacts/day43/day43_cprofile_top.txt`

