# Day 43 — Performance Profiling

## Status
Complete.

## Goal
Measure the runtime profile of the integrated ontology-calibrated anomaly explanation pipeline before formal evaluation and manuscript writing.

## Input Artifact
- `artifacts/day36_repair_ready/repair_ready_scores.csv`

## Profiling Scope
The profiling run covered 1,000 records and measured:

- detector score loading and ranking
- calibrated score computation
- counterfactual candidate search
- explanation generation
- vectorized vs row-wise scoring
- cached vs uncached lookup behavior

## Main Runtime Finding

| Component | Mean Total Runtime (ms) | Mean Runtime per Record (ms) | Runtime Share |
|---|---:|---:|---:|
| Detector score loading/ranking | 0.0558 | 0.000056 | 0.03% |
| Calibrated score computation | 0.1654 | 0.000165 | 0.10% |
| Counterfactual candidate search | 90.8222 | 0.090822 | 54.82% |
| Explanation generation | 74.6419 | 0.074642 | 45.05% |

## Bottleneck

The dominant bottleneck is:

`counterfactual_candidate_search`

This accounts for approximately 54.82% of the measured runtime.

The second largest component is:

`explanation_generation`

This accounts for approximately 45.05% of the measured runtime.

## Optimization Evidence

- Vectorized score computation is approximately 74.75x faster than row-wise scoring.
- Cached lookup behavior is approximately 14.29x faster than repeated uncached lookup.

## Scientific Interpretation

The core calibrated scoring step is computationally lightweight. The main runtime burden comes from downstream explanation-oriented operations, especially counterfactual candidate search and explanation generation.

This supports the paper argument that the anomaly scoring layer is efficient, while future engineering optimization should focus on caching, pruning counterfactual candidates, and rendering explanations more efficiently.

## Important Scope Note

This is artifact-level integrated pipeline profiling. It measures downstream scoring, ranking, counterfactual-search glue code, and explanation rendering from existing evaluation artifacts.

Live neural model inference latency should be reported separately if required in the final manuscript.

## Main Artifacts

- `day43_profile_summary.json`
- `day43_component_timings.csv`
- `day43_bottleneck_report.md`
- `day43_cprofile_top.txt`

## Day 43 Conclusion

Day 43 closes the performance-profiling milestone. The project now has a reproducible runtime audit that can be used in the computational-efficiency section of the paper.
