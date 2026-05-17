# Day 42 — Next Phase Plan

## Immediate transition

The project should now move from core implementation to evaluation hardening.

## Day 43
- Profile runtime for detector scoring, ontology scoring, calibrated scoring, counterfactual search, and explanation generation.
- Produce component-level latency table.

## Day 44
- Build a CLI or notebook interface that runs the full pipeline on selected records.
- Output readable explanations and counterfactual edits.

## Day 45
- Run full evaluation on the held-out benchmark.
- Report ROC-AUC, AP, F1, threshold behavior, and anomaly-type breakdown.

## Day 46
- Curate representative explanations for plausibility review.
- Refine templates and identify misleading cases.

## Paper strategy
- Lead with ontology-calibrated scoring and counterfactual explanation.
- Present diffusion/Sgen honestly as implemented but currently weak as a standalone discriminative signal.
- Use ablation results to motivate a conservative final configuration.
