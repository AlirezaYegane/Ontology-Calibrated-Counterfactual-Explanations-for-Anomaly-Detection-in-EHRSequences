# Day 42 — Milestone 3: Core System Completion

## Status

Complete with documented limitations.

## Purpose

Day 42 closes the core-system milestone by reviewing the implemented detector, diffusion/generative, ontology scoring, counterfactual, explanation, and ablation modules before moving into formal evaluation and paper-oriented polishing.

## Module Completion

| Module | Status | Missing required items |
|---|---|---|
| `data_and_ontology_pipeline` | complete | None |
| `supervised_detector` | complete | None |
| `diffusion_generative_model` | complete | None |
| `calibrated_scoring` | complete | None |
| `counterfactual_generator` | complete | None |
| `explanation_and_case_studies` | complete | None |
| `ablation_framework_and_results` | complete | None |

## Main Scientific Position

The current evidence supports the project as an ontology-calibrated anomaly explanation pipeline. The strongest empirical signal comes from the detector and ontology-calibrated scoring components.

The generative diffusion component has been implemented and evaluated, but the current raw `Sgen` proxy should be treated as an auxiliary or diagnostic signal rather than the main anomaly-discrimination signal.

## What We Can Claim

- The core pipeline is implemented end-to-end.
- The system can decompose anomaly evidence into detector/statistical and ontology-informed components.
- Counterfactual and explanation artifacts exist and can support qualitative case studies.
- Ablation evidence is available for comparing full and simplified variants.
- The current results should be presented conservatively, especially around the generative `Sgen` component.

## What We Should Not Overclaim

- Do not claim clinical deployment readiness.
- Do not claim clinician-validated explanation usefulness unless a human review is actually completed.
- Do not claim that raw diffusion `Sgen` is the main discriminative anomaly signal.
- Do not claim cross-dataset robustness until MIMIC-IV/eICU validation is run.

## Gap Register

| ID | Area | Severity | Finding | Next Action |
|---|---|---|---|---|
| G1 | Generative surprise Sgen | high_for_claims_low_for_pipeline | Current diffusion denoising-error Sgen is not discriminative enough to be claimed as the main anomaly signal. | In future work, test stronger likelihood proxies, conditional diffusion scoring, or reconstruction-calibrated generative objectives. |
| G2 | Counterfactual evaluation | medium | Counterfactual outputs exist but need broader test-set evaluation and more systematic plausibility review. | Day 45-46 should run comprehensive test evaluation and human/plausibility review. |
| G3 | Runtime profiling | medium | Core functionality is implemented, but component-level latency is not yet profiled. | Measure scoring, ontology checks, counterfactual search, and explanation generation separately. |
| G4 | Clinical validation | expected_limitation | No clinician-rated validation has been completed yet. | Prepare a curated review sheet for representative explanations. |
| G5 | Cross-dataset robustness | future_work | Current evidence is strongest on the current MIMIC-derived benchmark and synthetic anomaly setup. | Later evaluate on MIMIC-IV/eICU subsets if time allows. |

## Readiness Decision

The core system is ready to move into Weeks 7–8: profiling, interface wrapping, comprehensive test-set evaluation, plausibility review, and paper-oriented polishing.

## Next Phase

- Day 43: performance profiling
- Day 44: simple user-facing interface / CLI
- Day 45: comprehensive held-out evaluation
- Day 46: plausibility review and explanation refinement
