# Day 39 — End-to-End Testing and Case Studies

## Status
Complete.

## Goal
Day 39 performs paper-oriented end-to-end testing of the explanation pipeline. The purpose is not to retrain any model, but to verify whether the implemented calibrated scoring, ontology checking, counterfactual repair, and explanation generation components work together coherently.

## Source artifact
The selected source artifact is:

`artifacts/day38/real/day38_explanations.csv`

This file contains real Day 38 explanation outputs, including score decomposition, compact ontology violations, compact counterfactual actions, and multiple explanation styles.

## Final result
The Day 39 run selected 10 representative case studies covering:

- demographic_conflict
- medication_mismatch
- missing_diagnosis

Summary:

- selected cases: 10
- mean ΔScal: 2.05
- median edit count: 1
- automatic coherence pass rate: 1.00

## Paper-oriented interpretation
These examples are intended for qualitative case-study analysis in the future paper/report.

The strongest defensible claims are:

- the pipeline produces readable end-to-end explanations;
- counterfactual repairs are sparse, usually one edit;
- calibrated anomaly scores decrease after repair;
- ontology-related issues are reflected in the explanation text;
- `missing_diagnosis`, the hardest anomaly family from earlier detector evaluation, is included in the representative cases.

## Conservative note on Sgen
The diffusion-based `Sgen` field is retained as an auxiliary diagnostic signal only. Earlier Day 34 evaluation showed that the current denoising-error-based Sgen proxy is weak for anomaly separation, so Day 39 does not rely on Sgen as the dominant explanation evidence.

## Outputs
- `day39_case_studies.json`
- `day39_case_studies.md`
- `day39_end_to_end_summary.json`
- `README.md`
