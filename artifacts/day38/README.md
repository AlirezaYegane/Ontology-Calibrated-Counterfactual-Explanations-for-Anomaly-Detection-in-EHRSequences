# Day 38 — Explanation Text Generator

## Status
Complete.

## Goal
Convert Day 37 counterfactual evaluation outputs into reproducible, template-based explanation text.

## Why this matters for the paper
The project is not only an anomaly detector. Its scientific contribution is an ontology-calibrated counterfactual explanation framework for EHR sequences. Day 38 adds the textual explanation layer that connects:

- calibrated anomaly score before and after counterfactual repair,
- anomaly family / violation type,
- edit count,
- concrete counterfactual edit action,
- ontology violation score,
- ontology/counterfactual evidence,
- conservative interpretation of generative surprise.

## Scientific caution
Based on the Day 34 finding, raw diffusion-based `Sgen` is not treated as the dominant anomaly evidence. In Day 38, `Sgen` is reported as a diagnostic auxiliary signal unless explicitly configured otherwise.

## Record-level fields used
The generator reads Day 37 record-level fields such as:

- `anomaly_type`
- `score_before`
- `score_after`
- `edit_count`
- `action_raw`
- `raw_edits_text`
- `raw_original_violation_score`
- `violations_before`
- `raw_counterfactual_violation_score`
- `violations_after`

When explicit edit text is available, it is used directly in the explanation. When it is missing, the generator falls back to conservative anomaly-type-specific explanation templates.

## Final real-run summary
The real Day 38 run generated 50 paper-inspection explanations from Day 37 counterfactual evaluation records.

Key results:

- sampled cases: 50
- mean ΔScal: 1.8
- median ΔScal: 2.0
- positive score reduction rate: 100%
- one-or-two-edit explanation rate: 100%
- primary driver: ontology violation signal
- anomaly families:
  - missing_diagnosis: 16
  - demographic_conflict: 30
  - medication_mismatch: 4

## Outputs
Generated outputs are stored under:

- `artifacts/day38/real/`

Main files:

- `day38_explanations.csv`
- `day38_explanations.jsonl`
- `day38_case_studies.md`
- `day38_explanation_summary.json`
- `day38_final_summary.json`

## Interpretation
A good Day 38 explanation should:

1. identify the anomaly type,
2. report the ontology/counterfactual driver,
3. state the concrete counterfactual edit,
4. report the calibrated score reduction,
5. avoid presenting model-generated edits as clinical advice.

## Next step
Day 39 should run a small set of representative records through the full end-to-end pipeline and collect polished case studies.
