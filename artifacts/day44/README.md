# Day 44 — User-Facing Explanation Interface

## Status

Complete.

## Goal

Day 44 adds a user-facing command-line interface for inspecting individual anomaly-detection records.

The interface converts internal score artifacts into readable record-level explanations that can support paper case studies, qualitative inspection, and demo screenshots.

## Main Script

Main CLI script:

    scripts/explain_day44_record.py

Example command:

    python scripts/explain_day44_record.py --scores_csv artifacts/day36_repair_ready/repair_ready_scores.csv --row 55318

## Implemented Capabilities

- Load CSV, JSON, and JSONL score artifacts.
- List explainable records.
- Select a record by row index or record id.
- Display detector, ontology, generative, and calibrated score components when available.
- Display available counterfactual or edit evidence.
- Export explanation reports as Markdown and JSON.
- Avoid inventing unavailable evidence.

## Generated Examples

Day 44 includes two paper-facing examples.

| Example | Row | Label | Interpretation |
|---|---:|---:|---|
| High-risk synthetic anomaly | 55318 | 1 | Detector score is very high and the interface labels the case as high risk. |
| Low-risk normal record | 0 | 0 | Detector score is low and the interface labels the case as low risk. |

## Scientific Interpretation Policy

The interface reports `Sgen` as a diagnostic auxiliary signal only.

This follows the earlier project finding that the current diffusion-based generative score is not strong enough to serve as the dominant anomaly-discrimination signal.

The main explanation should therefore prioritize:

1. detector score,
2. ontology score,
3. calibrated score,
4. available counterfactual or edit evidence.

## Output Files

Generated explanations are stored as:

    artifacts/day44/day44_explanation_synthetic_anomaly.md
    artifacts/day44/day44_explanation_synthetic_anomaly.json
    artifacts/day44/day44_explanation_normal.md
    artifacts/day44/day44_explanation_normal.json

## Acceptance Checks

Run:

    python -m py_compile scripts/explain_day44_record.py
    python -m pytest tests/test_day44_explain_cli.py -q

Expected result:

    2 passed

## Files Added or Updated

- `scripts/explain_day44_record.py`
- `tests/test_day44_explain_cli.py`
- `artifacts/day44/README.md`
- `artifacts/day44/day44_explanation_synthetic_anomaly.md`
- `artifacts/day44/day44_explanation_synthetic_anomaly.json`
- `artifacts/day44/day44_explanation_normal.md`
- `artifacts/day44/day44_explanation_normal.json`

## Paper-Facing Summary

Day 44 turns the internal scoring and explanation artifacts into a reproducible record-level inspection interface.

This is important for the paper because it supports qualitative evidence, case-study examples, and transparent explanation review. The interface makes clear which evidence is available, which score components are missing, and which signals should not be over-interpreted.

The output can be used later in the paper as a compact example of how the proposed framework explains an anomalous EHR sequence at the individual-record level.
