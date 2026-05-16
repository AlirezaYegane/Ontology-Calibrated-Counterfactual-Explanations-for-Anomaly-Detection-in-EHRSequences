# Day 36 — Ontology-Constrained Counterfactual Generator

## Status
Complete.

## Goal
Implement and evaluate a repair-ready counterfactual generator for ontology-driven EHR anomalies.

## Why this matters
The project is not only about detecting anomalous EHR sequences. The scientific contribution is to explain anomalies through minimal clinically meaningful edits.

Day 36 implements the first working version of this explanation layer.

## Repair-ready dataset
The original Day 35 score files contained record-level anomaly scores and `sequence_tokens`, but they did not contain direct repair targets such as `expected_code` or `bad_code`.

To avoid unsupported edits, Day 36 reconstructs repair targets from the synthetic anomaly metadata:

- `codes_original`
- `codes_corrupted`

This allows us to infer:

- codes removed during corruption -> `expected_code` candidates to add back
- codes injected during corruption -> `bad_code` candidates to remove

## Implemented components
- `src/explanations/counterfactual.py`
- `scripts/inspect_day36_repair_sources.py`
- `scripts/build_day36_repair_ready_scores.py`
- `scripts/evaluate_day36_counterfactuals.py`
- `tests/test_counterfactual_generator.py`

## Counterfactual edit space
The generator supports:

1. add expected code
2. remove bad/conflicting code
3. replace conflicting code when replacement evidence exists

## Selection objective
Candidates are selected using:

`Cost(X* | X) = ontology_violation_score(X*) + edit_penalty * number_of_edits`

This encourages sparse and interpretable explanations.

## Methodological note
The generator is intentionally conservative. It does not invent edits from anomaly labels alone. It proposes edits only when repair-ready metadata provides evidence for a missing or corrupted code.

This makes the method more defensible for a scientific paper because each proposed counterfactual can be traced to the known synthetic perturbation process.

## Expected evaluation focus for Day 37
Day 37 should formally report:

- improvement rate
- mean score reduction
- one-edit success rate
- edit-count distribution
- results by anomaly family
- failure cases, if any
