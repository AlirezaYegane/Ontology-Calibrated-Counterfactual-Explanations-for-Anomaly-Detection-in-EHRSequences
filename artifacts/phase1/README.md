# Phase 1 — Benchmark Validity Repair + Non-Circular Skeleton

## Phase 1a — Triviality diagnostic (run)
Script: `scripts/diagnose_anomaly_triviality.py`
Outputs: `artifacts/diagnostics/rf2_triviality.{json,md}`

**Verdict: the current synthetic benchmark is artifact-driven (severity: high) and
redesign is mandatory.** Highlights:

| Anomaly type | Best trivial signal | Discriminative power |
|---|---|---:|
| demographic_conflict | contains_pregnancy_or_sex_specific_token | **0.94** |
| medication_mismatch | sequence_length | **0.72** |
| missing_diagnosis | diagnosis_token_count | 0.56 |

The largest anomaly class (demographic_conflict) is ~0.94 recoverable by a single
label-free token indicator — i.e., the "anomaly" is essentially the presence of the
exact token family the ontology rule flags. This is circular by construction.

## Phase 1b — Non-circular skeleton (scaffolding only)
New modules:
- `src/evaluation/protocol.py` — subject-overlap guards, forbidden-column leakage
  guards, leave-anomaly-type-out splits, label-distribution summaries.
- `src/preprocessing/anomaly_injection_v2.py` — anomaly generator skeleton with strict
  separation of `model_visible` / `audit` / `hidden_eval` fields. Ontology-backed
  injectors are intentionally guarded with `NotImplementedError` until Phase 2 — no
  fake medical logic is included.

Tests: `tests/test_phase1_protocol.py`, `tests/test_anomaly_injection_v2.py` (17 pass).

## What this does NOT do
- It does not redesign the anomalies yet (that needs the real ontology from Phase 2).
- It does not modify the detector, scoring, diffusion, counterfactual, or ontology
  engine code.
- It does not fake ontology integration.

## Next step
Proceed to **Phase 2 (Real Ontology Integration)**; then implement the ontology-backed
anomaly v2 injectors on top of it and regenerate the benchmark.
