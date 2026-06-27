# Phase 4 — Counterfactual Repair Status

**Status: `phase4_success_leakage_free_repairs_supported`.** The counterfactual system
was rewritten from a circular, answer-key-reading module into a leakage-free,
ontology-grounded repair generator, validated by an independent scorer on the
non-circular benchmark-v2. Smoke-scale ⇒ `final_paper_evidence_claimable = false`.

## Old code: leaky/circular (now removed)
The previous `src/explanations/counterfactual.py` read `bad_code` / `expected_code` /
`replacement_code` / `anomaly_type` and literally reversed the injection. It never
touched the real ontology. Audit: `artifacts/phase4/counterfactual_audit_before.md`.
The two leaky consumers (`tests/test_counterfactual_generator.py`,
`scripts/evaluate_day36_counterfactuals.py`) were **deleted** — there is no silent
leaky fallback.

## New design (leakage-free)
- **Inputs (only):** model-visible token sequence, gender, age_group; the real
  `OntologyAwareScorer(mode="real")` violations + `S_ont`; ontology neighborhoods /
  distance; the unsupervised detector as a **diagnostic-only** signal.
- **Generation:** deterministic beam search. Candidate edits are derived from the
  scorer's violation `kind` + evidence (mapped back to model-visible tokens), never
  from `anomaly_type` or answer keys.
- **Validation:** re-score with the same independent scorer; a repair is *valid* only
  if it meaningfully reduces `S_ont` (or resolves all violations), introduces no new
  higher-severity violation, respects the edit budget, and leaves a non-empty record.

## Results (benchmark-v2 test, 1000 anomalies)
| Metric | Value |
|---|---|
| ontology-flagged anomalies | 686 (314 unflagged = Phase 3b detection gaps) |
| **repair success among flagged** | **90.5%** (621/686) |
| repair success overall | 62.1% |
| mean ΔS_ont | 0.644 |
| mean ΔS_cal / ΔS_det | 0.170 / 0.047 (S_det diagnostic-only) |
| edits (median / mean) | 1 / 1.23 |

Per family (when flagged): forbidden **100%**, medication **100%**, demographic
**67.5%**. Demographic is budget-limited for heavily-obstetric gender-flips (the minimal
clinical repair, correcting the sex, is outside the specified token-edit vocabulary).

## Leakage protections
`extract_model_visible` is the only place the record is read (allowlist of
sequence/gender/age). `tests/test_phase4_counterfactual_leakage.py` proves: adding any
answer key leaves the repair identical; misleading keys are ignored; the generator works
with no hidden metadata; outputs contain no hidden fields.

## Failures remaining
- Demographic gender-flips with many obstetric codes exceed the 3-edit budget.
- 314/1000 anomalies are not ontology-flagged (Phase 3b coverage gap) → unrepairable.
- Validity = ontology-violation resolution, not clinician-verified correctness (C11).

## Claim impact
- **C10** ontology counterfactual repair → `supported_now` (scoped): leakage-free,
  ontology-grounded, minimal, independently validated.
- **C9** diffusion counterfactual → still unsupported (this is ontology/edit-based, not
  diffusion; diffusion is the Phase 5 gate).

## Next
Phase 5 generative/diffusion **decision gate** may proceed (a gate, not an assumed win;
`Sgen` ROC-AUC 0.475, `w_gen=0`). Full-scale detector training remains a later
experiment phase. No H200.
