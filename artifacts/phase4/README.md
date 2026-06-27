# Phase 4 — Leakage-Free Counterfactual Repair

## Status: `phase4_success_leakage_free_repairs_supported`

The old counterfactual was **circular** (audit:
[counterfactual_audit_before.md](counterfactual_audit_before.md)): it read injection
answer keys (`bad_code` / `expected_code` / `replacement_code` / `anomaly_type`) to
undo the synthetic anomaly. Phase 4 replaces it with a generator that uses **only**
model-visible content + the real ontology + scorer feedback, and validates repairs by
re-scoring with the same independent scorer.

> Smoke-scale detector ⇒ `final_paper_evidence_claimable = false`. "Validity" means the
> ontology violation is resolved by a minimal, plausible edit — **not** clinician-
> verified clinical correctness (that is C11 / Phase 7).

## What was built
- Rewrote [src/explanations/counterfactual.py](../../src/explanations/counterfactual.py):
  `generate_counterfactual(record, scorer, ontology_index, *, detector=None,
  max_edits=3, beam_size=20, seed=42)` → `CounterfactualResult` (structured
  `CounterfactualEdit`s, before/after S_ont/S_cal/S_det, violations, validity, sparsity).
- Deterministic beam search driven by the real ontology **S_ont**; the smoke detector
  is scored before/after as a **diagnostic-only** signal that never drives repair.
- Eval harness [scripts/run_phase4_counterfactual_eval.py](../../scripts/run_phase4_counterfactual_eval.py).
- **Removed** the leaky Day-36 test + eval script (no silent leaky fallback remains).

## Repair strategies (chosen by the scorer's violation `kind`, not `anomaly_type`)
| Family | Action | Notes |
|---|---|---|
| `demographic_mismatch` | **remove** the sex-incompatible diagnosis token | high-confidence; replace-with-neighbor generated but rarely wins |
| `missing_required_code` | **remove** the unsupported medication (preferred) or **add** a curated required-context dx | add is higher clinical-risk, used when removal doesn't resolve |
| `mutual_exclusion` | **remove**/generalize ONE side, chosen by score reduction | which side is decided by the objective, never by hidden metadata |

**Objective:** minimize residual S_ont, then fewer edits, smaller ontology distance,
lower risk: `cost = S_ont + 0.05·edits + 0.02·distance + 0.10·risk`.

## Evaluation (benchmark-v2 test, 1000 anomalies)
- ontology-**flagged** anomalies: 686 (the rest, 314, are Phase 3b detection gaps — mostly
  uncovered medication anomalies — **not** Phase 4 failures).
- **repair success among flagged: 90.5%** (621/686). Overall (incl. unflagged): 62.1%.
- mean ΔS_ont **0.644**; mean ΔS_cal 0.170; mean ΔS_det 0.047 (diagnostic-only).
- edits: **median 1**, mean 1.23. (682 remove, 84 add.)

| Repair family (when flagged) | n | success | rate |
|---|---:|---:|---:|
| forbidden_cooccurrence | 323 | 323 | **1.00** |
| medication (missing_required_code) | 204 | 204 | **1.00** |
| demographic_mismatch | 200 | 135 | **0.675** |

**Demographic 67.5%** is the soft spot: gender-flip records with many obstetric codes
exceed the 3-edit budget (`no_score_reducing_edit_found`: 63). The clinically minimal
repair for a flip is correcting the sex — outside the token-edit vocabulary this phase
specifies — so per-token removal is honest-but-limited here.

## Leakage protections (the most important part)
- `generate_counterfactual` reads only `{sequence, gender, age_group}` via
  `extract_model_visible`.
- [tests/test_phase4_counterfactual_leakage.py](../../tests/test_phase4_counterfactual_leakage.py):
  adding every answer key leaves the repair **byte-identical**; **misleading** answer
  keys are ignored (the generator still targets the real ontology conflict); works with
  zero hidden metadata; outputs carry no hidden fields; static dict-access guard.

## Sanitized successful repairs
- **demographic**: `M, [O80(pregnancy), I10]` → remove `DX_10_O80`; S_ont 1.0→0.0 (1 edit).
- **forbidden**: `[E11.9 type-2, E10.9 type-1]` → remove `DX_10_E10_9`; 0.5→0.0 (1 edit).
- **medication**: `F, [INSULIN, I10]` (no diabetes) → remove `MED_INSULIN` (or add diabetes
  context); 0.5→0.0 (1 edit).

## Tests
Phase 4: 19 tests (`test_phase4_counterfactual_{leakage,generation,eval}.py`). Full
suite **236 passed**.

## Claim impact
- **C10** ontology counterfactual repair: `cut_unless_fixed → supported_now` (scoped) —
  leakage-free, ontology-grounded, minimal, independently validated.
- **C9** diffusion counterfactual: **still unsupported** — this counterfactual is
  ontology/edit-based, not diffusion (diffusion is the Phase 5 gate).

## Next
Phase 5 (generative/diffusion decision gate) **may** proceed — as a gate, not an assumed
win (`Sgen` ROC-AUC 0.475, `w_gen=0`). Full-scale detector training stays a later
experiment phase. **No H200, no large final training now.**
