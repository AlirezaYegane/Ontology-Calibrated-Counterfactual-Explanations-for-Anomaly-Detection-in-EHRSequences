# Phase 4 — Counterfactual Audit (BEFORE)

**Verdict: the old `src/explanations/counterfactual.py` is LEAKY / CIRCULAR.**
It reverses the synthetic anomaly by reading the benchmark's injection answer keys,
and never invokes the real ontology engine, scorer, or distance utilities.

## What the old code does
A self-contained rule-and-edit search over a code list + a row `Mapping`. It
"detects violations" and "proposes edits" almost entirely from **answer-key
columns** and `anomaly_type`, not from clinical reasoning over the ontology.

## Leakage findings (concrete)
| Function | Reads | Verdict |
|---|---|---|
| `expected_codes_from_row` | `expected_code(s)`, `missing_code(s)`, `removed_code(s)`, `required_code(s)`, `target_code(s)`, `indication_code(s)` | **answer-key leakage** |
| `bad_codes_from_row` | `bad_code(s)`, `injected_code(s)`, `added_code(s)`, `conflict_code(s)`, `flagged_code(s)`, `problem_code(s)` | **answer-key leakage** |
| `replacement_codes_from_row` | `replacement_code(s)`, `suggested_replacement(s)` | **answer-key leakage** |
| `anomaly_type_from_row` | `anomaly_type` / `type` / `violation_type` | **label leakage** (used as a direct rule selector, e.g. `if anomaly_type == "demographic_conflict"`) |
| `detect_violations` | `anomaly_type` + `bad_codes` + `expected_codes` | violations are *defined by* the answer key |
| `propose_one_step_edits` | removes `bad_codes`, adds `expected_codes`, replaces `bad`→`replacement` | **circular**: literally undoes the injection using the answer key |

So a "successful" repair was tautological — it copied the original normal code
back from `expected_code` / `replacement_code`, or deleted exactly the
`bad_code`/`injected_code` the injector recorded.

## Uses the real ontology? NO
- No import of `OntologyEngine`, `OntologyIndex`, `OntologyAwareScorer`,
  `src.ontology.distance`, or the Phase 3b rule packs.
- "Violations" are a bespoke `Violation` dataclass driven by row metadata, not the
  canonical `OntologyViolation` from the real engine.
- Pregnancy detection is a hand-rolled token-substring list (`PREGNANCY_PATTERNS`),
  not the ontology.

## Tests / callers depending on leaky behavior?
Two Day-36-era consumers depended on the leaky API and were **removed** (git history
preserves them) because they validate / exercise exactly the circular behavior Phase 4
eliminates:
- `tests/test_counterfactual_generator.py` — asserted that `anomaly_type` selects the
  rule, `bad_code`→remove, and `expected_code`→add (i.e. it tested leakage as a
  feature). Superseded by `tests/test_phase4_counterfactual_*`.
- `scripts/evaluate_day36_counterfactuals.py` — a legacy evaluation script that read
  answer-key columns (6 references) via the removed helpers.

No Phase-1/2/3 module imports the counterfactual module, so the rewrite is safe. There
is **no** silent leaky fallback: the leaky functions no longer exist.

## How Phase 3b violations are represented (what the new code will consume)
`OntologyAwareScorer(ontology_mode="real").score(row, s_det)` returns
`{"s_ont", "s_cal", "violations": [ {rule_id, kind, message, codes, severity} ], ...}`
where `kind ∈ {demographic_mismatch, missing_required_code, mutual_exclusion}` and
`codes` are model-visible evidence (source ICD tokens, mapped SNOMED concepts,
`RXNORM:<cui>`, or `DRUGNAME:<ingredient>`). These are produced from model-visible
content only (verified by `tests/test_phase3b_rule_leakage.py`).

## Candidate edits from real ontology neighborhoods (plan)
- Map violation evidence back to the **model-visible tokens** that produced it.
- **Remove** the implicated token (primary, minimal, high-confidence) for all three
  families.
- **Replace** an offending diagnosis with a non-restricted ontology neighbor
  (`src/ontology/distance.neighborhood`) where a safe nearby concept exists.
- **Add** a curated required-context diagnosis for medication violations (lower
  confidence; higher clinical-risk weight so the search prefers removal).
- Drive the search with the **real scorer's S_ont** (never labels); validate with the
  same independent scorer (+ smoke detector as a diagnostic-only signal).

## Conclusion
Old code = **leaky/circular, legacy-only, ontology-free**. Phase 4 rewrites it to a
leakage-free generator that consumes only model-visible content + the real ontology
+ scorer feedback.
