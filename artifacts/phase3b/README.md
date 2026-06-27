# Phase 3b — Ontology Rule-Coverage Authoring & Re-evaluation

## Status: `phase3b_success_rule_signal_supported`

Phase 3 found the real ontology scorer at **chance** on benchmark-v2 (ROC-AUC
0.5013) because the canonical engine's default rules were too sparse. Phase 3b
authored curated, auditable rule packs. The real ontology scorer's **standalone**
ranking rose to **ROC-AUC 0.7881 (95% CI [0.7743, 0.8016])** — clearly above
chance — and `combined_real_ontology` (0.6581) now beats `detector_only` (0.4247).

> **Read this honestly.** `combined_real_ontology` (0.658) is **below**
> `ontology_only_real` (0.788): the **smoke-scale detector** (0.425, *below* chance,
> weight 0.7) drags the calibrated combination down. So the ontology is a strong
> **standalone** ranker; the *combination* is not yet the best variant and won't be
> until a full-scale detector replaces the smoke one. Still smoke-scale ⇒
> `final_paper_evidence_claimable = false`.

## What was added
Two new modules (real mode loads them by default; legacy mode untouched):
- [src/ontology/rule_packs.py](../../src/ontology/rule_packs.py) — curated clinical
  tables (ICD families, RxCUIs, SNOMED roots) with `rationale/source/limitations`,
  the concept-group builder, and three rule types.
- [src/ontology/rule_loader.py](../../src/ontology/rule_loader.py) — binds the
  tables to a loaded `OntologyIndex` and emits an audit manifest.

| Rule | Severity | Mechanism | Precision |
|---|---:|---|---|
| `sex_restricted_concepts` | 1.0 | **Source ICD-family** (pregnancy `O*`, prostate `N40*`, …) vs model-visible gender, + clean SNOMED pregnancy subtree | high (normal FP **0.16%**, recall **100%**) |
| `medication_required_context` | 0.5 | Anticoagulant→thromboembolic/AF, levothyroxine→hypothyroid, insulin (by name; `MED_INSULIN` is unmapped)→diabetes | weak/noisy (recall ~0.50) |
| `diabetes_type_exclusion` | 0.5 | type-1 `E10*` vs type-2 `E11*` via SNOMED groups **and** source-token families | medium (recall **100%**, normal FP ~6.7%) |

### Why high precision needed engineering
Naive concept groups (all SNOMED targets of an ICD family via the crosswalk) fired
on **75% of normals**: the lossy many-to-many crosswalk maps generic codes
(GERD, hypertension, cardiac arrest) to obstetric *variants*, and diabetes-
complication codes to shared generic concepts. Fixes: (1) keyword-filter crosswalk
concepts to those whose SNOMED term names the sex/type; (2) exclude secondary
"X **in pregnancy**" / "X **due to** type N diabetes" concepts; (3) **anchor on the
source ICD token's family** (normal patients don't carry the opposite sex's
diagnosis codes). Normal FP fell 0.75 → **0.13**.

## Coverage before → after (benchmark-v2 test, n=6,307)
| class | violation rate before | after |
|---|---:|---:|
| normal (false positives) | 0.0010 | **0.1304** |
| demographic_incompatibility | 0.0147 | **1.0000** |
| medication_indication_mismatch | 0.0011 | **0.5006** |
| forbidden_cooccurrence | 0.0000 | **1.0000** |

(Diagnostic: [ontology_rule_coverage_v2_after_rules.json](ontology_rule_coverage_v2_after_rules.json).)

## Scoring before → after (test ROC-AUC with bootstrap CIs)
| variant | before | after |
|---|---:|---:|
| detector_only (smoke) | 0.4247 | 0.4247 |
| **ontology_only_real** | 0.5013 [0.500, 0.503] | **0.7881 [0.774, 0.802]** |
| combined_real_ontology | 0.4255 | 0.6581 [0.640, 0.676] |
| combined_legacy_ontology | 0.6625 | 0.6625 |

## Leakage protections
The scorer reads only `{codes, gender, age_group}`; `source_tokens` carry only
model-visible diagnosis/medication tokens. Rules never read `label`,
`anomaly_type`, `hidden_eval_metadata`, `audit_metadata`, or any repair answer key.
Guarded by [tests/test_phase3b_rule_leakage.py](../../tests/test_phase3b_rule_leakage.py)
(e.g. adding every answer-key column leaves `S_ont` unchanged).

## Tests
`tests/test_phase3b_rule_packs.py`, `test_phase3b_rule_leakage.py`,
`test_phase3b_rule_coverage.py` (23 pass); full suite **223 passed**.

## Verdict & next step
The ontology now provides a **real, statistically credible ranking signal** (CI
clears chance) with **no label leakage** — `phase3b_success_rule_signal_supported`.
Phase 4 (leakage-free counterfactual repair) may proceed. But the calibrated
**combination** cannot be the headline until the unsupervised detector is trained at
**full scale** (later experiment phase). **No H200, no large final training now.**
