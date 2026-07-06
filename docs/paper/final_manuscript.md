# Ontology-Calibrated Counterfactual Explanations for Relational Anomaly Detection in EHR Sequences

**Author:** Alireza Yegane
**Supervisor:** Professor Xuyun Zhang
**Affiliation:** Macquarie University, Faculty of Science and Engineering

> This is the combined, self-contained manuscript. The individual section files
> (`abstract.md`, `introduction.md`, `methods.md`, `experiments.md`, `results.md`,
> `discussion.md`, `limitations.md`, `reproducibility_statement.md`,
> `ethics_and_data_statement.md`, `future_work.md`) hold the same content in modular form.
> All evidence is on MIMIC-IV **benchmark-v2**. The old circular benchmark is not used as
> final evidence. Aggregate sources are under `artifacts/phase7/`.

---

## Abstract

Electronic health record (EHR) sequences contain documentation errors and clinically
incoherent combinations of codes that are hard to surface and harder to explain. Purely
statistical rarity is a poor detector for these cases, because the offending codes are
individually common; the incoherence lives in the *relationship* between fields (for
example, a sex-specific diagnosis paired with the wrong recorded sex, or a medication
without its required indication). We study whether a real medical ontology can both flag
such relational anomalies and explain them with minimal, coherent repairs.

We first build **benchmark-v2**, a leakage-controlled anomaly benchmark on MIMIC-IV in
which every anomaly is a relational violation while every individual model-visible token
stays common. Subject-level splits have zero overlap, model-visible fields are separated
from answer keys, and the strongest label-free trivial signal is 0.61 (below the 0.80
non-circularity gate), in contrast to an earlier circular benchmark where a single
token-presence feature recovered the label at 0.94.

On this benchmark, a **real UMLS / SNOMED CT / RxNorm ontology scorer** with three
auditable rule families reaches ROC-AUC 0.7881 (95% CI [0.774, 0.802]), significantly
above a legacy ICD-prefix baseline (0.7358; paired bootstrap +0.052, p ≈ 0). We report two
transparent negative results: a full-scale unsupervised sequence detector is *below chance*
(0.4525) and, when combined with the ontology score, significantly *lowers* ROC-AUC
(−0.085, p ≈ 0); and a diffusion-based generative-surprise term is near-random and
mode-collapsed and is removed from the core method. The recommended method is therefore
**ontology-only**.

Finally, we present a **leakage-free, ontology-guided counterfactual repair** that never
reads the corruption answer key: it proposes minimal ontology-valid edits validated by an
independent scorer, achieving 89.99% valid repair among ontology-flagged anomalies with a
median of one edit. All evidence is on MIMIC-IV benchmark-v2; external validation on eICU
is blocked by a token-schema mismatch and is left as future work.

---

## 1. Introduction

### 1.1 EHR anomaly detection

Electronic health records accumulate coded events — diagnoses (ICD), procedures, and
medications — across an admission. These records drive downstream analytics, billing, and
research cohorts, so documentation errors and clinically inconsistent code combinations
matter. Detecting such anomalies automatically is attractive, but the problem is subtler
than generic time-series or tabular outlier detection: the "anomaly" is often not a rare
event at all.

### 1.2 Why statistical rarity misses relational incoherence

Most sequence anomaly detectors learn what is *usual* and score what is *surprising* — for
example, an unsupervised next-token model that flags low-probability continuations. This
works when anomalies are individually rare tokens. But many clinically important
inconsistencies are made of common tokens whose *combination* is wrong: a pregnancy code on
a patient documented as male; an anticoagulant or insulin with no indicating diagnosis;
type-1 and type-2 diabetes on the same admission. Every token is common; only the
relationship is incoherent, and a surprise-based detector sees nothing unusual.

### 1.3 Why ontologies help

Medical ontologies (SNOMED CT for concepts, RxNorm for drugs, connected through the UMLS
Metathesaurus) encode exactly the relationships that make a combination coherent: which
concepts are sex-specific, which drugs require which indications, which diagnoses are
mutually exclusive. A rule engine grounded in a real ontology can flag the joint violation
a surprise-based detector cannot, and can say *why* a record is inconsistent.

### 1.4 Why leakage and circularity matter

Synthetic anomaly benchmarks are easy to get wrong: if anomalies are injected by adding a
distinctive token, a trivial token-presence feature recovers the label and any model looks
good for the wrong reason. An earlier version of our own benchmark had this defect (a single
feature reached ≈ 0.94). Likewise, a counterfactual explainer that peeks at how an anomaly
was made can "repair" it while learning nothing. Guarding against both is a prerequisite for
credible claims.

### 1.5 Contributions

1. a leakage-controlled MIMIC-IV benchmark (**benchmark-v2**) for relational EHR anomalies;
2. a real UMLS / SNOMED CT / RxNorm ontology scoring engine;
3. evidence that real ontology scoring **significantly outperforms** legacy ICD-prefix
   rules;
4. a leakage-free, ontology-guided counterfactual repair method; and
5. transparent negative results for the detector and generative components.

We do **not** claim external validation, clinical deployment, clinician-validated repairs,
state-of-the-art deep anomaly detection, or any benefit from the detector or the generative
term.

---

## 2. Methods

### 2.1 EHR sequence construction

Each MIMIC-IV admission becomes an ordered sequence of namespaced clinical tokens
(diagnoses, procedures, medications) plus model-visible demographics (`gender`,
`age_group`). Only these model-visible fields are exposed to the scorer and the
counterfactual generator; labels and injection metadata live in separate hidden fields.

### 2.2 Benchmark-v2 design

Each anomaly is a violation of a relationship between fields, while every individual
model-visible token stays common. Three injectors implement this:

- **Demographic incompatibility** — flip `gender` on a record that already contains
  sex-specific codes; *no token injected*, so the conflict is the (code, gender) pair.
- **Medication-indication mismatch** — remove the indication diagnosis a present drug
  requires (insulin→diabetes, anticoagulant→AF/thrombosis, levothyroxine→hypothyroid).
- **Forbidden co-occurrence** — add a curated mutually-exclusive, non-rare partner
  (type-2 E11 ↔ type-1 E10).

Splits are subject-level with zero overlap; the train split is clean-normal-only. Fields
are separated into `model_visible` / `audit` / `hidden_eval`, answer keys live only in
`hidden_eval`, and a guard raises if any answer-key column leaks. A triviality diagnostic
enforces a non-circularity gate (strongest label-free signal < 0.80); benchmark-v2 passes at
0.6127. Split sizes: train 20,570 (normal-only), val 3,123 (610 anomaly), test 6,307
(1,376 anomaly); test anomaly rate 21.8%.

### 2.3 Real ontology mapping

ICD-9/10 → SNOMED CT and drug → RxNorm crosswalks come from authoritative UMLS mappings
(release 2026AA; SNOMED CT US; RxNorm current). MIMIC-IV coverage: diagnosis 0.80,
medication 0.78.

### 2.4 Ontology rule packs

Three auditable families, each reading only model-visible content + demographics:
**sex restriction** (severity 1.0; normal FP ≈ 0.16%), **medication required context**
(severity 0.5; recall ≈ 0.50 by design, since the semantics are genuinely ambiguous), and
**diabetes-type mutual exclusion** (severity 0.5; normal FP ≈ 6.7%). Keyword-filtering of
concept groups cut the normal false-positive rate from 0.75 to 0.13.

### 2.5 Scoring

```
S_cal = (w_det · S_det + w_ont · S_ont) / (w_det + w_ont),   w_gen = 0.
```

The generative term is excluded (`w_gen = 0`). Because the detector is non-additive on this
benchmark, the recommended final method is **ontology-only**, `S_main = S_ont`. The
proposal's additive identity `S_det = S_gen + S_ont` is not used.

### 2.6 Leakage controls (scoring)

The real-mode scorer reads only `{codes, gender, age_group}` and never reads labels,
`anomaly_type`, hidden/audit metadata, or injection answer keys; rule-leakage tests enforce
this.

### 2.7 Counterfactual generation

Given a flagged record, a beam search (beam size 20) finds the smallest set of
ontology-valid edits that lowers `S_ont`, using only model-visible fields, the ontology, and
scorer feedback — never the answer key (leaky logic was deleted, not retained). Operators:
**remove** (primary), **add context** (curated required-context diagnosis), and
**replace/generalize** (ontology neighbor). Cost
`= S_ont + 0.05·n_edits + 0.02·distance + 0.10·risk`. A repair is **valid** if it reduces
`S_ont` by ≥ 0.05 or resolves all violations, adds no higher-severity violation, respects a
1–3 edit budget, yields a non-empty record, and used no hidden metadata. Validity is
ontology-violation resolution, not clinician-verified correctness.

---

## 3. Experiments

All experiments use benchmark-v2 (MIMIC-IV). Threshold calibration is validation-only
(best-F1 on val, applied unchanged to test; no test tuning). Confidence intervals are
bootstrap; variant comparisons use a paired bootstrap on ROC-AUC. Variants:
`ontology_only_real`, `legacy_baseline`, `detector_only_full`,
`combined_real_without_sgen`. The detector is trained on the clean-normal-only split
(20,570 records, vocab 17,867) with early stopping, deterministic seeds, and resumable
checkpoints (full run: 25 epochs, best val ROC-AUC 0.4698 at epoch 19). Ablations cover
rule families, score components, and counterfactual edit strategies. External validation is
attempted on eICU as a schema-compatibility check.

---

## 4. Results

### 4.1 Main results (benchmark-v2 test, n = 6,307)

| Variant | ROC-AUC | 95% CI | AP | F1 |
|---|---:|---|---:|---:|
| **ontology_only_real** (main) | **0.7881** | [0.7743, 0.8015] | 0.5422 | 0.6349 |
| legacy_baseline | 0.7358 | [0.7197, 0.7511] | 0.5429 | 0.5928 |
| detector_only_full | 0.4525 | [0.4357, 0.4702] | 0.1904 | 0.3608 |
| combined_real_without_sgen | 0.7036 | [0.6866, 0.7201] | 0.4039 | 0.4596 |
| Sgen (diagnostic, excluded) | 0.4868 | — | — | — |

### 4.2 Statistical tests (paired bootstrap, ROC-AUC)

| Comparison | Δ | 95% CI | p | Sig. |
|---|---:|---|---:|---|
| ontology_only_real − legacy | +0.0524 | [0.0325, 0.0718] | ≈ 0 | yes |
| ontology_only_real − detector | +0.3356 | [0.3141, 0.3581] | ≈ 0 | yes |
| combined − ontology_only_real | −0.0845 | [−0.0963, −0.0732] | ≈ 0 | yes |
| combined − detector | +0.2511 | [0.2364, 0.2665] | ≈ 0 | yes |
| combined − legacy | −0.0322 | [−0.0567, −0.0103] | 0.004 | yes |

Real ontology significantly beats legacy (+0.052); adding the detector significantly hurts
(−0.085).

### 4.3 Ablations

Rule-family ROC-AUC: full 0.7881, legacy 0.7358, demographic-only 0.5988, medication-only
0.5711, forbidden-only 0.6252, disabled = chance. No single family reaches 0.7881 — the
three synergize. Score components: `S_ont` (0.7881) > `S_ont + S_det` (0.7036) > `S_det`
(0.4525); `S_ont + S_det + S_gen` excluded (`w_gen = 0`).

### 4.4 Counterfactual repair

| Metric | Value |
|---|---:|
| Attempted | 1,376 |
| Ontology-flagged | 939 |
| Success among flagged | 0.8999 |
| Success overall | 0.6141 |
| Mean ΔS_ont | 0.644 |
| Median edits | 1 |

Per rule type: missing-context 100%, mutual-exclusion 100%, demographic 65.7%
(edit-budget-limited). The 437 non-flagged anomalies are detection gaps, not repair
failures.

### 4.5 External validation

`external_validation_blocked_schema_mismatch`: 0/500 eICU records map to the ontology
(APACHE/body-system tokens). Documented as future work.

---

## 5. Discussion

The real ontology rule engine carries the discriminative signal (0.7881, significantly above
legacy), and the rule-family ablation shows the improvement is concentrated where a real
concept hierarchy beats flat prefix matching (forbidden co-occurrence: 0.93 vs 0.42). The
full-scale detector is below chance because the anomalies are relational and carry little
next-token surprise; mixing this near-noise signal into the score significantly lowers
ROC-AUC, which is why the final method is ontology-only. The generative term is near-random,
mis-oriented, and mode-collapsed, so it is removed from the core. Counterfactual repair is
100% for cleanly-flagged families with a median of one edit; the overall 61.4% is bounded by
detection coverage, not by the repair search. The two negative results are not gaps — they
explain the positive one: the structure of these anomalies is relational and ontological,
not statistical.

---

## 6. Limitations

MIMIC-IV only; synthetic constructed anomalies (not clinician-confirmed real errors); no
clinician validation of repairs; no external validation (eICU schema mismatch); curated
(not learned) rule packs with scoped coverage and an intentionally noisy medication family;
detector/generative components are negative results; a small counterfactual edit vocabulary
limits demographic-case repair; and exact reproduction requires the restricted data and
ontologies, which cannot be redistributed.

---

## 7. Reproducibility Statement

All source code and aggregate artifacts are included; restricted data, ontology dumps,
checkpoints, vocabularies, and per-record scores are git-ignored. `python -m pytest` runs on
CPU without the restricted data. Credentialed users can rebuild end-to-end following
`docs/reproducibility/runbook.md`. Random operations use fixed seeds; bootstrap CIs are
reported for headline numbers. Restricted data and ontology dumps are excluded for licensing
and privacy reasons, not omission.

---

## 8. Ethics and Data Statement

MIMIC-IV and eICU are used under PhysioNet credentialed access (CITI training + DUA); the
data are de-identified and no re-identification is attempted. The ontology layer uses UMLS /
SNOMED CT / RxNorm under the UMLS license. No raw patient data is committed. The method is a
research artifact on a synthetic benchmark: no clinical deployment claim, and repairs are not
clinician-validated. Negative results are reported rather than suppressed.

---

## 9. Future Work

The primary blocker is external validation: eICU needs an APACHE→ICD/SNOMED crosswalk (plus
re-injecting the benchmark anomalies), or another ICD/SNOMED/RxNorm-compatible external
dataset. Further directions: a clinician study of repair usefulness; broader/learned rule
coverage; a retrained anti-collapse generative variant (only if it clears the decision
gate); a richer counterfactual edit vocabulary; and temporal/physiological anomaly types.

---

## References

See [`references.md`](references.md) for the reference list (prior work drawn from the
project literature survey; data/ontology resources) and
[`references_todo.md`](references_todo.md) for bibliographic details still pending
verification. Consistent with the project's honesty constraints, DOIs and full author/year
strings are not fabricated.
