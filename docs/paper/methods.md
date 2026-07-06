# Methods

## EHR sequence construction

Each MIMIC-IV admission is converted into an ordered sequence of clinical tokens —
diagnoses (ICD-9/10), procedures, and medications — together with a small set of
**model-visible demographic fields** (`gender`, `age_group`). Tokens carry a namespace
prefix so that a diagnosis, a procedure, and a medication with the same numeric code remain
distinct. Only these model-visible fields are ever exposed to the scorer or the
counterfactual generator; all other information (labels, injection metadata) is kept in
separate hidden fields.

## Benchmark-v2 design

Benchmark-v2 is a leakage-controlled anomaly benchmark. Its design principle is that **each
anomaly is a violation of a relationship between fields, while every individual
model-visible token stays common in normal records.** This removes the circularity of the
first benchmark, in which one token-presence feature recovered the label at ROC-AUC ≈ 0.94.

Three anomaly injectors implement this principle:

- **Demographic incompatibility** — flip the `gender` field on a record that already
  contains sex-specific codes (e.g., obstetric/pregnancy or prostate codes). *No token is
  injected*; only the demographic field changes, so the conflict is the (sex-specific code,
  wrong gender) pair and can only be found by joint reasoning. A token-only feature cannot
  distinguish a flipped record from its source normal — the sequence is byte-identical.
- **Medication-indication mismatch** — remove the indication diagnosis that a present drug
  requires (insulin → diabetes; anticoagulant → atrial fibrillation / thromboembolism;
  levothyroxine → hypothyroidism). Only a diagnosis count changes; the drug token stays
  common.
- **Forbidden co-occurrence** — add a curated mutually-exclusive partner that is itself
  mapped and non-rare (type-2 diabetes E11 ↔ type-1 diabetes E10). The added token is
  common, not a rare giveaway.

**Splits and leakage controls.** Splits are subject-level with zero overlap across
train/val/test; the train split is clean-normal-only (for unsupervised training). Fields
are separated into `model_visible`, `audit`, and `hidden_eval`; answer keys (original
gender, removed/added codes) live only in `hidden_eval` and never reach model-visible
output. A guard function (`validate_model_visible_fields`) raises if any answer-key column
leaks into model-visible output. A triviality diagnostic scans label-free features
(token counts, sequence length, sex-specific-token presence) and enforces a
non-circularity gate: the strongest such signal must be below 0.80. Benchmark-v2 passes at
0.6127.

Split sizes: train 20,570 (normal-only), val 3,123 (610 anomaly), test 6,307 (1,376
anomaly); test anomaly rate 21.8%.

## Real ontology mapping

Diagnosis codes are crosswalked ICD-9/10 → SNOMED CT and drug names → RxNorm ingredients
using authoritative UMLS Metathesaurus mappings (release 2026AA; SNOMED CT US edition;
RxNorm full current). A hierarchical index provides parents, children, and terms per
concept. Coverage on MIMIC-IV meets the pre-registered thresholds: diagnosis 0.80,
medication 0.78.

## Ontology rule packs

The scorer applies three auditable rule families. Each reads only model-visible clinical
content plus demographics.

- **Demographic incompatibility (sex restriction).** Anchored on the source ICD
  diagnosis-code family (pregnancy `O*`/`Z3A`, prostate `N40–N53`, etc.) and a curated
  SNOMED pregnancy subtree, checked against the model-visible `gender`. Severity 1.0.
  Precision is high (normal false-positive ≈ 0.16%).
- **Medication-indication context (required context).** Therapeutic anticoagulants →
  thromboembolic/AF context; levothyroxine → hypothyroid context; insulin (matched by
  source-token name) → diabetes context. Severity 0.5. This family is *deliberately noisy*:
  insulin is also used for ICU hyperkalaemia and prophylactic anticoagulation lacks an
  active-clot diagnosis, so recall is ≈ 0.50 by design.
- **Forbidden co-occurrence (group mutual exclusion).** Type-1 (`E10*`) vs type-2 (`E11*`)
  diabetes co-occurrence, via SNOMED concept groups and source-token ICD families.
  Severity 0.5; normal false-positive ≈ 6.7%, reflecting genuine EHR co-coding.

Concept groups are keyword-filtered so that generic codes do not cross-map into "X in
pregnancy" / "X due to type-N diabetes" variants; this cut the normal false-positive rate
from 0.75 (naive groups) to 0.13.

## Scoring

The ontology score `S_ont` aggregates rule violations (weighted by severity) into a
per-record anomaly score. The project also implements an unsupervised sequence detector
score `S_det` (a next-token language model over the clean-normal training split) and a
transparent calibrated combination:

```
S_cal = (w_det · S_det + w_ont · S_ont) / (w_det + w_ont),   with  w_gen = 0.
```

The generative-surprise weight `w_gen` is fixed at 0: the diffusion term is excluded from
the core (see Experiments / Results). Because the detector is non-additive on this
benchmark (adding it lowers ROC-AUC), the **recommended final method is ontology-only**:

```
S_main = S_ont.
```

The proposal's original additive identity `S_det = S_gen + S_ont` is *not* used; the
defensible formulation is this explicit, configurable weighted score with the generative
term removed.

## Leakage controls (scoring)

The scorer's real mode reads only `{codes, gender, age_group}`; source tokens carry only
model-visible diagnosis/medication tokens. It never reads `label`, `anomaly_type`, hidden
or audit metadata, or any injection answer key. Rule-leakage tests enforce this.

## Counterfactual generation

Given a flagged record, the generator searches for the smallest set of ontology-valid edits
that drives `S_ont` down, using only model-visible fields, the real ontology, and scorer
feedback. It reads **no** answer key (no `bad_code` / `expected_code` / `replacement_code` /
`anomaly_type`, no hidden or audit metadata); the old answer-key-driven logic was deleted
rather than retained as a fallback.

Edit operators:

- **remove** the offending code (primary; lowest clinical risk);
- **add context** — insert a curated required-context diagnosis from the rule's allowed
  group (used when removal does not resolve the violation);
- **replace / generalize** — substitute an ontology neighbor where one exists (rarely
  selected, because neighbors of a sex-restricted code stay sex-restricted).

The search is a beam search (beam size 20) minimizing a cost that prefers fewer edits,
smaller ontology distance, and lower clinical risk:
`cost = S_ont + 0.05·n_edits + 0.02·distance + 0.10·risk`.

A repair is **valid** if it reduces `S_ont` by at least `min_delta` (0.05) or resolves all
violations, introduces no new violation of higher severity than the worst original, respects
the edit budget (1–3 edits), yields a non-empty record, and — by construction — used no
hidden metadata. Validity is *ontology-violation resolution*, not clinician-verified
clinical correctness.
