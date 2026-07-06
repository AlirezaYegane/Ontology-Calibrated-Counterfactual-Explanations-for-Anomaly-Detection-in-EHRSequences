# Future Work

The following are explicitly *not* current claims; they are directions the evidence points
to.

## External validation (the main blocker)

External validation on eICU is blocked because eICU records use APACHE/body-system tokens
(`EICU_APACHE2_DX:*`, `EICU_BODYSYS:*`) that do not map to the ICD/SNOMED/RxNorm ontology
(0/500 sampled records fire any rule). Unblocking it requires **either**:

- an **APACHE→ICD/SNOMED crosswalk** so eICU tokens enter the ontology, *plus* applying the
  benchmark-v2 anomaly injectors to eICU; **or**
- another external dataset that already uses ICD/SNOMED/RxNorm-compatible coding.

This is the single most valuable next step for a generalization claim.

## Clinician validation of repairs

The counterfactual repairs are validated by an ontology scorer, not by clinicians. A study
in which clinicians rate the plausibility and usefulness of proposed edits (with
inter-rater agreement) would upgrade the repair claim from "resolves the ontology violation"
to "clinically useful."

## Broader and learned rule coverage

The three curated rule families cover three anomaly types. Extending coverage — more
sex-restricted concepts, richer medication-indication logic, additional mutual-exclusion
groups — and exploring *learned* constraints (mined from the ontology and data rather than
hand-authored) would broaden detection, especially for the noisy medication-indication
family.

## A retrained generative variant (only if it clears the gate)

The diffusion generative term was removed because it is near-random and mode-collapsed on
the current checkpoint. A generative model retrained on the benchmark-v2 clean-normal-only
split with anti-collapse measures could be re-evaluated against the same decision gate
(ROC-AUC threshold, distributional match). It re-enters the core only if it passes; nothing
in the current evidence supports it.

## Richer counterfactual edit vocabulary

Some clinically minimal repairs (e.g., correcting a recorded sex instead of removing
obstetric codes) lie outside this phase's token-edit vocabulary. Allowing demographic-field
edits and larger budgets would improve demographic-case repair.

## Temporal and physiological anomalies

The current scope is coded events. High-frequency physiological time series and temporal
ordering anomalies are a distinct, complementary problem left for future work.
