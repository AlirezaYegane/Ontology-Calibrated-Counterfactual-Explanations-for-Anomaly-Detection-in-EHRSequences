# Ethics and Data Statement

## Data access and credentialing

This work uses **MIMIC-IV** (and attempts external validation on **eICU**), both
distributed by PhysioNet under **credentialed access**. Use requires completion of CITI
"Data or Specimens Only Research" human-subjects training and a signed data use agreement
for each dataset. The datasets are de-identified at source under HIPAA Safe Harbor. No
attempt is made to re-identify individuals.

## Ontology licensing

The ontology layer uses the **UMLS Metathesaurus** (release 2026AA), **SNOMED CT** (US
edition, obtained through the UMLS/UTS affiliate license), and **RxNorm** (NLM). Use
requires a UMLS Terminology Services account and acceptance of the UMLS license; SNOMED CT
carries additional affiliate terms. These resources are used only to construct code
crosswalks and concept hierarchies for scoring.

## No raw patient data committed

No raw or processed patient-level data is committed to this repository. Benchmark splits,
derived sequences, ontology dumps, vocabularies, model checkpoints, and per-record scores
are all git-ignored. Only aggregate statistics (counts, rates, ROC-AUC, confidence
intervals) appear in tracked artifacts, from which no individual record can be reconstructed.

## Scope and intended use

The anomaly scores and counterfactual repairs are research artifacts on a **synthetic,
controlled benchmark**. We make **no** claim of clinical deployment readiness, and the
counterfactual repairs are **not** clinician-validated: "validity" denotes resolution of an
ontology rule violation with a minimal coherent edit, not endorsement of a clinical action.
The method should not be used to alter real patient records or to make clinical decisions.

## Responsible reporting

Consistent with the project's honesty constraints, negative results (the below-chance
detector, the removed generative term) are reported rather than suppressed, effect sizes are
accompanied by confidence intervals and significance tests, and the boundary between what
the code *can* do and what the evidence *supports* is stated explicitly.
