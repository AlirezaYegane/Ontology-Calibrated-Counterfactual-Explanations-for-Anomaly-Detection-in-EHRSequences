# Introduction

## EHR anomaly detection

Electronic health records accumulate coded events — diagnoses (ICD), procedures, and
medications — across an admission. These records drive downstream analytics, billing, and
research cohorts, so documentation errors and clinically inconsistent code combinations
matter. Detecting such anomalies automatically is attractive, but the problem is subtler
than generic time-series or tabular outlier detection: the "anomaly" is often not a rare
event at all.

## Why statistical rarity misses relational incoherence

Most sequence anomaly detectors work by learning what is *usual* and scoring what is
*surprising* — for example, an unsupervised next-token language model that flags
low-probability continuations. This works when anomalies are individually rare tokens. But
many clinically important inconsistencies are made entirely of common tokens whose
*combination* is wrong:

- a pregnancy or obstetric code recorded on a patient whose sex is documented as male;
- an anticoagulant or insulin present without any diagnosis that would indicate it;
- type-1 and type-2 diabetes coded on the same admission.

Every token here is common; only the *relationship* between fields is incoherent. A model
that scores next-token surprise sees nothing unusual, because each token is well-supported
by the surrounding context. Detecting these cases requires *relational* clinical reasoning,
not rarity.

## Why ontologies help

Medical ontologies (SNOMED CT for concepts, RxNorm for drugs, connected through the UMLS
Metathesaurus) encode exactly the relationships that make a combination coherent or not:
which concepts are sex-specific, which drugs require which indications, which diagnoses are
mutually exclusive. A rule engine grounded in a real ontology can therefore flag the
*joint* violation that a surprise-based detector cannot, and — crucially — can say *why* a
record is inconsistent, which is the first step toward a usable explanation.

## Why leakage and circularity matter

Synthetic anomaly benchmarks are easy to get wrong. If anomalies are injected by adding a
distinctive token, a trivial "does this token appear?" feature recovers the label, and any
model then looks good for the wrong reason. An earlier version of our own benchmark had
exactly this defect: a single token-presence feature reached ROC-AUC ≈ 0.94. Similarly, a
counterfactual explainer that peeks at *how* the anomaly was made can "repair" it perfectly
while learning nothing generalizable. Guarding against both — trivial-baseline leakage in
the benchmark, and answer-key leakage in the explainer — is a prerequisite for any credible
claim.

## What this project ultimately contributes

We set out expecting a deep detector plus a diffusion generative-surprise term to carry the
work, with the ontology as a calibration layer. The evidence redirected the project. On a
rebuilt, leakage-controlled benchmark, the **real ontology scorer is the method** — it
significantly beats the legacy baseline — while the detector is below chance and the
generative term is near-random. Rather than hide these outcomes, we report them as
first-class negative results, because they explain *why* the signal is relational. On top
of the scorer, we contribute a leakage-free, ontology-guided counterfactual repair that
proposes minimal valid edits without ever reading the answer key.

Concretely, the contributions are:

1. a leakage-controlled MIMIC-IV benchmark (**benchmark-v2**) for relational EHR anomalies;
2. a real UMLS / SNOMED CT / RxNorm ontology scoring engine;
3. evidence that real ontology scoring **significantly outperforms** legacy ICD-prefix
   rules;
4. a leakage-free, ontology-guided counterfactual repair method; and
5. transparent negative results for the detector and generative components.

All evidence is on MIMIC-IV benchmark-v2. External validation and clinical validation are
future work; we are explicit about both.
