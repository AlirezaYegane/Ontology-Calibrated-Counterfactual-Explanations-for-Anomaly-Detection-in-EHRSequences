# Day 49 — Executive Summary and Background Draft

> **Status:** First draft — conservative, evidence-grounded.
> All citations marked `[CITATION NEEDED]` require replacement
> with verified references before submission.

---

## Executive Summary

Electronic Health Records (EHRs) are the primary medium through
which clinical observations, diagnoses, medications, and procedures
are documented in modern hospital information systems.  Despite
their centrality, EHR data are known to contain noise, omissions,
and inconsistencies that can compromise downstream clinical
decision support, research cohort selection, and patient safety
monitoring [CITATION NEEDED].

This work presents an **ontology-calibrated framework for anomaly
detection and counterfactual explanation in sequential EHR data**.
The framework decomposes abnormality assessment into three
complementary signals: (i) a supervised statistical anomaly score,
(ii) an ontology-violation score grounded in medical knowledge
graphs (SNOMED CT, RxNorm, UMLS), and (iii) a generative
plausibility score derived from a denoising diffusion model.
Where an anomaly is detected, the system proposes minimal
ontology-constrained counterfactual repairs — the smallest set of
clinically plausible changes that would render the record
non-anomalous.

Preliminary evaluation on synthetic injected anomalies indicates
that the supervised detector and ontology-calibrated scoring
components provide useful discriminative signal, while the
current diffusion-based generative proxy (Sgen) yields
near-random separation and should be regarded as auxiliary
pending methodological improvement.  The principal contribution
is the integration of clinical ontology constraints into both
the scoring and explanation stages of the anomaly-detection
pipeline.

---

## 1. Introduction

### 1.1 Clinical Motivation

Hospitals generate vast quantities of structured and semi-structured
clinical data through EHR systems.  These records — comprising
diagnoses (ICD-9/10), medications (RxNorm/NDC), procedures (CPT),
laboratory results, and demographic attributes — are used not only
for direct patient care but also for retrospective research,
quality improvement, and regulatory reporting [CITATION NEEDED].
Data quality failures in EHRs, including missing diagnoses,
implausible medication combinations, and demographic-clinical
conflicts, are well documented in the medical informatics
literature [CITATION NEEDED] and can propagate silently into
clinical decision-support systems, predictive models, and
real-world evidence studies.

Intensive Care Unit (ICU) settings amplify these risks.  The high
acuity, rapid temporal dynamics, and multi-provider documentation
patterns characteristic of critical care produce EHR sequences
that are especially susceptible to transcription errors, omitted
entries, and delayed coding [CITATION NEEDED].  Identifying and
explaining anomalous records within such sequences is therefore a
pressing clinical informatics challenge.

### 1.2 Problem Statement

Existing anomaly-detection approaches for EHR data typically
operate on flat, tabular representations and rely on purely
statistical deviation criteria [CITATION NEEDED].  These methods
suffer from two interconnected limitations:

1. **Lack of clinical grounding.**  Statistical outliers are not
   necessarily clinically implausible, and vice versa.  A rare
   but valid diagnosis–medication pair may be flagged as
   anomalous, while a common but ontologically incoherent
   combination may pass undetected.
2. **Absence of actionable explanations.**  When an anomaly is
   detected, clinicians and data stewards require not merely a
   flag but a *minimal, clinically plausible correction* — a
   counterfactual explanation — to guide review and remediation.

### 1.3 Aim and Objectives

The aim of this work is to develop and evaluate a framework that:

- Detects anomalies in sequential EHR data by combining
  statistical, ontological, and generative signals.
- Grounds anomaly assessment in established medical ontologies
  (SNOMED CT, RxNorm, UMLS) to distinguish statistical
  deviation from clinical implausibility.
- Generates minimal ontology-constrained counterfactual
  explanations that propose the smallest clinically coherent
  repair for each detected anomaly.
- Provides reproducible, auditable, and transparent outputs
  suitable for clinical informatics research.

### 1.4 Proposed Framework Overview

The proposed framework, referred to as **OntoCF-AD**
(*Ontology-Calibrated Counterfactual Anomaly Detection*),
consists of four integrated components:

1. **Supervised Anomaly Detector** — a sequence-aware neural
   classifier trained to discriminate normal from anomalous EHR
   records using learned clinical-code embeddings.
2. **Ontology-Violation Scorer (S_ont)** — a knowledge-graph
   traversal module that quantifies the degree to which a record
   violates expected ontological relationships (e.g., diagnosis–
   medication coherence, hierarchical code validity).
3. **Generative Plausibility Scorer (S_gen)** — a denoising
   diffusion model trained on normal EHR sequences; the
   reconstruction error at selected diffusion timesteps provides
   a proxy for distributional plausibility.
4. **Counterfactual Explanation Engine** — a constrained search
   procedure that identifies the minimal ontology-consistent
   perturbation to a flagged record such that the composite
   anomaly score falls below the detection threshold.

These four components are combined through a calibrated scoring
function that weights each signal according to its empirically
validated discriminative contribution.

### 1.5 Contributions

The principal contributions of this work are:

1. **Ontology-calibrated anomaly scoring** — a multi-signal
   decomposition that integrates statistical detection, ontology
   violation, and generative plausibility into a unified
   framework grounded in medical knowledge graphs.
2. **Constrained counterfactual explanations** — an explanation
   method that proposes minimal, ontology-consistent repairs
   rather than unconstrained perturbations, improving clinical
   interpretability and safety.
3. **Transparent reporting of negative findings** — an honest
   assessment of the current diffusion-based Sgen component,
   which does not yet provide strong standalone discriminative
   signal, reported alongside the stronger detector and ontology
   evidence.
4. **Reproducibility infrastructure** — a fully auditable
   codebase with day-level artefact tracking, environment
   capture, and private-data boundary documentation.

---

## 2. Background and Related Work

### 2.1 EHR Representation Learning

Clinical codes — ICD-9/10 diagnoses, RxNorm medications, CPT
procedures — are high-dimensional, sparse, and semantically
structured.  Representation learning for EHR data seeks to embed
these codes into dense vector spaces that capture clinical
similarity and temporal dynamics [CITATION NEEDED].

Early approaches adapted word-embedding techniques (e.g.,
word2vec, GloVe) to medical-code sequences, treating patient
timelines as analogues of natural-language sentences
[CITATION NEEDED].  More recent work has explored recurrent,
attention-based, and transformer architectures for sequential
clinical modelling [CITATION NEEDED].  However, most EHR
representation methods do not explicitly encode ontological
relationships between codes, relying instead on co-occurrence
statistics to learn implicit structure.

### 2.2 Anomaly Detection in Clinical Sequences

Anomaly detection in structured clinical data has been approached
through reconstruction-based autoencoders [CITATION NEEDED],
isolation forests adapted to medical feature spaces
[CITATION NEEDED], and supervised classifiers trained on
synthetically injected or expert-annotated anomalies
[CITATION NEEDED].

A persistent challenge is the scarcity of labelled anomalous
records: true clinical data-quality errors are often unknown or
ambiguous, motivating the use of synthetic anomaly injection
strategies [CITATION NEEDED].  The present work adopts a
supervised detector trained on synthetically injected anomalies,
complemented by ontology-based signals that do not require
anomaly labels.

### 2.3 Clinical Ontologies and Knowledge Graphs

Medical ontologies such as SNOMED CT [CITATION NEEDED],
RxNorm [CITATION NEEDED], the Unified Medical Language System
(UMLS) [CITATION NEEDED], and the OMOP Common Data Model
[CITATION NEEDED] provide hierarchical and relational structures
that encode clinical knowledge.  These resources enable semantic
reasoning about diagnosis–medication relationships, code
hierarchies, and therapeutic appropriateness.

While ontologies are widely used for data harmonisation and
cohort definition, their integration into anomaly-detection
pipelines as a *scoring signal* — rather than merely a
preprocessing step — remains underexplored [CITATION NEEDED].
This work uses ontology-graph traversal to compute a violation
score that quantifies how far a given record departs from
expected ontological patterns.

### 2.4 Generative Models for EHR Data

Generative models for clinical data have attracted growing
interest for tasks including synthetic data generation, missing
data imputation, and distributional novelty detection
[CITATION NEEDED].  Variational autoencoders (VAEs) and
generative adversarial networks (GANs) have been applied to
tabular and sequential EHR data [CITATION NEEDED], and more
recently, denoising diffusion probabilistic models (DDPMs) have
been explored for structured clinical sequence generation
[CITATION NEEDED].

In the present framework, a DDPM trained on normal EHR sequences
is used to compute a generative plausibility score (Sgen) based
on denoising error at selected timesteps.  However, as reported
transparently in the experimental evaluation, the current Sgen
proxy yields near-random discriminative performance (ROC-AUC ≈
0.508 at the best timestep) and should be regarded as an
auxiliary or diagnostic signal pending further methodological
refinement.  This negative finding is included in the interest
of scientific transparency.

### 2.5 Counterfactual Explanations in Healthcare AI

Counterfactual explanations answer the question: *"What is the
smallest change to this input that would alter the model's
output?"* [CITATION NEEDED].  In the healthcare domain,
counterfactual methods have been applied to tabular patient data
[CITATION NEEDED], clinical text [CITATION NEEDED], and medical
imaging [CITATION NEEDED].

A critical limitation of unconstrained counterfactual methods is
that the proposed perturbations may be clinically implausible —
for example, suggesting a biologically impossible medication
substitution or an ontologically incoherent diagnosis change
[CITATION NEEDED].  The present work addresses this by
constraining the counterfactual search to ontology-consistent
perturbations, ensuring that proposed repairs respect the
hierarchical and relational structure of medical knowledge
graphs.

### 2.6 Research Gap

Despite advances in each of the above areas, several gaps
remain at their intersection:

1. **No unified framework** currently integrates statistical
   anomaly detection, ontology-based violation scoring, and
   generative plausibility assessment into a single calibrated
   pipeline for EHR sequences.
2. **Counterfactual explanations for EHR anomalies** are rarely
   constrained by medical ontologies, risking clinically
   meaningless or unsafe suggestions.
3. **Honest benchmarking** of generative components is uncommon;
   negative or weak results for generative surprise signals are
   frequently unreported.
4. **Reproducibility infrastructure** for EHR anomaly-detection
   research is often incomplete, lacking environment capture,
   private-data boundary documentation, and artefact auditing.

This work addresses these gaps by proposing an ontology-calibrated
framework that combines detection, scoring, and explanation,
reports negative findings transparently, and provides
comprehensive reproducibility tooling.

### 2.7 Working Hypothesis

The working hypothesis of this research is:

> *An anomaly-detection framework that calibrates its scoring
> function using medical ontology signals (SNOMED CT, RxNorm,
> UMLS) and constrains its counterfactual explanations to
> ontology-consistent perturbations will produce more clinically
> interpretable, actionable, and safe outputs than frameworks
> relying on statistical deviation alone.*

This hypothesis is tested through ablation studies (Days 40–41),
end-to-end case studies (Day 39), and failure-mode analyses
(Day 47), with the current evidence indicating that the
ontology-calibrated and supervised-detector components provide
the strongest signal, while the generative component requires
further development.

---

## Preliminary Empirical Evidence

In supervised evaluation on synthetic injected anomalies (Day 20), the detector achieved a ROC-AUC of 0.800 and an average precision of 0.733, with precision 0.904, recall 0.539, and F1 0.675 at the selected operating threshold.  The hardest anomaly type was *missing diagnosis*; the easiest was *demographic conflict*.

In contrast, the diffusion-based generative surprise signal (Sgen), evaluated under exact checkpoint alignment at Day 34, achieved a best ROC-AUC of only 0.508 across all tested timesteps — effectively near random.  After exact checkpoint alignment, the current diffusion midpoint/timestep denoising-error Sgen proxy does not meaningfully separate injected anomalies from normal records. The result is close to random across all tested timesteps.  Consequently, Sgen is treated as a diagnostic or auxiliary signal in the current framework and should **not** be claimed as a strong standalone anomaly detector.

---

*Draft generated by `scripts/build_day49_executive_background.py`.
All `[CITATION NEEDED]` markers must be resolved with verified
references before manuscript submission.*
