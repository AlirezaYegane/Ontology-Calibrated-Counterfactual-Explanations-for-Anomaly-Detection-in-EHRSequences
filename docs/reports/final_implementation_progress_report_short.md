# Short Final Progress Summary

**Project:** Ontology-Calibrated Counterfactual Explanations for Anomaly Detection in EHR Sequences
**Author:** Alireza Yegane · **Supervisor:** Professor Xuyun Zhang · Macquarie University
**Evidence scope:** MIMIC-IV benchmark-v2 · **Status:** finalized (Phase 8 complete)

## Project Direction

The project aimed to detect clinically incoherent patterns in sequential electronic health
record (EHR) data and to explain them. The key insight is that many EHR errors are not rare
events but incoherent *combinations* of common codes — for instance, a sex-specific
diagnosis recorded against the wrong sex, or a medication present without its indicating
diagnosis. The original design combined four ideas: a sequence-based anomaly detector, an
ontology calibration layer (SNOMED CT / RxNorm), a generative/diffusion "surprise" score,
and a counterfactual explanation module that proposes minimal repairs. The intended novelty
was the interpretable, leakage-free counterfactual explanation on top of a calibrated score.

## What Was Implemented

We built the full pipeline: extraction of MIMIC-IV admissions into code sequences, parsing
of the licensed terminologies, and real ICD→SNOMED and drug→RxNorm crosswalks with coverage
of 0.80 for diagnoses and 0.78 for medications. Early on, we audited the first anomaly
benchmark and found it circular — a single token-presence feature recovered the label at
roughly 0.94, meaning a model could appear to "detect" anomalies without any clinical
reasoning. We therefore built a leakage-controlled replacement, **benchmark-v2**, in which
each anomaly is a violation of a *relationship* between fields while every individual token
stays common. It uses subject-level splits with zero overlap, a clean-normal-only training
split, and passes a pre-registered non-circularity gate (strongest trivial signal 0.6127 <
0.80). On top of this we rebuilt scoring around three auditable ontology rule families
(demographic sex-restriction, medication required-context, diabetes-type exclusion), trained
the detector at full scale, put the generative score through an explicit decision gate, and
rewrote the counterfactual module to be leakage-free.

## Key Findings

The evidence redirected the project. The real ontology scorer became the reliable signal,
reaching ROC-AUC 0.7881 and significantly beating the legacy ICD-prefix baseline. The two
components we had expected to carry the work did not. The full-scale detector scored 0.4525 —
below chance — because the anomalies are relational and carry almost no next-token surprise;
adding it to the ontology score made results significantly worse. The generative/diffusion
score (`Sgen`) scored 0.4868, pointed the wrong way, was mode-collapsed, and harmed the
combined score, so it was removed from the core method (`w_gen = 0`). Rather than keep the
detector and generative score because they were part of the original plan, we treated their
weak performance as a finding and narrowed the final claim. The counterfactual module, once
rewritten to read only model-visible fields and never the injection answer key, achieved
89.99% valid repair among the ontology-flagged anomalies (939 of 1,376), with a median of a
single edit.

## Final Result

The final method is ontology-centered: `S_main = S_ont`. The calibrated combination
infrastructure still exists but is not recommended, because the detector harms performance.
All numbers are on benchmark-v2 with validation-only thresholding and bootstrap confidence
intervals.

| Variant | ROC-AUC | 95% CI | AP | F1 |
| --- | ---: | --- | ---: | ---: |
| ontology_only_real | 0.7881 | [0.774, 0.802] | 0.542 | 0.635 |
| legacy_baseline | 0.7358 | [0.720, 0.751] | 0.543 | 0.593 |
| detector_only_full | 0.4525 | [0.436, 0.470] | 0.190 | 0.361 |
| combined_real_without_sgen | 0.7036 | [0.687, 0.720] | 0.404 | 0.460 |
| Sgen diagnostic | 0.4868 | diagnostic only | diagnostic only | diagnostic only |

```text
Real ontology beats legacy baseline:        ΔROC-AUC = +0.052, p ≈ 0
Adding the detector hurts vs ontology-only:  ΔROC-AUC = -0.085, p ≈ 0
Counterfactual repair among flagged (939):   89.99% valid, median 1 edit, mean ΔS_ont 0.644
```

The final contribution is an ontology-centered anomaly detection and counterfactual
explanation framework, with the detector and `Sgen` retained as transparent negative
findings. This is scientifically stronger than the original framing because unsupported
components were removed and reported rather than hidden. The repository is finalized: the full
test suite stands at 341 passed, and the README, paper package, and reproducibility package
are complete with a clean data-safety audit (only aggregate artifacts committed).

## Limitations and Future Work

All final evidence is on MIMIC-IV only. The anomalies are synthetic and constructed, not
clinician-confirmed real-world errors; the rule packs are curated rather than learned; and the
counterfactual repairs are validated by an ontology scorer, not by clinicians. External
validation on eICU is deferred because of a schema mismatch: eICU uses APACHE/body-system
tokens (0 of 500 sampled records mapped into the ontology), so direct evaluation is blocked.
We make no claim of external generalization, clinical deployment readiness, or
clinician-validated repairs. The main remaining work is external validation, which requires a
future APACHE-to-ICD/SNOMED crosswalk or another ontology-compatible external dataset; a
clinician study of repair usefulness would be a natural complement.
