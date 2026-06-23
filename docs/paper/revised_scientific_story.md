# Revised Scientific Story (Phase 0)

**One-paragraph honest narrative.** Electronic health record (EHR) sequences contain
documentation errors and clinically inconsistent code combinations that are hard to
surface and harder to explain. We build a reproducible benchmark for EHR-sequence
anomalies on real MIMIC-IV data, taking care that the anomalies require *relational*
clinical reasoning rather than trivially detectable artifacts. On top of a sequence
anomaly detector, we add an ontology-grounded explanatory layer (SNOMED CT / RxNorm)
that flags *why* a record is inconsistent, and a counterfactual repair module that
proposes minimal, ontology-valid edits to restore coherence. Crucially, the
counterfactual never sees the corruption answer key: it proposes edits only from
ontology neighborhoods and is validated by an independent anomaly scorer. The
contribution is an interpretable, leakage-free explanation pipeline plus the benchmark
and protocol that make such claims testable.

## What changed versus the original proposal

| Original framing | Revised, evidence-based framing |
|---|---|
| Anomaly decomposition `Sdet = Sgen + Sont` as core theory | Transparent, validation-fit calibrated score; additive identity dropped |
| Diffusion-based generative surprise is a pillar | Diffusion is diagnostic-only until it passes a decision gate (Phase 5) |
| Diffusion generates counterfactual repairs | Counterfactual repair is ontology-neighborhood search, independently validated; diffusion variant only if Phase 5 passes |
| "Ontology-calibrated" (assumed) | Ontology must be really integrated (Phase 2) before the word is used |
| Detector performance is the headline | Detector is a non-circular backbone; the headline is the explanation method + benchmark |

## The main novelty (what we will actually defend)

**Leakage-free, ontology-constrained counterfactual explanation for EHR-sequence
anomalies, evaluated on a non-circular benchmark.** Everything else (detector,
calibrated score, ontology rules) is supporting machinery.

## What is explicitly future work

- Generative / diffusion surprise and diffusion-based repair (unless Phase 5 passes).
- Temporal/physiological anomaly types beyond coded events.
- Prospective clinical deployment and multi-site generalization beyond eICU.

## Non-negotiable honesty constraints

- No "ontology" claims until a real ontology is loaded and consulted at scoring time.
- No counterfactual claim that could be produced by reading the corruption answer key.
- No ablation "improvement" claim without a confidence interval and significance test.
- Report negative results (e.g., Sgen, near-zero ontology ranking gain) plainly.
