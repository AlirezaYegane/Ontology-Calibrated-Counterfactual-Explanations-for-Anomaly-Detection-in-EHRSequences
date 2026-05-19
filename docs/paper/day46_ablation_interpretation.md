# Day 46 — Results Interpretation for Manuscript

## Main result

The strongest variant in this run is `detector_only`, with ROC-AUC=0.8002, AP=0.7332, F1=0.6748, precision=0.9120, and recall=0.5355.

## Generative-only signal

The `generative_only` variant remains weak as a standalone anomaly discriminator. This supports reporting the generative component conservatively as an auxiliary or diagnostic signal, rather than claiming that it independently drives anomaly detection performance.
In this run, `generative_only` achieved ROC-AUC=0.5000 and AP=0.2299.

## Detector-only baseline

The `detector_only` baseline achieved ROC-AUC=0.8002 and AP=0.7332. This is the key statistical baseline for judging whether ontology calibration adds useful information.

## Ontology ablation

The `no_ontology` variant achieved ROC-AUC=0.8002 and AP=0.7332. This variant is important because it isolates how much performance remains when explicit ontology calibration is removed.

## Paper-safe interpretation

A conservative manuscript claim should focus on comparative evidence: which scoring component contributes signal, which component fails as a standalone discriminator, and how ontology-aware calibration changes performance relative to ablated variants. Avoid overstating the diffusion/generative component if the ablation shows weak discriminative value.

## Recommended manuscript wording

> The ablation study indicates that anomaly discrimination is primarily driven by the detector and ontology-calibrated scoring components, while the current generative-only score provides limited standalone discrimination. We therefore treat the generative score as an auxiliary diagnostic signal in the present implementation and report ontology/detector contributions transparently through ablation results.
