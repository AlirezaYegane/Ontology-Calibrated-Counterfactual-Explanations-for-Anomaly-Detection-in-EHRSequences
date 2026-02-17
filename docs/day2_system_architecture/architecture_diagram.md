# Day 2 — System Architecture Diagram (Mermaid)

```mermaid
flowchart LR
  A[Raw EHR Tables<br/>(MIMIC/eICU)] --> B[Preprocessing & Mapping<br/>Extract events + demographics<br/>ICD->SNOMED, Drug/NDC->RxNorm]
  B --> C[Normalized Record Store<br/>Record{events, demographics, raw_codes}]

  C --> D[Baseline Anomaly Detector<br/>(GRU/Transformer)]
  D -->|S_det + optional token surprise| F[Score Orchestrator / Calibrator]

  C --> E[Diffusion-based Generative Model<br/>(ontology-regularized)]
  E -->|S_gen + candidate reconstructions| F

  C --> G[Ontology Engine<br/>Mappings + Graph + Rules]
  G -->|violations + S_ont| F

  F -->|ScorePack<br/>{S_det,S_gen,S_ont,S_cal,violations}| H[Counterfactual Generator<br/>Ontology-constrained edits<br/>min lambda·#edits + S_cal(X')]
  H -->|X* + edits + scores_before/after| I[Explanation Interface<br/>Human-readable report]
  G -->|labels + relations| I

  I --> J[Output<br/>Explanation + Counterfactual]
