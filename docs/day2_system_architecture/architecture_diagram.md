# Day 2 — System Architecture Diagram (Mermaid)

```mermaid
flowchart LR
  A["Raw EHR Tables\n(MIMIC/eICU)"] --> B["Preprocessing and Mapping\nExtract events + demographics\nICD to SNOMED, Drug/NDC to RxNorm"]
  B --> C["Normalized Record Store\nRecord(events, demographics, raw_codes)"]

  C --> D["Baseline Anomaly Detector\n(GRU/Transformer)"]
  D -->|S_det + optional token surprise| F["Score Orchestrator / Calibrator"]

  C --> E["Diffusion-based Generative Model\n(ontology-regularized)"]
  E -->|S_gen + candidate reconstructions| F

  C --> G["Ontology Engine\nMappings + Graph + Rules"]
  G -->|violations + S_ont| F

  F -->|"ScorePack (S_det, S_gen, S_ont, S_cal, violations)"| H["Counterfactual Generator\nOntology-constrained edits\nmin lambda*edits + S_cal(X_cf)"]
  H -->|"X_cf + edits + scores before/after"| I["Explanation Interface\nHuman-readable report"]
  G -->|labels + relations| I

  I --> J["Output\nExplanation + Counterfactual"]
