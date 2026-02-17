
flowchart LR
  A[Raw EHR Tables\n(MIMIC/eICU)] --> B[Preprocessing & Mapping\nExtract events + demographics\nICD->SNOMED, Drug/NDC->RxNorm]
  B --> C[Normalized Record Store\nRecord{events, demographics, raw_codes}]

  C --> D[Baseline Anomaly Detector\n(GRU/Transformer)]
  D -->|S_det + optional token surprise| F[Score Orchestrator / Calibrator]

  C --> E[Diffusion-based Generative Model\n(ontology-regularized)]
  E -->|S_gen + candidate reconstructions| F

  C --> G[Ontology Engine\nMappings + Graph + Rules]
  G -->|violations + S_ont| F

  F -->|ScorePack\n{S_det,S_gen,S_ont,S_cal,violations}| H[Counterfactual Generator\nOntology-constrained edits\nmin λ·#edits + S_cal(X')]
  H -->|X* + edits + scores_before/after| I[Explanation Interface\nHuman-readable report]
  G -->|labels + relations| I

  I --> J[Output\nExplanation + Counterfactual]
