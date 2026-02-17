\# Day 2 — System Architecture Diagram (Mermaid)



```mermaid

flowchart LR

&nbsp; A\[Raw EHR Tables\\n(MIMIC/eICU)] --> B\[Preprocessing \& Mapping\\nExtract events + demographics\\nICD->SNOMED, Drug/NDC->RxNorm]

&nbsp; B --> C\[Normalized Record Store\\nRecord{events, demographics, raw\_codes}]



&nbsp; C --> D\[Baseline Anomaly Detector\\n(GRU/Transformer)]

&nbsp; D -->|S\_det + optional token surprise| F\[Score Orchestrator / Calibrator]



&nbsp; C --> E\[Diffusion-based Generative Model\\n(ontology-regularized)]

&nbsp; E -->|S\_gen + candidate reconstructions| F



&nbsp; C --> G\[Ontology Engine\\nMappings + Graph + Rules]

&nbsp; G -->|violations + S\_ont| F



&nbsp; F -->|ScorePack\\n{S\_det,S\_gen,S\_ont,S\_cal,violations}| H\[Counterfactual Generator\\nOntology-constrained edits\\nmin λ·#edits + S\_cal(X')]

&nbsp; H -->|X\* + edits + scores\_before/after| I\[Explanation Interface\\nHuman-readable report]

&nbsp; G -->|labels + relations| I



&nbsp; I --> J\[Output\\nExplanation + Counterfactual]



