# [3] GRAM: Graph-based Attention Model for Healthcare Representation Learning

**Authors:** Choi et al.  
**Venue/Year:** KDD, 2017  
**Bucket:** Medical Ontologies / Knowledge Graph

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** ICD/clinical code hierarchies

## Core idea / method

- **Model:** Ontology-aware attention over code ancestors

## Evaluation (as reported)

- EHR tasks on large datasets

## Key takeaways we can reuse

- **Key contribution (project view):** Ontology calibration to enforce semantic consistency in representations

- **Strengths (project view):** Strong clinical semantics; interpretable hierarchy attention

## Limitations / risks (project view)

- Depends on ontology quality/coverage; hierarchy can be coarse

## How it maps to our system

- Core for ontology-calibrated embeddings and anomaly/counterfactual plausibility constraints

## Sources (for verification)

- ACM DL: GRAM (KDD 2017): https://dl.acm.org/doi/10.1145/3097983.3098126

- PMC full text (GRAM): https://pmc.ncbi.nlm.nih.gov/articles/PMC7954122/
