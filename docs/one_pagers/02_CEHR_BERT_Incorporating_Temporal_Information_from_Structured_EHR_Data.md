# [2] CEHR-BERT: Incorporating Temporal Information from Structured EHR Data

**Authors:** Pang et al.  
**Venue/Year:** PMLR (ML4H), 2021  
**Bucket:** EHR Representation Learning

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** Structured EHR + time

## Core idea / method

- **Model:** Transformer with explicit temporal encoding for irregular intervals

## Evaluation (as reported)

- EHR predictive tasks

## Key takeaways we can reuse

- **Key contribution (project view):** Temporal encoding to model irregular gaps; critical for ICU temporal signals

- **Strengths (project view):** Better temporal fidelity than token-only modeling

## Limitations / risks (project view)

- Temporal assumptions may not generalize across hospitals/granularities

## How it maps to our system

- Supports temporal anomaly detection component in ICU setting

## Sources (for verification)

- PMLR: CEHR-BERT (ML4H 2021): https://proceedings.mlr.press/v158/pang21a.html

- arXiv: CEHR-BERT (2111.08585): https://arxiv.org/abs/2111.08585
