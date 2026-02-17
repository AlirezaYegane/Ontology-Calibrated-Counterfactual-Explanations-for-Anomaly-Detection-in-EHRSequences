# [8] EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models

**Authors:** Li et al.  
**Venue/Year:** TMLR, 2023  
**Bucket:** Generative Modeling for Clinical Data

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** EHR records

## Core idea / method

- **Model:** Diffusion models for EHR synthesis

## Evaluation (as reported)

- EHR synthesis quality comparisons

## Key takeaways we can reuse

- **Key contribution (project view):** Stable diffusion-based generator for synthetic EHR and counterfactual sampling

- **Strengths (project view):** Training stability; strong generative fidelity

## Limitations / risks (project view)

- Compute heavy; privacy requires auditing; conditioning design matters

## How it maps to our system

- Primary candidate generative backbone for counterfactual generation

## Sources (for verification)

- arXiv: EHRDiff (2303.05656): https://arxiv.org/abs/2303.05656

- OpenReview: EHRDiff: https://openreview.net/forum?id=DIGkJhGeqi
