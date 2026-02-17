# [1] Med-BERT: Pre-trained Contextualized Embeddings on Large-scale Structured EHRs

**Authors:** Rasmy et al.  
**Venue/Year:** NPJ Digital Medicine, 2021  
**Bucket:** EHR Representation Learning

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** Structured EHR tokens

## Core idea / method

- **Model:** Transformer/BERT; self-supervised pretraining; time encoding

## Evaluation (as reported)

- Large-scale structured EHR; downstream prediction tasks

## Key takeaways we can reuse

- **Key contribution (project view):** Backbone encoder; demonstrates viability of massive pretraining for robust clinical representations

- **Strengths (project view):** Strong transferable embeddings; scalable pretraining; supports irregular timing with explicit encoding

## Limitations / risks (project view)

- Needs very large data; tokenization/time encoding choices affect stability

## How it maps to our system

- Candidate backbone encoder for our pipeline; aligns with representation learning foundation

## Sources (for verification)

- Nature: Med-BERT (npj Digital Medicine): https://www.nature.com/articles/s41746-021-00455-y

- arXiv: Med-BERT (2005.12833): https://arxiv.org/abs/2005.12833
