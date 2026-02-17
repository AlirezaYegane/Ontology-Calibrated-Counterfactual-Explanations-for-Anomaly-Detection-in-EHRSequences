# [6] EHR-BERT: A BERT-based Model for Effective Anomaly Detection in EHRs

**Authors:** Niu et al.  
**Venue/Year:** Journal of Biomedical Informatics, 2024  
**Bucket:** Anomaly Detection in EHR/ICU

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** Discrete EHR event logs

## Core idea / method

- **Model:** Transformer anomaly detection via masked token prediction objective

## Evaluation (as reported)

- EHR anomaly detection via event prediction

## Key takeaways we can reuse

- **Key contribution (project view):** Transformer-based anomaly objective tailored to event sequences

- **Strengths (project view):** Strong sequence modeling; direct anomaly objective

## Limitations / risks (project view)

- Prediction-error anomaly definition needs calibration (rare-but-valid vs truly abnormal)

## How it maps to our system

- Anchor method for discrete-event anomaly stream before ontology calibration and counterfactuals

## Sources (for verification)

- ScienceDirect: EHR-BERT (JBI 2024): https://www.sciencedirect.com/science/article/pii/S1532046424000236

- PubMed: EHR-BERT: https://pubmed.ncbi.nlm.nih.gov/38331082/
