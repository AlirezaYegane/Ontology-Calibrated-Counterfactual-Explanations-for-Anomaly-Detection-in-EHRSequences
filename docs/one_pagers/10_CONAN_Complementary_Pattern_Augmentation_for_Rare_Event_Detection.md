# [10] CONAN: Complementary Pattern Augmentation for Rare Event Detection

**Authors:** Xiao et al.  
**Venue/Year:** AAAI, 2020  
**Bucket:** Anomaly Detection (Rare Events)

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** Event sequences / rare patterns

## Core idea / method

- **Model:** Augmentation-driven rare event detection

## Evaluation (as reported)

- Rare event detection benchmarks

## Key takeaways we can reuse

- **Key contribution (project view):** Couples augmentation with rare-event detection to boost sensitivity

- **Strengths (project view):** Addresses imbalance; relevant for safety-critical rare anomalies

## Limitations / risks (project view)

- Augmentation can inject artifacts; needs clinical constraints

## How it maps to our system

- Motivates coupling generative module with anomaly detection for rare anomaly sensitivity

## Sources (for verification)

- AAAI: CONAN (2020): https://ojs.aaai.org/index.php/AAAI/article/view/5401

- arXiv: CONAN (1911.13232): https://arxiv.org/abs/1911.13232
