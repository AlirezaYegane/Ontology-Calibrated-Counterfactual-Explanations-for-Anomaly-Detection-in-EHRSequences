# [5] DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series

**Authors:** Munir et al.  
**Venue/Year:** IEEE Access, 2019  
**Bucket:** Anomaly Detection in EHR/ICU

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** Continuous time-series (vitals/waveforms)

## Core idea / method

- **Model:** Unsupervised CNN-based forecasting/anomaly scoring

## Evaluation (as reported)

- Time-series anomaly benchmarks

## Key takeaways we can reuse

- **Key contribution (project view):** Baseline for continuous-signal anomaly module

- **Strengths (project view):** Practical baseline; handles continuous channels

## Limitations / risks (project view)

- Weak clinical grounding; can over-flag noise without constraints

## How it maps to our system

- Baseline comparator for vitals/waveform anomaly detection stream

## Sources (for verification)

- DeepAnT PDF: https://www.dfki.de/fileadmin/user_upload/import/10175_DeepAnt.pdf

- MathWorks reference page: https://www.mathworks.com/help/predmaint/ref/deepantdetector.html
