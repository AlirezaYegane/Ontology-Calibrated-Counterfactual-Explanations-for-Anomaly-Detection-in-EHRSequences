# Phase 5 -- Generative / Diffusion Sgen Evaluation (benchmark-v2)

**Status:** `evaluated_diagnostic_only` | **gate:** `remove_from_core` | evidence: diagnostic_only_old_data_modecollapsed

> DIAGNOSTIC-ONLY: checkpoint trained on OLD circular-era data + mode-collapsed; time-embedding MLP not loaded (architecture drift). NOT paper evidence.

- Sgen ROC-AUC **0.4868** CI [0.4633, 0.5109] | AP 0.1983 | score std 0.0404
- mean Sgen normal 0.5915 vs anomaly 0.5876
- corr(Sgen,S_ont) -0.0715 | corr(Sgen,S_det) 0.3512
- AUC detector_only 0.4271 | ontology_only_real 0.807
- AUC combined **without** Sgen 0.6545 | **with** Sgen 0.637
- Sgen adds signal beyond ont+det: **False** | combined improves: **False**

## Sgen ROC-AUC by anomaly family
| family | n | Sgen ROC-AUC |
|---|---:|---:|
| demographic_incompatibility | 133 | 0.4686 |
| medication_indication_mismatch | 399 | 0.4786 |
| forbidden_cooccurrence | 110 | 0.5388 |