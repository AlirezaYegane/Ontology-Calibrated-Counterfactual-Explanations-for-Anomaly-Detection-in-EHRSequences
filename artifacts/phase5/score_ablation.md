# Phase 5 -- Score-Variant Ablation (benchmark-v2)

**Status:** `ablation_complete_sgen_diagnostic_only` | n=3000

| variant | ROC-AUC | 95% CI | AP |
|---|---:|---|---:|
| ontology_only_real | 0.807 | [0.7913, 0.8266] | 0.5614 |
| detector_only | 0.4271 | [0.4019, 0.4532] | 0.1779 |
| combined_real_without_sgen | 0.6545 | [0.6294, 0.6837] | 0.3792 |
| combined_real_with_sgen | 0.637 | [0.6134, 0.6622] | 0.3161 |
| legacy_baseline | 0.6367 | [0.6135, 0.6617] | 0.3466 |

**Sgen ΔROC-AUC (with − without): -0.0175** (paired diff -0.0175, CI [-0.0266, -0.0087], p=0.0).
- Sgen improves ROC-AUC: **False**
- Sgen improves AP: **False**
- Sgen harms combined: **True**
- improvement statistically credible: **False**

> Sgen is DIAGNOSTIC-only (old-data, mode-collapsed checkpoint). w_gen stays 0.0 in the core.