# Phase 7 -- Ablation Results (benchmark-v2)

## 1. Ontology-rule ablation

| variant | ROC-AUC | 95% CI | AP | normal FP | demo | med | forbidden |
|---|---:|---|---:|---:|---:|---:|---:|
| real_ontology_rules_full | 0.7881 | [0.7743, 0.8015] | 0.5422 | 0.1304 | 0.9963 | 0.6858 | 0.9318 |
| legacy_icd_prefix_rules | 0.7358 | [0.7197, 0.7511] | 0.5429 | None | 0.9969 | 0.737 | 0.4206 |
| demographic_rules_only | 0.5988 | [0.5875, 0.6092] | 0.3682 | 0.0016 | 0.9992 | 0.5003 | 0.4992 |
| medication_rules_only | 0.5711 | [0.5607, 0.5823] | 0.271 | 0.0635 | 0.4995 | 0.6186 | 0.4748 |
| forbidden_cooccurrence_rules_only | 0.6252 | [0.6124, 0.638] | 0.3296 | 0.0671 | 0.4995 | 0.575 | 0.9664 |
| ontology_disabled | None | [None, None] | 0.2182 | None | None | None | None |

## 2. Score-component ablation (Sgen excluded)

| variant | ROC-AUC | AP | note |
|---|---:|---:|---|
| S_ont_only | 0.7881 | 0.5422 |  |
| S_det_only | 0.4525 | 0.1904 |  |
| S_ont_plus_S_det | 0.7036 | 0.4039 |  |
| S_ont_plus_S_det_plus_Sgen | None | - | Sgen removed in Phase 5 (diagnostic ROC-AUC 0.4868, harms combined). w_gen=0. |

> Sgen excluded from all core variants (w_gen=0).