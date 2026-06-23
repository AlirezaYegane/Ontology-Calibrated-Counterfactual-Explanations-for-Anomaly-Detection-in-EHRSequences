# Phase 3 Scoring-Variant Evaluation

**Status:** `valid_v2_benchmark`  
**Benchmark-v2 triviality pass (<0.80):** True  
**Final paper evidence:** False (smoke_scale_valid_non_circular)

## Test-set detection metrics (non-circular benchmark-v2, smoke-scale detector)

| variant | ROC-AUC | ROC-AUC 95% CI | AP | F1 (val-thr) |
|---|---:|---|---:|---:|
| detector_only | 0.4247 | [0.4074, 0.4401] | 0.1803 | - |
| ontology_only_real | 0.7881 | [0.7743, 0.8016] | 0.5422 | - |
| combined_real_ontology | 0.6581 | [0.6395, 0.6758] | 0.396 | 0.4185 |
| combined_legacy_ontology | 0.6625 | [0.644, 0.6791] | 0.4464 | - |

> ⚠️ Detector is SMOKE-SCALE (small model / few epochs / capped train). Metrics are preliminary non-circular evidence, NOT final paper SOTA; full-scale training is a later experiment phase.