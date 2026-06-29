# Phase 6 -- Combined Score Evaluation (Sgen-free)

**run:** `phase6_detector_full_gpu` | evidence full_gpu | w_gen=0.0 (Sgen excluded)

| variant | ROC-AUC | 95% CI | AP |
|---|---:|---|---:|
| detector_only | 0.4525 | [0.4353, 0.4693] | 0.1904 |
| ontology_only_real | 0.7881 | [0.7743, 0.8016] | 0.5422 |
| combined_real_without_sgen | 0.7036 | [0.6873, 0.7202] | 0.4039 |
| legacy_baseline | 0.6989 | [0.681, 0.715] | 0.4269 |

## Paired bootstrap (ROC-AUC diff)
| comparison | Δ | 95% CI | p |
|---|---:|---|---:|
| combined_vs_detector_only | 0.2511 | [0.2365, 0.2664] | 0.0 |
| combined_vs_ontology_only | -0.0845 | [-0.0965, -0.074] | 0.0 |
| ontology_only_vs_legacy | 0.0892 | [0.0692, 0.109] | 0.0 |
| detector_vs_ontology_only | -0.3356 | [-0.3578, -0.315] | 0.0 |

## Answers
- detector_improves_over_ontology_only: **False**
- combined_improves_over_ontology_only: **False**
- combined_improves_over_detector_only: **True**

> Sgen excluded (w_gen=0). If the detector underperforms, the main claim stays ontology-centered (ontology_only_real is the strongest variant).