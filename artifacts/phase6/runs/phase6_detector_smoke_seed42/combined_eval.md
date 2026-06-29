# Phase 6 -- Combined Score Evaluation (Sgen-free)

**run:** `phase6_detector_smoke_seed42` | evidence smoke | w_gen=0.0 (Sgen excluded)

| variant | ROC-AUC | 95% CI | AP |
|---|---:|---|---:|
| detector_only | 0.426 | [0.4087, 0.4414] | 0.181 |
| ontology_only_real | 0.7881 | [0.7743, 0.8016] | 0.5422 |
| combined_real_without_sgen | 0.6587 | [0.6399, 0.6762] | 0.3953 |
| legacy_baseline | 0.6637 | [0.645, 0.6803] | 0.4442 |

## Paired bootstrap (ROC-AUC diff)
| comparison | Δ | 95% CI | p |
|---|---:|---|---:|
| combined_vs_detector_only | 0.2327 | [0.2196, 0.2471] | 0.0 |
| combined_vs_ontology_only | -0.1294 | [-0.1416, -0.1186] | 0.0 |
| ontology_only_vs_legacy | 0.1245 | [0.1036, 0.1455] | 0.0 |
| detector_vs_ontology_only | -0.3622 | [-0.3829, -0.3435] | 0.0 |

## Answers
- detector_improves_over_ontology_only: **False**
- combined_improves_over_ontology_only: **False**
- combined_improves_over_detector_only: **True**

> Sgen excluded (w_gen=0). If the detector underperforms, the main claim stays ontology-centered (ontology_only_real is the strongest variant).