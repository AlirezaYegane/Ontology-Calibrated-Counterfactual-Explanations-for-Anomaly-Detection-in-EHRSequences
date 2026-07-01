# Phase 7 -- Statistical Tests (paired bootstrap ROC-AUC diff)

| comparison (A vs B) | Δ(A−B) | 95% CI | p | significant |
|---|---:|---|---:|---|
| ontology_only_real vs legacy_baseline | 0.0524 | [0.0325, 0.0718] | 0.0 | True |
| ontology_only_real vs detector_only_full | 0.3356 | [0.3141, 0.3581] | 0.0 | True |
| combined_real_without_sgen vs ontology_only_real | -0.0845 | [-0.0963, -0.0732] | 0.0 | True |
| combined_real_without_sgen vs detector_only_full | 0.2511 | [0.2364, 0.2665] | 0.0 | True |
| combined_real_without_sgen vs legacy_baseline | -0.0322 | [-0.0567, -0.0103] | 0.004 | True |