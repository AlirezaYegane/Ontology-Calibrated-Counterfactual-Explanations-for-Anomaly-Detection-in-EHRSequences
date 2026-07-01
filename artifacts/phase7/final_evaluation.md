# Phase 7 -- Final Evaluation (benchmark-v2)

**Benchmark:** benchmark-v2 (non-circular; FINAL benchmark) | **Sgen in core:** False (w_gen=0.0) | **strongest:** `ontology_only_real`

Score equation: `S_cal = (w_det*S_det + w_ont*S_ont') / (w_det + w_ont), w_gen=0`

| variant | ROC-AUC | 95% CI | AP | F1 (val-thr) | P | R |
|---|---:|---|---:|---:|---:|---:|
| ontology_only_real | 0.7881 | [0.7743, 0.8015] | 0.5422 | 0.6349 | 0.5936 | 0.6824 |
| legacy_baseline | 0.7358 | [0.7197, 0.7511] | 0.5429 | 0.5928 | 0.6294 | 0.5603 |
| detector_only_full | 0.4525 | [0.4357, 0.4702] | 0.1904 | 0.3608 | 0.2205 | 0.9927 |
| combined_real_without_sgen | 0.7036 | [0.6866, 0.7201] | 0.4039 | 0.4596 | 0.3909 | 0.5574 |

**Answers:** {'real_ontology_beats_legacy': True, 'detector_improves_over_ontology_only': False, 'combined_improves_over_ontology_only': False}

> Sgen excluded from core (Phase 5 remove_from_core; diagnostic ROC-AUC 0.4868).