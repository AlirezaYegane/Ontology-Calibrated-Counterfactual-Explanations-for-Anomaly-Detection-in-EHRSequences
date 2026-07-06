# Experiments

All experiments use **benchmark-v2 (non-circular, MIMIC-IV)** as the final benchmark. The
old circular benchmark is not used as final evidence.

## Dataset and splits

| Split | Records | Anomalies | Subjects |
|---|---:|---:|---:|
| Train (normal-only) | 20,570 | 0 | 10,931 |
| Validation | 3,123 | 610 | 1,561 |
| Test | 6,307 | 1,376 | 3,125 |

Subject overlap across splits is zero. Test anomaly rate is 21.8%. Anomaly composition:
demographic 423, medication-indication 1,245, forbidden co-occurrence 318 (across the full
30,000-record pool before splitting).

## Final evaluation protocol

- **Threshold calibration is validation-only.** The best-F1 threshold is selected on the
  validation split and applied unchanged to the test split; there is no test-set tuning.
- **Confidence intervals** on every metric are bootstrap (percentile) intervals.
- **Significance** between variants uses a **paired bootstrap** on ROC-AUC (same resampled
  indices for both variants), reporting the observed difference, its CI, and a p-value.

Variants evaluated:

- `ontology_only_real` — the real ontology scorer (main method);
- `legacy_baseline` — the legacy ICD-prefix rule scorer;
- `detector_only_full` — the full-scale unsupervised detector alone;
- `combined_real_without_sgen` — the calibrated `S_det + S_ont` combination (`w_gen = 0`).

## Detector training

The unsupervised detector is trained on the clean-normal-only train split (20,570 records,
vocab 17,867) as a next-token model, with validation-based early stopping, deterministic
seeds, resumable checkpoints, and mini-batched scoring. The full run (`phase6_detector_full_gpu`)
ran 25 epochs; the best validation ROC-AUC was 0.4698 at epoch 19. Checkpoints and the
MIMIC-derived vocabulary are git-ignored.

## Ablations

- **Rule-family ablation** — each of the three ontology rule families alone, the full set,
  the legacy baseline, and an ontology-disabled control.
- **Score-component ablation** — `S_ont` only, `S_det` only, `S_ont + S_det`, and
  `S_ont + S_det + S_gen` (marked excluded).
- **Counterfactual edit-strategy ablation** — remove-only, replace-only,
  add-context-allowed, and full policy, on a capped sample of ontology-flagged anomalies.

## Counterfactual evaluation

The leakage-free generator is run on all test anomalies (1,376). We report repair success
overall and among the ontology-flagged subset, mean ΔS_ont, edit counts, and per-rule-type
success. Leakage is exercised behaviorally: adding every answer key leaves the repair
byte-identical, misleading answer keys are ignored, and outputs contain no hidden fields.

## External validation check

We attempt external validation on eICU by sampling 500 records and testing whether their
tokens map into the ICD/SNOMED/RxNorm ontology and fire any rule. This is a schema-compatibility
check, reported honestly whether it passes or fails.
