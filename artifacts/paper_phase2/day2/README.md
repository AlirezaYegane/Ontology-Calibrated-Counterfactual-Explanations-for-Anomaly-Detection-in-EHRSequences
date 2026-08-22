# Paper Phase 2 — Day 2 Validation Selection

## Status

Complete — validation-only B0/B1 tuning and length/pair-count bias audit.

This remains a preliminary benchmark-v2 thesis-diagnostic experiment until benchmark-v3 generator-independent strata are ready.

## Data policy

- Statistics fitted from clean training data only.
- Hyperparameter selection used validation only.
- Test data was not accessed.
- Ontology information was not used.
- Hidden/audit answer keys were not used as model features.
- No per-record scores were saved.

## Frozen B0

- Candidate: `b0|surprisal|alpha=0.1|agg=topk3`
- PR-AUC: `0.245461`
- ROC-AUC: `0.597859`
- Max |Spearman bias|: `0.5332609951265737`

## Frozen B1

- Candidate: `b1|conditional_relation|alpha=0.1|minsup=20|agg=topk10`
- PR-AUC: `0.250262`
- ROC-AUC: `0.610746`
- Max |Spearman bias|: `0.6253312195997744`

## Robust vs extreme audit

### B0

- PR-AUC delta selected - extreme: `-0.003605`
- Bias delta selected - extreme: `0.09968829371795285`
- Bias reduced: `False`

### B1

- PR-AUC delta selected - extreme: `0.004279`
- Bias delta selected - extreme: `0.12704909856994834`
- Bias reduced: `False`

## Selection policy

Extreme max/worst aggregations were audited but were not eligible for the final frozen scorer.

Among non-extreme candidates, selection was:

1. highest validation PR-AUC;
2. lower maximum absolute Spearman correlation with sequence length / candidate-pair count as the first tie-breaker;
3. higher ROC-AUC as the second tie-breaker;
4. lexical candidate ID as the final deterministic tie-breaker.

No post-hoc numerical correlation threshold was introduced.

## Next step

Day 3 may open the test split once, using only these frozen B0/B1 configurations.

---

## Final Scientific Assessment

Day 2 identified a clear performance-versus-length-bias tradeoff.

### Performance-selected configurations

- B0: `alpha=0.1`, `topk3`
  - PR-AUC: 0.2455
  - ROC-AUC: 0.5979
  - max length/pair Spearman: 0.5333

- B1: `alpha=0.1`, `min_support=20`, `topk10`
  - PR-AUC: 0.2503
  - ROC-AUC: 0.6107
  - max length/pair Spearman: 0.6253

These configurations provide the strongest validation discrimination under the pre-run selection rule, but they are not length robust.

### Length-robust audit configurations

- B0 q95 (`alpha=0.1`)
  - PR-AUC: 0.2104
  - ROC-AUC: 0.5309
  - max length/pair Spearman: 0.0248

- B1 q90 (`alpha=0.1`, `min_support=20`)
  - PR-AUC: 0.2066
  - ROC-AUC: 0.5132
  - max length/pair Spearman: 0.0439

- B1 q95 (`alpha=0.1`, `min_support=20`) provides an intermediate tradeoff:
  - PR-AUC: 0.2209
  - ROC-AUC: 0.5478
  - max length/pair Spearman: 0.1543

### Interpretation

The apparent B1 relational signal is strongly attenuated when sequence-length and candidate-pair-count sensitivity is reduced.

Therefore:

- do not claim strong length-robust relational anomaly detection from benchmark-v2;
- describe the top-k result as performance-selected but shortcut-sensitive;
- retain q90 as the strict length-robust B1 audit comparator;
- treat q95 as an intermediate robustness diagnostic;
- perform no further validation tuning.

### Frozen Day 3 test policy

The test split may be opened once.

Evaluate only the already frozen configurations:

1. B0 topk3
2. B0 q95
3. B1 topk10
4. B1 q90
5. B1 q95 diagnostic

No hyperparameters may be changed after observing test results.

