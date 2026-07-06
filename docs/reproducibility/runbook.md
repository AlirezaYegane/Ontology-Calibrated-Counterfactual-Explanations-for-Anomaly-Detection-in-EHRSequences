# Runbook

The exact command order to rebuild from restricted inputs and regenerate the aggregate
evidence. Steps 1–5 require PhysioNet + UMLS credentials; steps 6–9 produce the committed
aggregate artifacts. If you only want to run the tests, skip to "Tests only".

Prerequisites: environment set up per [`environment.md`](environment.md); data obtained per
[`data_access.md`](data_access.md); `dataset_roots.yaml` configured.

## 1. Parse ontologies (licensed)

```bash
python scripts/parse_snomed.py  --snomed-dir <snomed_rf2_dir> --output-dir ontologies/processed
python scripts/parse_rxnorm.py  --rxnorm-dir <rxnorm_rrf_dir> --output-dir ontologies/processed
```

## 2. Build ontology maps

```bash
python -m src.preprocessing.build_umls_maps   --mrconso <MRCONSO.RRF>  --output-dir ontologies/umls_maps
python -m src.preprocessing.build_rxnorm_maps --rxnconso <RXNCONSO.RRF> --output-dir ontologies/umls_maps
```

Verify coverage (should meet diagnosis 0.80 / medication 0.78):

```bash
python scripts/diagnose_ontology_rule_coverage.py    # writes artifacts/phase2b_fix/coverage_report.json
```

## 3. Extract MIMIC-IV sequences

```bash
python -m src.preprocessing.extract_mimiciv \
    --input-dir <mimic4_root> \
    --output-path data/processed/mimiciv_sequences.parquet \
    --stats-path data/processed/mimiciv_stats.json
```

## 4. Map sequences into ontology space

```bash
python -m src.preprocessing.map_sequences_to_ont \
    --sequences-dir data/processed --maps-dir ontologies/umls_maps --output-dir data/processed
```

## 5. Build benchmark-v2 and verify non-circularity

```bash
python scripts/build_benchmark_v2.py             # writes data/processed/benchmark_v2/ (git-ignored)
python scripts/diagnose_anomaly_triviality.py    # gate: strongest trivial signal < 0.80 (expect ~0.6127)
```

## 6. (Optional) Retrain the unsupervised detector

Only needed to reproduce the detector negative result from scratch; requires a GPU.

```bash
python scripts/run_phase6_train_detector.py --config configs/phase6_detector_full.yaml
python scripts/run_phase6_evaluate_detector.py
```

## 7. Final evaluation, statistics, ablations

```bash
python scripts/run_phase7_final_evaluation.py    # main results + bootstrap CIs + paired tests
python scripts/run_phase7_ablations.py           # rule-family + score-component ablations
```

## 8. Counterfactual evaluation and external check

```bash
python scripts/run_phase7_counterfactual_final.py       # leakage-free repair on test anomalies
python scripts/run_phase7_external_validation_check.py  # eICU schema check (reports blocked)
```

## 9. Tables and figures

```bash
python scripts/run_phase7_tables.py     # writes artifacts/phase7/tables/table1..5
```

## Tests only (no restricted data)

```bash
python -m pytest
python scripts/run_phase8_final_checks.py
```

## Expected headline outputs

- `ontology_only_real` ROC-AUC ≈ **0.7881** (CI [0.774, 0.802]); legacy ≈ 0.7358.
- detector_only ≈ **0.4525** (below chance); combined ≈ 0.7036 (worse than ontology-only).
- Counterfactual repair among flagged ≈ **89.99%**, median 1 edit.
- External validation: `external_validation_blocked_schema_mismatch`.
