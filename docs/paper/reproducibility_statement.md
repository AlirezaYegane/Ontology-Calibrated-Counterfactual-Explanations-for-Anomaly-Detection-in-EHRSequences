# Reproducibility Statement

## What is included

The repository contains all source code for preprocessing, ontology parsing/mapping, the
ontology scorer and rule packs, the unsupervised detector and its training/eval
infrastructure, the leakage-free counterfactual generator, and the full phase-by-phase
evaluation scripts. It also contains **aggregate-level artifacts** (JSON / Markdown / CSV /
figure data) for every phase, and a pytest suite that exercises the pipeline logic and
guards.

## What cannot be redistributed

The restricted clinical data (MIMIC-IV, eICU) and licensed ontologies (UMLS, SNOMED CT,
RxNorm) cannot be committed or redistributed. Consequently the following are **git-ignored**
and absent from the repository: raw and processed patient-level records, benchmark-v2 split
`.pkl` files, ontology dumps and derived maps, model checkpoints (`.pt`), MIMIC-derived
vocabularies, and any per-record score files. Every committed number is an *aggregate*
statistic; no individual patient record is recoverable from what is tracked.

## Rerunning the tests

```bash
python -m pytest
```

The suite runs on CPU and does not require the restricted data; data-dependent tests skip
cleanly when local splits are absent.

## Rebuilding from data (for credentialed users)

Holders of PhysioNet and UMLS credentials can rebuild end-to-end:

1. Obtain MIMIC-IV and the UMLS Metathesaurus (see
   [`ethics_and_data_statement.md`](ethics_and_data_statement.md) and
   `docs/reproducibility/data_access.md`).
2. Parse ontologies and build the ICD→SNOMED / drug→RxNorm maps.
3. Extract MIMIC-IV sequences and map them into ontology space.
4. Build benchmark-v2 (`scripts/build_benchmark_v2.py`) and verify the non-circularity gate
   (`scripts/diagnose_anomaly_triviality.py`).
5. Run the final evaluation, ablations, counterfactual evaluation, tables, and the external
   validation check (`scripts/run_phase7_*.py`).

The detailed order is in `docs/reproducibility/runbook.md`.

## Determinism

Random operations use fixed seeds (benchmark build seed 42; deterministic counterfactual
search; seeded detector training). Aggregate metrics are stable across reruns; bootstrap
confidence intervals are reported for all headline numbers.

## Why restricted data and ontology dumps are excluded

This is a licensing and privacy requirement, not an omission. Redistributing MIMIC-IV,
UMLS/SNOMED/RxNorm content, or patient-derived vocabularies would violate the PhysioNet data
use agreement and the UMLS license. The reproducibility path is therefore "obtain the same
sources under your own credentials, then run the committed code," which the runbook supports.
