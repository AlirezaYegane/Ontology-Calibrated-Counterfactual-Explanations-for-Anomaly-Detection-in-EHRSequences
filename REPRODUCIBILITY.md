# Reproducibility

This is the top-level entry point for reproducing the results of *Ontology-Calibrated
Counterfactual Explanations for Relational Anomaly Detection in EHR Sequences*. It tells you
what is and is not in the repository, how to set up an environment, how to obtain the
restricted inputs, and how to regenerate the aggregate evidence.

All final evidence is on **MIMIC-IV benchmark-v2**. External validation on eICU is **blocked**
by a token-schema mismatch and is future work.

## What is in git

- All source code (`src/`), CLI scripts (`scripts/`), configs (`configs/`), tests (`tests/`).
- **Aggregate-only** artifacts for every phase under `artifacts/` (JSON / Markdown / CSV /
  figure PNGs). Every headline number in the paper is backed by one of these.
- The full paper package (`docs/paper/`) and this reproducibility package
  (`docs/reproducibility/`).

## What is NOT in git (and why)

Restricted or patient-derived material is git-ignored for licensing and privacy reasons:

- raw and processed clinical data (`data/processed/`, `*.pkl`, `*.parquet`);
- benchmark-v2 split files (patient-derived);
- ontology dumps and derived maps (`ontologies/raw/`, `ontologies/processed/`);
- model checkpoints (`*.pt`, `*.pth`, `*.ckpt`) and MIMIC-derived vocabularies;
- per-record score files (`**/per_record*`, run `ignored/` folders).

No individual patient record can be reconstructed from what is committed.

## Quick start (no restricted data needed)

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest
python -m pytest        # full suite runs on CPU; data-dependent tests skip cleanly
```

## Full rebuild (credentialed users)

Follow, in order:

1. [`docs/reproducibility/environment.md`](docs/reproducibility/environment.md) — Python,
   packages, seeds.
2. [`docs/reproducibility/data_access.md`](docs/reproducibility/data_access.md) — obtain
   MIMIC-IV, eICU, UMLS/SNOMED/RxNorm.
3. [`docs/reproducibility/runbook.md`](docs/reproducibility/runbook.md) — the exact command
   order to rebuild ontology maps, sequences, benchmark-v2, and to rerun the final
   evaluation, ablations, counterfactual evaluation, tables, and the external check.

## Interpreting the artifacts

- [`docs/reproducibility/artifact_manifest.md`](docs/reproducibility/artifact_manifest.md)
  and [`artifacts/phase8/artifact_manifest.json`](artifacts/phase8/artifact_manifest.json)
  describe what each committed artifact contains.
- The claim ledger is [`docs/paper/final_claims_matrix.md`](docs/paper/final_claims_matrix.md).
- The paper asset index is
  [`artifacts/phase8/paper_asset_index.md`](artifacts/phase8/paper_asset_index.md).

## Verifying the finalization

```bash
python scripts/run_phase8_final_checks.py
```

This lightweight script checks that the README, manuscript, reproducibility docs, claims
matrix, and `.gitignore` protections are present and consistent, and that no per-record
dumps are staged under `artifacts/phase8/`.

## Avoiding accidental data commits

Before any commit, confirm nothing restricted is staged:

```bash
git diff --cached --name-only | grep -E "data/processed|ontologies/(raw|processed)|\.pkl$|\.pt$|\.parquet$|\.zip$|per_record|checkpoint|vocab" && echo "STOP: restricted file staged" || echo "clean"
```

The `.gitignore` already covers these patterns; the check above is a belt-and-braces guard.
