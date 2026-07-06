# Phase 8 Reproducibility Guide

Phase 8 finalizes the repository: it does not run new science. This guide ties the pieces
together and states exactly what a reader can reproduce today versus what requires restricted
inputs.

## What Phase 8 produced

- A humanized, final [`README.md`](../../README.md).
- The paper package under [`docs/paper/`](../paper/) — modular sections plus the combined
  [`final_manuscript.md`](../paper/final_manuscript.md).
- This reproducibility package under [`docs/reproducibility/`](.) and the top-level
  [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).
- The final claim/contribution ledger:
  [`final_claims_matrix.md`](../paper/final_claims_matrix.md),
  [`final_contribution_statement.md`](../paper/final_contribution_statement.md).
- Phase 8 artifacts under [`artifacts/phase8/`](../../artifacts/phase8/): manifest, paper
  asset index, and `phase8_summary.json`.
- A finalization check script `scripts/run_phase8_final_checks.py` and the
  `tests/test_phase8_*.py` suite.

## Reproducible today (no restricted data)

```bash
python -m pytest                          # full suite (CPU)
python scripts/run_phase8_final_checks.py # finalization consistency checks
```

You can also read and cross-check every headline number against the committed aggregate
artifacts under `artifacts/phase7/` (see
[`artifact_manifest.md`](artifact_manifest.md)).

## Reproducible with credentials (restricted data)

The full pipeline — ontology parsing, sequence extraction, benchmark-v2 build, detector
training, final evaluation, ablations, counterfactual evaluation, tables — is reproducible by
credentialed users following [`runbook.md`](runbook.md). The expected headline outputs are
listed at the end of the runbook.

## Scientific scope (do not overclaim)

- Final method is **ontology-only** (`S_main = S_ont`); the calibrated combination is
  implemented but not recommended (the detector is non-additive).
- The detector (below chance) and `Sgen` (removed from core) are **negative results**.
- Final evidence is **MIMIC-IV benchmark-v2 only**.
- External validation on eICU is **blocked** (schema mismatch) — future work.
- Repairs are ontology-valid, **not** clinician-validated.

## Status

`phase8_complete_external_validation_deferred` — the repository and the MIMIC-IV paper
package are complete; external validation remains deferred pending an APACHE→ICD/SNOMED
crosswalk or another compatible external dataset.
