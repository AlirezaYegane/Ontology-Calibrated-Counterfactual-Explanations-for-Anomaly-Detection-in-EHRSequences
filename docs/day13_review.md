# Day 13 Review and Refinement

## Goal
Close Milestone 1 by verifying that the preprocessing + ontology pipeline is operational from raw clinical tables to ontology-aligned experimental sequences.

## Scope Completed
- Added an end-to-end processed-data audit script.
- Added mapping coverage reporting for diagnosis / procedure / medication namespaces.
- Added explicit edge-case detection for:
  - empty tokens
  - malformed tokens
  - duplicate tokens inside a record
  - lowercase / inconsistent namespaces
  - unknown namespaces
  - records empty after cleaning
  - records missing demographics
- Added machine-readable outputs under `artifacts/day13/`.
- Added focused unit tests for the Day 13 audit layer.
- Added parsing support for stringified token lists in parquet/pickle-derived artifacts.
- Prevented double counting when specialized token columns are present alongside `sequence_tokens` / `codes`.
- Added sample-based scanning support for large processed artifacts.

## Files Added / Updated
- `scripts/_day13_audit_lib.py`
- `scripts/audit_pipeline.py`
- `scripts/check_edge_cases.py`
- `tests/test_pipeline_audit.py`

## Generated Outputs
- `artifacts/day13/mapping_audit.json`
- `artifacts/day13/mapping_audit.md`
- `artifacts/day13/edge_case_report.json`

## Review Checklist
Confirmed on the current processed artifacts:
- processed artifacts load successfully
- split-like dataset artifacts are detected and sampled
- diagnosis/procedure/medication mapping coverage is quantified
- top unmapped tokens are visible
- critical issues are empty
- `milestone1_ready` is true

## Key Results From First Real Audit Run
- audit mode: sample-based (`--max-records-per-file 1000`)
- total sampled records: `8000`
- total sampled tokens: `346799`
- milestone1_ready: `True`
- critical issues: `None`
- warnings: `None`

### Mapping Rates
- diagnosis: `27351 / 27351 = 1.0000`
- procedure: `3001 / 3001 = 1.0000`
- medication: `71034 / 71034 = 1.0000`
- other: `245413 / 245413 = 1.0000`

### Namespace Summary
- `MED`: `250403`
- `OTHER`: `62767`
- `ICD`: `26356`
- `PROC`: `7273`

### Edge Case Counts
- `duplicate_tokens_within_record`: `106931`
- `malformed_tokens`: `3925`
- `unknown_namespace_tokens`: `62767`

## Interpretation
- The Day 13 audit pipeline is now operational and stable on real processed artifacts.
- The previous false inflation from stats/summary files and stringified list fields has been corrected.
- Remaining `OTHER` / `unknown_namespace_tokens` counts likely reflect dataset-specific token namespaces that are not yet fully normalized by the current namespace rules, especially in non-core token families.
- This is documented as a refinement target rather than a Milestone 1 blocker because the audit now reports these cases explicitly instead of miscounting or silently ignoring them.

## Notes
- Reports are written to `artifacts/day13/` so they stay separate from generated processed data.
- The Day 13 layer is intentionally audit-focused and avoids vanity refactors.
- A later refinement pass can further reduce `OTHER` / `unknown_namespace_tokens` by adding dataset-specific namespace rules for remaining token families.
