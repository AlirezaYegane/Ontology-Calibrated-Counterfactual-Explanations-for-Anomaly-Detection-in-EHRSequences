# Day 33 — Ontology-Regularized Diffusion

## Status
Complete.

## Goal
Add a lightweight ontology-consistency regularization signal to the diffusion baseline.

## What was fixed
The first Day 33 run executed successfully, but the ontology metrics were not meaningful because the vocabulary path was missing. That caused decoded token ids to be treated as unknown/other.

The fixed run explicitly uses the aligned diffusion vocabulary:

`outputs/detector/day20_supervised/run_luxury/vocab.json`

This vocabulary has 47,010 tokens and matches the diffusion artifact vocabulary size used in the Day 31/32 diffusion data.

## Final fixed-run results
- Best epoch: 6
- Train loss: 0.6738
- Train diffusion loss: 0.6733
- Train ontology loss: 0.0049
- Validation loss: 0.6130
- Validation diffusion loss: 0.6126
- Validation ontology loss: 0.0041
- Validation violation rate: 0.0203
- Generated violation rate: 0.0000
- Generated unknown/other rate: 0.0000

## What was implemented
- Ontology-regularized diffusion fine-tuning script
- Vocabulary diagnostic utility
- Lightweight ontology-consistency loss
- Proxy monitoring for:
  - medication_without_diagnosis
  - procedure_without_diagnosis
  - unknown_or_other_token_mass
- Generated violation-rate reporting
- Best checkpoint selection and JSON summary artifacts

## Main outputs
- `scripts/run_day33_ontology_regularization.py`
- `scripts/build_day33_vocab.py`
- `config/diffusion/day33_ontology_regularized.yaml`
- `artifacts/day33/mimiciv_val_vocab.json`
- `artifacts/day33/mimiciv_val_vocab.summary.json`
- `artifacts/day33/day33_ontology_regularization_report.json`
- `outputs/diffusion/day33_ontology_regularized_fixed/best.pt`
- `outputs/diffusion/day33_ontology_regularized_fixed/last.pt`
- `outputs/diffusion/day33_ontology_regularized_fixed/metrics.jsonl`

## Important limitation
This is still a conservative ontology-regularization pass. Full SNOMED graph-distance loss, reliable demographic metadata rules, and counterfactual repair search are intentionally deferred to later integration/evaluation days.
