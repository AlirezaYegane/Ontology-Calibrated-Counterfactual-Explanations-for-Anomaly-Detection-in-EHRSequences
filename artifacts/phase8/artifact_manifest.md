# Phase 8 — Safe Aggregate Artifact Manifest

Machine-readable version: [`artifact_manifest.json`](artifact_manifest.json).

**Safety statement.** Every artifact listed here is aggregate-level (counts, rates, metrics,
confidence intervals, or small sanitized examples) or code-facing. **No** raw or processed
patient records, ontology dumps, vocabularies, model checkpoints, or per-record score dumps
are committed. Restricted patterns are git-ignored.

## Excluded from git (enforced by `.gitignore`)

- `data/processed/` — patient-derived sequences and benchmark-v2 split `.pkl` files
- `ontologies/raw/`, `ontologies/processed/` — licensed SNOMED/UMLS/RxNorm content
- `*.pt` / `*.pth` / `*.ckpt` — model checkpoints
- run `vocab/` folders — MIMIC-derived vocabularies
- `artifacts/**/per_record*`, `artifacts/**/ignored/` — per-record score dumps
- `*.parquet`, `*.zip`

## Committed aggregate artifacts by phase

| Phase | Role | Representative artifacts | Content |
|---|---|---|---|
| phase0 | Claim reset | `phase0_summary.json` | claim/contribution reset (text/JSON) |
| phase1 | Old-benchmark audit | `phase1_summary.json` | triviality/circularity audit |
| phase1b | benchmark-v2 | `phase1b_summary.json` | design, split sizes, overlap 0, gate 0.6127 |
| phase2 | Ontology architecture | `coverage_report.json`, `unmapped_codes.csv` | coverage counts; token strings only |
| phase2b | UMLS/SNOMED/RxNorm | `coverage_report.json`, `raw_asset_inventory.json` | coverage + asset counts/hashes |
| phase2b_fix | Coverage repair | `coverage_report.json`, `phase2b_fix_summary.json` | final coverage 0.80 / 0.78 |
| phase3 | Detector/scoring | `phase3_scoring_eval.json`, `score_variant_table.csv` | scoring eval + CIs |
| phase3b | Rule authoring | `phase3b_scoring_eval.json`, `phase3b_summary.json` | rule coverage + scoring before/after |
| phase4 | Counterfactual repair | `counterfactual_summary.json`, `counterfactual_results.jsonl` | summary + 30-record sanitized token-edit sample (synthetic anomalies; no patient IDs) |
| phase5 | Sgen gate | `phase5_summary.json`, `sgen_decision.json` | Sgen = removed-from-core evidence |
| phase6 | Training infra | `phase6_summary.json`, `runs/*/detector_eval.json`, `runs/*/train_metrics.jsonl` | per-run + per-epoch aggregate metrics |
| phase7 | Final evaluation | `final_evaluation.json`, `final_stat_tests.json`, `ablation_results.json`, `counterfactual_final.json`, `tables/`, `figures/` | **the paper's evidence base** |
| phase8 | Finalization | `phase8_summary.json`, `artifact_manifest.json`, `paper_asset_index.json` | manifest + index; no data |

## Notes

- The **authoritative final evidence** is `artifacts/phase7/`; the paper cites it directly.
- `artifacts/phase4/counterfactual_results.jsonl` and `failure_cases.jsonl` hold a small
  illustrative sample (30 records) of model-visible token edits on **synthetic** benchmark-v2
  anomalies — code tokens only, no patient identifiers.
- `artifacts/phase6/runs/*/train_metrics.jsonl` holds per-epoch loss/AUC only.
- The full claim ledger is [`docs/paper/final_claims_matrix.md`](../../docs/paper/final_claims_matrix.md).
