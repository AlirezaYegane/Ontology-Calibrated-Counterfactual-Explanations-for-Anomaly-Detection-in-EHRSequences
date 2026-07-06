# Phase 8 — Repository Finalization

Phase 8 finalizes the repository for the paper. It runs **no new science**: it packages the
existing MIMIC-IV benchmark-v2 evidence into a humanized README, a complete paper, a
reproducibility package, and a final claim ledger.

## Contents of this directory

| File | Purpose |
|---|---|
| `phase8_summary.json` | machine-readable Phase 8 summary + preserved final results |
| `artifact_manifest.json` / `.md` | safe aggregate artifact manifest (what is committed vs git-ignored) |
| `paper_asset_index.json` / `.md` | index of paper tables and figures |
| `README.md` | this file |

## Final status

`phase8_complete_external_validation_deferred`

## Scientific position (unchanged from Phase 7)

- Final method is **ontology-only** (`S_main = S_ont`); ROC-AUC **0.7881**, significantly
  above legacy **0.7358** (+0.052, p ≈ 0).
- The full-scale detector is **below chance** (0.4525) and **non-additive** (combining hurts,
  −0.085, p ≈ 0) — a transparent negative result.
- `Sgen` / diffusion is **removed from the core** (`w_gen = 0`).
- Counterfactual repair is **leakage-free** and **89.99% effective among ontology-flagged**
  anomalies (median 1 edit).
- External validation on eICU is **blocked** by schema mismatch — future work.

## Where the evidence lives

The authoritative final evidence is in [`../phase7/`](../phase7/). The paper is in
[`../../docs/paper/final_manuscript.md`](../../docs/paper/final_manuscript.md); the claim
ledger is [`../../docs/paper/final_claims_matrix.md`](../../docs/paper/final_claims_matrix.md).

## Remaining work

External validation requires a future APACHE→ICD/SNOMED crosswalk or another
ICD/SNOMED/RxNorm-compatible external dataset.
