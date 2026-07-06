# Artifact Manifest (interpretation guide)

This page explains how to read the committed artifacts. The canonical, machine-readable
manifest is [`artifacts/phase8/artifact_manifest.json`](../../artifacts/phase8/artifact_manifest.json)
(human table: [`artifacts/phase8/artifact_manifest.md`](../../artifacts/phase8/artifact_manifest.md)).

> Historical note: an earlier day-level manifest lives at
> [`docs/artifact_manifest.md`](../artifact_manifest.md). The Phase 8 manifest above is the
> current, authoritative one for the final paper.

## The one directory that matters most

**`artifacts/phase7/`** is the paper's evidence base. If you only read one thing:

| File | What it answers |
|---|---|
| `final_evaluation.json` | main results table (ROC-AUC, CI, AP, F1) for the four variants |
| `final_stat_tests.json` | paired-bootstrap significance (ontology > legacy; combined < ontology) |
| `ablation_results.json` | rule-family and score-component ablations |
| `counterfactual_final.json` | leakage-free repair results (89.99% among flagged, median 1 edit) |
| `external_validation_status.json` | why eICU is blocked (schema mismatch, 0/500 map) |
| `final_claims_decision.json` | the authoritative claim decisions |
| `tables/table1..5.csv`, `figures/*` | paper-ready tables and figure data |

## How to sanity-check a headline number

Each README/manuscript number cites an artifact. For example, "ontology 0.7881 > legacy
0.7358, +0.052, p ≈ 0":

```bash
python - <<'PY'
import json
d = json.load(open("artifacts/phase7/final_evaluation.json"))
print({v["variant"]: v["roc_auc"] for v in d["variant_metrics"]})
s = json.load(open("artifacts/phase7/final_stat_tests.json"))
print(s["paired_bootstrap_roc_auc"][0])   # ontology vs legacy
PY
```

## Safety

Everything committed is aggregate. See the safety statement and the git-ignored exclusion
list in [`artifacts/phase8/artifact_manifest.md`](../../artifacts/phase8/artifact_manifest.md).
Per-record scores, checkpoints, vocabularies, benchmark splits, and ontology dumps are not in
git.
