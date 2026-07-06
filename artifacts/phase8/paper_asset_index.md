# Phase 8 — Paper Asset Index

Human-readable index of the paper-facing tables and figures. Machine-readable version:
[`paper_asset_index.json`](paper_asset_index.json). Narrative index for the manuscript:
[`docs/paper/tables_and_figures.md`](../../docs/paper/tables_and_figures.md).

All assets are **aggregate-level only**; per-record score dumps behind them are git-ignored.

## Tables (`artifacts/phase7/tables/`)

| Paper label | File | Contents |
|---|---|---|
| Table 1 | `table1_dataset_summary.csv` | benchmark-v2 counts, splits, subject overlap, trivial-signal gate, coverage |
| Table 2 | `table2_main_results.csv` | ROC-AUC + CI, AP, F1 for the four variants |
| Table 3 | `table3_ablation_results.csv` | rule-family ablation |
| Table 4 | `table4_counterfactual_results.csv` | counterfactual repair summary |
| Table 5 | `table5_statistical_tests.csv` | paired-bootstrap tests |

## Figures (`artifacts/phase7/figures/`)

| Paper label | File | Contents |
|---|---|---|
| Figure 1 | `fig1_pipeline_summary.json` | pipeline/scope summary data |
| Figure 2 | `fig2_main_auc_bar.png` (+ `.csv`) | main ROC-AUC with CIs |
| Figure 3 | `fig3_ablation_auc_bar.png` (+ `.csv`) | rule-family ablation |
| Figure 4 | `fig4_counterfactual_delta_box_summary.csv` | repair ΔS_ont / edits / success |

## Regeneration

```bash
python scripts/run_phase7_final_evaluation.py
python scripts/run_phase7_ablations.py
python scripts/run_phase7_counterfactual_final.py
python scripts/run_phase7_tables.py
```
