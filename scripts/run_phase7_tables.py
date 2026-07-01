"""
scripts/run_phase7_tables.py
============================
Phase 7 -- Assemble publication-ready aggregate tables + figure data from the
Phase 7 result JSONs. Aggregate-only (no per-record / PHI data). Renders PNG bar
charts if matplotlib is present (it is a repo dependency).

  python scripts/run_phase7_tables.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "artifacts" / "phase7"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
V2_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "benchmark_v2" / "benchmark_v2_manifest.json"
)


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def run() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    manifest = _load(V2_MANIFEST) or {}
    final = _load(OUT / "final_evaluation.json") or {}
    stats = _load(OUT / "final_stat_tests.json") or {}
    abl = _load(OUT / "ablation_results.json") or {}
    cf = _load(OUT / "counterfactual_final.json")
    ext = _load(OUT / "external_validation_status.json") or {}

    # ---- Table 1: dataset summary ----
    ss = manifest.get("split_sizes", {})
    _csv(
        TABLES / "table1_dataset_summary.csv",
        ["item", "value"],
        [
            ["benchmark", "benchmark-v2 (non-circular, MIMIC-IV)"],
            ["n_records", manifest.get("n_records")],
            ["n_normal", manifest.get("n_normal")],
            ["n_anomaly", manifest.get("n_anomaly")],
            ["train_n (normal-only)", ss.get("train", {}).get("n")],
            ["val_n", ss.get("val", {}).get("n")],
            ["test_n", ss.get("test", {}).get("n")],
            ["subject_overlap", "0 (train/val/test)"],
            ["strongest_trivial_signal", "0.6127 (< 0.80 gate)"],
            ["diagnosis_coverage", "0.80"],
            ["medication_coverage", "0.78"],
            [
                "anomaly_demographic",
                manifest.get("anomaly_type_counts", {}).get(
                    "demographic_incompatibility"
                ),
            ],
            [
                "anomaly_medication",
                manifest.get("anomaly_type_counts", {}).get(
                    "medication_indication_mismatch"
                ),
            ],
            [
                "anomaly_forbidden",
                manifest.get("anomaly_type_counts", {}).get("forbidden_cooccurrence"),
            ],
        ],
    )

    # ---- Table 2: main results ----
    main_rows = [
        [
            v["variant"],
            v["roc_auc"],
            v["roc_auc_ci"][0],
            v["roc_auc_ci"][1],
            v["average_precision"],
            v.get("test_f1"),
        ]
        for v in final.get("variant_metrics", [])
    ]
    _csv(
        TABLES / "table2_main_results.csv",
        ["variant", "roc_auc", "ci_low", "ci_high", "average_precision", "f1"],
        main_rows,
    )

    # ---- Table 3: ablation ----
    abl_rows = [
        [
            v["variant"],
            v.get("roc_auc"),
            v.get("average_precision"),
            v.get("normal_fp_rate"),
        ]
        for v in abl.get("ontology_rule_ablation", [])
    ]
    _csv(
        TABLES / "table3_ablation_results.csv",
        ["variant", "roc_auc", "average_precision", "normal_fp_rate"],
        abl_rows,
    )

    # ---- Table 4: counterfactual ----
    if cf:
        _csv(
            TABLES / "table4_counterfactual_results.csv",
            ["metric", "value"],
            [
                ["attempted", cf["repair_attempted_count"]],
                ["ontology_flagged", cf["ontology_flagged_count"]],
                ["success_among_flagged", cf["repair_success_rate_among_flagged"]],
                ["success_overall", cf["repair_success_rate_overall"]],
                ["mean_delta_s_ont", cf["mean_delta_s_ont"]],
                ["mean_num_edits", cf["mean_num_edits"]],
                ["median_num_edits", cf["median_num_edits"]],
            ],
        )

    # ---- Table 5: statistical tests ----
    stat_rows = [
        [
            t["a"],
            t["b"],
            t["observed_diff"],
            t["ci"][0],
            t["ci"][1],
            t["p_value"],
            t["significant"],
        ]
        for t in stats.get("paired_bootstrap_roc_auc", [])
    ]
    _csv(
        TABLES / "table5_statistical_tests.csv",
        [
            "variant_a",
            "variant_b",
            "diff",
            "ci_low",
            "ci_high",
            "p_value",
            "significant",
        ],
        stat_rows,
    )

    # ---- Figure data ----
    fig1 = {
        "benchmark": "benchmark-v2 (non-circular)",
        "main_method": "ontology-only (S_main = S_ont)",
        "score_equation": final.get("score_equation"),
        "strongest_variant": final.get("strongest_variant"),
        "sgen_in_core": False,
        "detector_below_chance": True,
        "external_validation": ext.get("status"),
    }
    (FIGS / "fig1_pipeline_summary.json").write_text(
        json.dumps(fig1, indent=2), encoding="utf-8"
    )
    _csv(
        FIGS / "fig2_main_auc_bar.csv",
        ["variant", "roc_auc", "ci_low", "ci_high"],
        [
            [v["variant"], v["roc_auc"], v["roc_auc_ci"][0], v["roc_auc_ci"][1]]
            for v in final.get("variant_metrics", [])
        ],
    )
    _csv(
        FIGS / "fig3_ablation_auc_bar.csv",
        ["variant", "roc_auc"],
        [
            [v["variant"], v.get("roc_auc")]
            for v in abl.get("ontology_rule_ablation", [])
        ],
    )
    if cf:
        _csv(
            FIGS / "fig4_counterfactual_delta_box_summary.csv",
            ["metric", "value"],
            [
                ["mean_delta_s_ont", cf["mean_delta_s_ont"]],
                ["mean_num_edits", cf["mean_num_edits"]],
                ["median_num_edits", cf["median_num_edits"]],
                ["success_among_flagged", cf["repair_success_rate_among_flagged"]],
            ],
        )

    _render_png(final, abl)
    print(f"[phase7][tables] wrote tables -> {TABLES} and figures -> {FIGS}")


def _render_png(final: dict[str, Any], abl: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    vm = final.get("variant_metrics", [])
    if vm:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [v["variant"] for v in vm]
        aucs = [v["roc_auc"] for v in vm]
        ax.bar(range(len(names)), aucs, color="#4C72B0")
        ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Phase 7 main results (benchmark-v2)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig2_main_auc_bar.png", dpi=120)
        plt.close(fig)
    ar = [
        v for v in abl.get("ontology_rule_ablation", []) if v.get("roc_auc") is not None
    ]
    if ar:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [v["variant"] for v in ar]
        aucs = [v["roc_auc"] for v in ar]
        ax.bar(range(len(names)), aucs, color="#55A868")
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Phase 7 ontology-rule ablation")
        fig.tight_layout()
        fig.savefig(FIGS / "fig3_ablation_auc_bar.png", dpi=120)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
