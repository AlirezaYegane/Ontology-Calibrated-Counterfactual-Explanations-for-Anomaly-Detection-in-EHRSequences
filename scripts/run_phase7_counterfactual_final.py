"""
scripts/run_phase7_counterfactual_final.py
==========================================
Phase 7 -- Final counterfactual-repair evaluation on benchmark-v2 test anomalies.

Reuses the Phase 4 leakage-free generator + eval harness for the headline metrics,
and adds an edit-STRATEGY ablation (remove_only / replace_only / add_context_allowed
/ full_policy) via the generator's ``allowed_operations`` filter.

Per-record outputs (MIMIC-derived) are written under an IGNORED dir; only aggregate
metrics are committed.

  python scripts/run_phase7_counterfactual_final.py
  python scripts/run_phase7_counterfactual_final.py --ablation-sample 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_ONT = PROJECT_ROOT / "ontologies" / "processed"
V2 = PROJECT_ROOT / "data" / "processed" / "benchmark_v2"
OUT = PROJECT_ROOT / "artifacts" / "phase7"
IGNORED = OUT / "ignored"


def _real_scorer():
    from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

    return OntologyAwareScorer.from_processed_dir(
        PROCESSED_ONT, ontology_mode="real", weights=ScoreWeights()
    )


def run(
    max_records: int = 0, ablation_sample: int = 300, seed: int = 42
) -> dict[str, Any]:
    import pandas as pd

    from scripts.run_phase6_combined_eval import PROCESSED_ONT as _  # noqa: F401 (path setup)
    from scripts.run_phase4_counterfactual_eval import evaluate_records
    from src.explanations.counterfactual import generate_counterfactual

    rows = pd.read_pickle(V2 / "test.pkl").to_dict(orient="records")
    anomalies = [r for r in rows if int(r.get("label", 0)) == 1]
    scorer = _real_scorer()
    index = scorer.engine.index

    # ---- headline full-policy metrics (Phase 4 harness; per-record -> ignored) ----
    IGNORED.mkdir(parents=True, exist_ok=True)
    summary = evaluate_records(
        anomalies,
        scorer,
        index,
        detector=None,
        detector_status="disabled",
        split="test",
        seed=seed,
        max_edits=3,
        beam_size=20,
        max_records=max_records,
        out_dir=IGNORED / "cf_full",
    )

    # ---- edit-strategy ablation on a capped sample of flagged anomalies ----
    def _mv(r):
        s = r.get("model_visible_sequence", [])
        return {
            "model_visible_sequence": [str(t) for t in s]
            if isinstance(s, (list, tuple))
            else [],
            "gender": r.get("gender"),
            "age_group": r.get("age_group"),
        }

    flagged = []
    for r in anomalies:
        res = generate_counterfactual(_mv(r), scorer, index, max_edits=3, seed=seed)
        if res.s_ont_before > 0:
            flagged.append(r)
        if len(flagged) >= ablation_sample:
            break

    policies = {
        "remove_only": ("remove",),
        "replace_only": ("replace",),
        "add_context_allowed": ("remove", "add"),
        "full_policy": None,
    }
    ablation = {}
    for name, ops in policies.items():
        n_valid, n_edits = 0, []
        for r in flagged:
            res = generate_counterfactual(
                _mv(r), scorer, index, max_edits=3, seed=seed, allowed_operations=ops
            )
            if res.validity:
                n_valid += 1
                n_edits.append(res.num_edits)
        ablation[name] = {
            "n": len(flagged),
            "valid": n_valid,
            "success_rate": round(n_valid / len(flagged), 4) if flagged else 0.0,
            "mean_edits": round(sum(n_edits) / len(n_edits), 3) if n_edits else None,
        }

    result = {
        "phase": 7,
        "benchmark": "benchmark-v2 test anomalies",
        "generator": "leakage_free_ontology_counterfactual (Phase 4)",
        "repair_attempted_count": summary["repair_attempted_count"],
        "ontology_flagged_count": summary["ontology_flagged_count"],
        "repair_success_count": summary["repair_success_count"],
        "repair_success_rate_overall": summary["repair_success_rate"],
        "repair_success_rate_among_flagged": summary[
            "repair_success_rate_among_flagged"
        ],
        "validity_rate": summary["validity_rate"],
        "mean_delta_s_ont": summary["mean_delta_s_ont"],
        "mean_delta_s_cal": summary["mean_delta_s_cal"],
        "mean_delta_s_det": summary["mean_delta_s_det"],
        "mean_num_edits": summary["mean_num_edits"],
        "median_num_edits": summary["median_num_edits"],
        "edit_type_distribution": summary["edit_operation_counts"],
        "failure_reason_counts": summary["failure_reason_counts"],
        "success_by_anomaly_type": summary["success_by_anomaly_type"],
        "success_by_rule_type": summary["success_by_rule_type"],
        "edit_strategy_ablation": ablation,
        "edit_strategy_ablation_note": f"on a capped sample of {len(flagged)} ontology-flagged anomalies",
        "leakage_note": "generator read model-visible rows only; no hidden/audit metadata.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "counterfactual_final.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_md(result, OUT / "counterfactual_final.md")
    with (OUT / "counterfactual_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k in (
            "repair_attempted_count",
            "ontology_flagged_count",
            "repair_success_count",
            "repair_success_rate_overall",
            "repair_success_rate_among_flagged",
            "mean_delta_s_ont",
            "mean_num_edits",
            "median_num_edits",
        ):
            w.writerow([k, result[k]])
        w.writerow([])
        w.writerow(["edit_strategy", "success_rate", "mean_edits"])
        for name, m in ablation.items():
            w.writerow([name, m["success_rate"], m["mean_edits"]])

    print(
        f"[phase7][cf] attempted={result['repair_attempted_count']} flagged={result['ontology_flagged_count']} "
        f"success_among_flagged={result['repair_success_rate_among_flagged']} mean_edits={result['mean_num_edits']}"
    )
    for name, m in ablation.items():
        print(
            f"[phase7][cf][ablation] {name}: success={m['success_rate']} mean_edits={m['mean_edits']}"
        )
    return result


def _write_md(r: dict[str, Any], path: Path) -> None:
    md = [
        "# Phase 7 -- Counterfactual Final (benchmark-v2 test)\n",
        f"- attempted: **{r['repair_attempted_count']}** (ontology-flagged {r['ontology_flagged_count']})",
        f"- repair success among flagged: **{r['repair_success_rate_among_flagged']}** "
        f"(overall {r['repair_success_rate_overall']})",
        f"- mean delta S_ont: **{r['mean_delta_s_ont']}** | mean edits {r['mean_num_edits']} "
        f"(median {r['median_num_edits']}) | edit ops {r['edit_type_distribution']}\n",
        "## Success by rule type",
        "| rule_kind | n | success | rate |",
        "|---|---:|---:|---:|",
    ]
    for k, v in r["success_by_rule_type"].items():
        md.append(f"| {k} | {v['n']} | {v['success']} | {v['rate']} |")
    md.append(f"\n## Edit-strategy ablation ({r['edit_strategy_ablation_note']})")
    md.append("| strategy | success_rate | mean_edits |")
    md.append("|---|---:|---:|")
    for name, m in r["edit_strategy_ablation"].items():
        md.append(f"| {name} | {m['success_rate']} | {m['mean_edits']} |")
    md.append(f"\n> {r['leakage_note']}")
    path.write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 counterfactual final.")
    ap.add_argument(
        "--max-records", type=int, default=0, help="0 = all test anomalies."
    )
    ap.add_argument("--ablation-sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    run(args.max_records, args.ablation_sample, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
