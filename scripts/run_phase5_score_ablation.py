"""
scripts/run_phase5_score_ablation.py
====================================
Phase 5 -- Score-variant ablation: does Sgen help the combined score?

Compares, on benchmark-v2, with bootstrap CIs + a paired-bootstrap test for the
with-vs-without-Sgen difference:
  * ontology_only_real
  * detector_only
  * combined_real_without_sgen   (w_gen = 0)
  * combined_real_with_sgen      (w_gen > 0; DIAGNOSTIC-only Sgen)
  * legacy_baseline              (ICD-prefix compute_s_ont, documented circular)

Consumes the per-record scores written by run_phase5_generative_eval.py
(artifacts/phase5/per_record_scores.csv). If that file is absent (no valid Sgen),
writes a BLOCKED ablation rather than fabricated rows.

Outputs: artifacts/phase5/score_ablation.{json,md}, score_ablation_table.csv
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

V2_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_v2"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase5"


def _seq(rec: dict[str, Any]) -> list[str]:
    s = rec.get("model_visible_sequence", rec.get("codes", []))
    return [str(t) for t in s] if isinstance(s, (list, tuple)) else []


def run(split: str, seed: int, out_dir: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    scores_path = out_dir / "per_record_scores.csv"
    if not scores_path.exists():
        report = {
            "phase": 5,
            "status": "blocked_no_sgen_scores",
            "reason": "per_record_scores.csv not found; run run_phase5_generative_eval.py first (or no valid Sgen).",
        }
        (out_dir / "score_ablation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (out_dir / "score_ablation.md").write_text(
            f"# Phase 5 Score Ablation: BLOCKED\n\n{report['reason']}\n",
            encoding="utf-8",
        )
        with (out_dir / "score_ablation_table.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            csv.writer(fh).writerow(["variant", "roc_auc"])
        return report

    from src.evaluation.stats import (
        average_precision,
        bootstrap_auc_ap,
        paired_bootstrap_diff,
        roc_auc,
    )
    from src.ontology.rules import compute_s_ont
    from src.scoring.ontology_aware import compute_calibrated_score, ScoreWeights

    df = pd.read_csv(scores_path)
    labels = df["label"].to_numpy().astype(int)
    s_gen_n = df["s_gen_norm"].to_numpy(dtype=float)
    s_ont = df["s_ont"].to_numpy(dtype=float)
    s_det = df["s_det"].to_numpy(dtype=float)
    n = len(df)

    # legacy ICD-prefix S_ont over the SAME records (recomputed; fast string match)
    records = pd.read_pickle(V2_DIR / f"{split}.pkl").to_dict(orient="records")[:n]
    legacy = []
    for r in records:
        try:
            legacy.append(float(compute_s_ont({"codes": _seq(r)})["sont"]))
        except Exception:
            legacy.append(0.0)
    legacy = np.array(legacy)

    from src.scoring.ontology_aware import normalize_sont

    s_ont_n = np.array([normalize_sont(x) for x in s_ont])
    w_no = ScoreWeights()
    w_yes = ScoreWeights(w_det=0.7, w_ont=0.3, w_gen=0.3)
    comb_no = np.array(
        [compute_calibrated_score(d, o, weights=w_no) for d, o in zip(s_det, s_ont)]
    )
    comb_yes = np.array(
        [
            compute_calibrated_score(d, o, g, weights=w_yes)
            for d, o, g in zip(s_det, s_ont, s_gen_n)
        ]
    )

    variants = {
        "ontology_only_real": s_ont_n,
        "detector_only": s_det,
        "combined_real_without_sgen": comb_no,
        "combined_real_with_sgen": comb_yes,
        "legacy_baseline": legacy,
    }
    rows = []
    for name, sc in variants.items():
        boot = bootstrap_auc_ap(labels, list(map(float, sc)), n_boot=500, seed=seed)
        rows.append(
            {
                "variant": name,
                "roc_auc": round(roc_auc(labels, sc), 4),
                "roc_auc_ci": [
                    round(boot["roc_auc"]["ci_low"], 4),
                    round(boot["roc_auc"]["ci_high"], 4),
                ],
                "average_precision": round(average_precision(labels, sc), 4),
            }
        )

    paired = paired_bootstrap_diff(
        roc_auc,
        labels,
        list(map(float, comb_yes)),
        list(map(float, comb_no)),
        n_boot=500,
        seed=seed,
    )
    delta = round(rows[3]["roc_auc"] - rows[2]["roc_auc"], 4)

    report = {
        "phase": 5,
        "status": "ablation_complete_sgen_diagnostic_only",
        "split": split,
        "n": int(n),
        "variants": rows,
        "sgen_delta_roc_auc_with_minus_without": delta,
        "paired_bootstrap_with_minus_without": {
            "observed_diff": round(paired.get("observed_diff", 0.0), 4),
            "ci": [
                round(paired.get("ci_low", 0.0), 4),
                round(paired.get("ci_high", 0.0), 4),
            ],
            "p_value": round(paired.get("p_value", 1.0), 4),
        },
        "answers": {
            "sgen_improves_roc_auc": delta > 0,
            "sgen_improves_ap": rows[3]["average_precision"]
            > rows[2]["average_precision"],
            "sgen_harms_combined": delta < 0,
            "improvement_statistically_credible": (
                delta > 0 and paired.get("p_value", 1.0) < 0.05
            ),
        },
        "note": "Sgen is DIAGNOSTIC-only (old-data, mode-collapsed checkpoint). w_gen stays 0.0 in the core.",
    }
    (out_dir / "score_ablation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_md(report, out_dir / "score_ablation.md")
    with (out_dir / "score_ablation_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "roc_auc", "roc_auc_ci", "average_precision"])
        for r in rows:
            w.writerow(
                [r["variant"], r["roc_auc"], r["roc_auc_ci"], r["average_precision"]]
            )
    return report


def _write_md(report: dict[str, Any], path: Path) -> None:
    md = [
        "# Phase 5 -- Score-Variant Ablation (benchmark-v2)\n",
        f"**Status:** `{report['status']}` | n={report['n']}\n",
        "| variant | ROC-AUC | 95% CI | AP |",
        "|---|---:|---|---:|",
    ]
    for r in report["variants"]:
        md.append(
            f"| {r['variant']} | {r['roc_auc']} | {r['roc_auc_ci']} | {r['average_precision']} |"
        )
    p = report["paired_bootstrap_with_minus_without"]
    a = report["answers"]
    md += [
        "",
        f"**Sgen ΔROC-AUC (with − without): {report['sgen_delta_roc_auc_with_minus_without']}** "
        f"(paired diff {p['observed_diff']}, CI {p['ci']}, p={p['p_value']}).",
        f"- Sgen improves ROC-AUC: **{a['sgen_improves_roc_auc']}**",
        f"- Sgen improves AP: **{a['sgen_improves_ap']}**",
        f"- Sgen harms combined: **{a['sgen_harms_combined']}**",
        f"- improvement statistically credible: **{a['improvement_statistically_credible']}**",
        f"\n> {report['note']}",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 5 score ablation.")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    report = run(args.split, args.seed, Path(args.out_dir))
    print(f"[phase5-ablation] status={report['status']}")
    for r in report.get("variants", []):
        print(
            f"[phase5-ablation]   {r['variant']}: ROC-AUC={r['roc_auc']} CI={r['roc_auc_ci']}"
        )
    if "sgen_delta_roc_auc_with_minus_without" in report:
        print(
            f"[phase5-ablation] Sgen delta={report['sgen_delta_roc_auc_with_minus_without']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
