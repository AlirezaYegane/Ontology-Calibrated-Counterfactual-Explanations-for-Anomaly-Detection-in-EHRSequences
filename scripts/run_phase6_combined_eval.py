"""
scripts/run_phase6_combined_eval.py
====================================
Phase 6 -- Sgen-free combined-score evaluation on benchmark-v2.

Compares detector_only / ontology_only_real / combined_real_without_sgen /
legacy_baseline with bootstrap CIs and paired-bootstrap differences. Sgen is NOT
included (w_gen = 0). Threshold/normalization fit on val, applied to test.

  python scripts/run_phase6_combined_eval.py --config configs/phase6_detector_smoke.yaml
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


def run(config_path: str, run_id: str | None = None) -> dict[str, Any]:
    import numpy as np

    from src.evaluation.stats import (
        average_precision,
        bootstrap_auc_ap,
        paired_bootstrap_diff,
        roc_auc,
    )
    from src.experiments.config import ExperimentConfig
    from src.experiments.eval_common import (
        combined_scores,
        detector_scores,
        load_records,
        minmax_apply,
        minmax_fit,
        ontology_scores,
    )
    from src.models.detector_unsup import UnsupervisedSequenceDetector
    from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

    cfg = ExperimentConfig.from_file(config_path)
    rid = run_id or f"{cfg.experiment_name}_seed{cfg.seed}"
    run_dir = PROJECT_ROOT / cfg.output_dir / rid
    ckpt_dir = run_dir / "checkpoints"
    if not (ckpt_dir / "detector_unsup.pt").exists():
        raise FileNotFoundError(f"No trained checkpoint at {ckpt_dir}; train first.")

    detector = UnsupervisedSequenceDetector.load(ckpt_dir, device=cfg.resolved_device())
    real = OntologyAwareScorer.from_processed_dir(
        PROCESSED_ONT, ontology_mode="real", weights=ScoreWeights()
    )
    legacy = OntologyAwareScorer(ontology_mode="legacy", weights=ScoreWeights())

    val = load_records(cfg.split_path("val"), cfg.sequence_key, cfg.label_key)
    test = load_records(cfg.split_path("test"), cfg.sequence_key, cfg.label_key)
    test_y = np.array([r["label"] for r in test])

    # detector (normalize via val min-max, applied to test)
    val_det = detector_scores(detector, val, cfg.batch_size)
    test_det = detector_scores(detector, test, cfg.batch_size)
    lo, hi = minmax_fit(val_det)
    test_det_n = minmax_apply(test_det, lo, hi)

    test_ont_real = ontology_scores(real, test)
    test_ont_legacy = ontology_scores(legacy, test)
    test_comb = combined_scores(test_det_n, test_ont_real, cfg.scoring_weights)
    test_comb_legacy = combined_scores(test_det_n, test_ont_legacy, cfg.scoring_weights)

    variants = {
        "detector_only": test_det_n,
        "ontology_only_real": test_ont_real,
        "combined_real_without_sgen": test_comb,
        "legacy_baseline": test_comb_legacy,
    }
    rows = []
    for name, scores in variants.items():
        s = np.array(list(map(float, scores)))
        boot = bootstrap_auc_ap(test_y, s, n_boot=500, seed=cfg.seed)
        rows.append(
            {
                "variant": name,
                "roc_auc": round(roc_auc(test_y, s), 4),
                "roc_auc_ci": [
                    round(boot["roc_auc"]["ci_low"], 4),
                    round(boot["roc_auc"]["ci_high"], 4),
                ],
                "average_precision": round(average_precision(test_y, s), 4),
                "ap_ci": [
                    round(boot["average_precision"]["ci_low"], 4),
                    round(boot["average_precision"]["ci_high"], 4),
                ],
            }
        )

    def _pair(a, b):
        d = paired_bootstrap_diff(
            roc_auc, test_y, np.array(a), np.array(b), n_boot=500, seed=cfg.seed
        )
        return {
            "observed_diff": round(d["observed_diff"], 4),
            "ci": [round(d["ci_low"], 4), round(d["ci_high"], 4)],
            "p_value": round(d["p_value"], 4),
        }

    paired = {
        "combined_vs_detector_only": _pair(test_comb, test_det_n),
        "combined_vs_ontology_only": _pair(test_comb, test_ont_real),
        "ontology_only_vs_legacy": _pair(test_ont_real, test_comb_legacy),
        "detector_vs_ontology_only": _pair(test_det_n, test_ont_real),
    }
    auc = {r["variant"]: r["roc_auc"] for r in rows}
    answers = {
        "detector_improves_over_ontology_only": auc["detector_only"]
        > auc["ontology_only_real"],
        "combined_improves_over_ontology_only": (
            auc["combined_real_without_sgen"] > auc["ontology_only_real"]
            and paired["combined_vs_ontology_only"]["observed_diff"] > 0
            and paired["combined_vs_ontology_only"]["ci"][0] > 0
        ),
        "combined_improves_over_detector_only": auc["combined_real_without_sgen"]
        > auc["detector_only"],
    }

    result = {
        "phase": 6,
        "run_id": rid,
        "evidence_level": cfg.evidence_level,
        "sgen_included": False,
        "w_gen": 0.0,
        "n_test": len(test),
        "threshold_protocol": "detector normalized on val, applied to test; no test tuning",
        "variant_metrics": rows,
        "paired_bootstrap_roc_auc": paired,
        "answers": answers,
        "note": (
            "Sgen excluded (w_gen=0). If the detector underperforms, the main claim "
            "stays ontology-centered (ontology_only_real is the strongest variant)."
        ),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "combined_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_md(result, run_dir / "combined_eval.md")
    with (run_dir / "combined_eval_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "variant",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "average_precision",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["variant"],
                    r["roc_auc"],
                    r["roc_auc_ci"][0],
                    r["roc_auc_ci"][1],
                    r["average_precision"],
                ]
            )

    print(f"[phase6][combined] run_id={rid}")
    for r in rows:
        print(
            f"[phase6][combined]   {r['variant']}: ROC-AUC={r['roc_auc']} CI={r['roc_auc_ci']} AP={r['average_precision']}"
        )
    print(f"[phase6][combined] answers={answers}")
    return result


def _write_md(r: dict[str, Any], path: Path) -> None:
    md = [
        "# Phase 6 -- Combined Score Evaluation (Sgen-free)\n",
        f"**run:** `{r['run_id']}` | evidence {r['evidence_level']} | w_gen={r['w_gen']} (Sgen excluded)\n",
        "| variant | ROC-AUC | 95% CI | AP |",
        "|---|---:|---|---:|",
    ]
    for v in r["variant_metrics"]:
        md.append(
            f"| {v['variant']} | {v['roc_auc']} | {v['roc_auc_ci']} | {v['average_precision']} |"
        )
    md.append("\n## Paired bootstrap (ROC-AUC diff)")
    md.append("| comparison | Δ | 95% CI | p |")
    md.append("|---|---:|---|---:|")
    for k, v in r["paired_bootstrap_roc_auc"].items():
        md.append(f"| {k} | {v['observed_diff']} | {v['ci']} | {v['p_value']} |")
    md.append("\n## Answers")
    for k, v in r["answers"].items():
        md.append(f"- {k}: **{v}**")
    md.append(f"\n> {r['note']}")
    path.write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 6 combined-score evaluation (Sgen-free)."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    run(args.config, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
