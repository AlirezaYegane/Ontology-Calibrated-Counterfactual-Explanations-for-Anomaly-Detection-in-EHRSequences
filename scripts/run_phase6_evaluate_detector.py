"""
scripts/run_phase6_evaluate_detector.py
========================================
Phase 6 -- Evaluate a trained detector on benchmark-v2.

Loads a run's best checkpoint, scores val + test, selects the threshold on VAL
only, applies to test, and reports ROC-AUC / AP / F1 / precision / recall with
bootstrap CIs and a per-anomaly-family breakdown. Per-record scores (MIMIC-derived)
go to the run's ignored/ subdir.

  python scripts/run_phase6_evaluate_detector.py --config configs/phase6_detector_smoke.yaml
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


def run(config_path: str, run_id: str | None = None) -> dict[str, Any]:
    import numpy as np

    from src.evaluation.calibration import apply_threshold, select_best_f1_threshold
    from src.evaluation.stats import average_precision, bootstrap_auc_ap, roc_auc
    from src.experiments.config import ExperimentConfig
    from src.experiments.eval_common import detector_scores, load_records
    from src.models.detector_unsup import UnsupervisedSequenceDetector

    cfg = ExperimentConfig.from_file(config_path)
    rid = run_id or f"{cfg.experiment_name}_seed{cfg.seed}"
    run_dir = PROJECT_ROOT / cfg.output_dir / rid
    ckpt_dir = run_dir / "checkpoints"
    if not (ckpt_dir / "detector_unsup.pt").exists():
        raise FileNotFoundError(f"No trained checkpoint at {ckpt_dir}; train first.")

    detector = UnsupervisedSequenceDetector.load(ckpt_dir, device=cfg.resolved_device())
    val = load_records(cfg.split_path("val"), cfg.sequence_key, cfg.label_key)
    test = load_records(cfg.split_path("test"), cfg.sequence_key, cfg.label_key)

    val_s = detector_scores(detector, val, cfg.batch_size)
    test_s = detector_scores(detector, test, cfg.batch_size)
    val_y = [r["label"] for r in val]
    test_y = np.array([r["label"] for r in test])

    thr = select_best_f1_threshold(val_y, val_s)  # VALIDATION only
    applied = apply_threshold(test_y, test_s, thr.threshold)
    boot = bootstrap_auc_ap(test_y, np.array(test_s), n_boot=500, seed=cfg.seed)

    # per-anomaly-family ROC-AUC (normals + that family only)
    by_family: dict[str, Any] = {}
    fams = sorted({r["anomaly_type"] for r in test if r["label"] == 1})
    for fam in fams:
        idx = [
            i for i, r in enumerate(test) if r["label"] == 0 or r["anomaly_type"] == fam
        ]
        yy = np.array([test[i]["label"] for i in idx])
        ss = np.array([test_s[i] for i in idx])
        if len(np.unique(yy)) == 2:
            by_family[fam] = {
                "n": int(len(idx)),
                "roc_auc": round(roc_auc(yy, ss), 4),
                "average_precision": round(average_precision(yy, ss), 4),
            }

    result = {
        "phase": 6,
        "run_id": rid,
        "evidence_level": cfg.evidence_level,
        "device": cfg.resolved_device(),
        "n_val": len(val),
        "n_test": len(test),
        "test_anomaly_rate": round(float(test_y.mean()), 4),
        "threshold_protocol": "selected on val (best-F1), applied to test; no test tuning",
        "test_roc_auc": round(roc_auc(test_y, np.array(test_s)), 4),
        "test_roc_auc_ci": [
            round(boot["roc_auc"]["ci_low"], 4),
            round(boot["roc_auc"]["ci_high"], 4),
        ],
        "test_average_precision": round(average_precision(test_y, np.array(test_s)), 4),
        "test_ap_ci": [
            round(boot["average_precision"]["ci_low"], 4),
            round(boot["average_precision"]["ci_high"], 4),
        ],
        "val_selected_threshold": round(thr.threshold, 6),
        "test_precision": round(applied["precision"], 4),
        "test_recall": round(applied["recall"], 4),
        "test_f1": round(applied["f1"], 4),
        "by_anomaly_family": by_family,
        "note": (
            "Smoke-scale results are diagnostic. The unsupervised next-token "
            "detector has limited signal on relational benchmark-v2 anomalies; "
            "ontology_only_real remains the strongest variant (see combined_eval)."
        ),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detector_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_md(result, run_dir / "detector_eval.md")
    with (run_dir / "detector_eval_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "ci_low", "ci_high"])
        w.writerow(["roc_auc", result["test_roc_auc"], *result["test_roc_auc_ci"]])
        w.writerow(
            [
                "average_precision",
                result["test_average_precision"],
                *result["test_ap_ci"],
            ]
        )
        w.writerow(["f1", result["test_f1"], "", ""])

    # per-record scores -> IGNORED subdir (MIMIC-derived)
    ignored = run_dir / "ignored"
    ignored.mkdir(parents=True, exist_ok=True)
    with (ignored / "per_record_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["split", "label", "anomaly_type", "s_det"])
        for r, s in zip(test, test_s):
            w.writerow(["test", r["label"], r["anomaly_type"], round(float(s), 6)])

    print(
        f"[phase6][eval] run_id={rid} evidence={cfg.evidence_level} "
        f"test_roc_auc={result['test_roc_auc']} CI={result['test_roc_auc_ci']} "
        f"AP={result['test_average_precision']} F1={result['test_f1']}"
    )
    return result


def _write_md(r: dict[str, Any], path: Path) -> None:
    md = [
        "# Phase 6 -- Detector Evaluation\n",
        f"**run:** `{r['run_id']}` | **evidence:** {r['evidence_level']} | device {r['device']}\n",
        f"- test ROC-AUC: **{r['test_roc_auc']}** CI {r['test_roc_auc_ci']}",
        f"- test AP: {r['test_average_precision']} CI {r['test_ap_ci']}",
        f"- test F1 (val-threshold {r['val_selected_threshold']}): {r['test_f1']} "
        f"(P {r['test_precision']} / R {r['test_recall']})\n",
        "## By anomaly family",
        "| family | n | ROC-AUC | AP |",
        "|---|---:|---:|---:|",
    ]
    for fam, m in r["by_anomaly_family"].items():
        md.append(f"| {fam} | {m['n']} | {m['roc_auc']} | {m['average_precision']} |")
    md.append(f"\n> {r['note']}")
    path.write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 detector evaluation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    run(args.config, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
