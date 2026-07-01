"""
scripts/run_phase7_final_evaluation.py
======================================
Phase 7 -- Final aggregate evaluation on benchmark-v2 (the FINAL benchmark).

Recomputes the main-method score variants from the Phase 6 full-scale detector
checkpoint + the real/legacy ontology scorers, with bootstrap CIs, validation-only
threshold calibration, and paired-bootstrap significance tests. Sgen is EXCLUDED
(w_gen=0); a diagnostic Sgen row is referenced from Phase 5, not recomputed.

Variants:
  ontology_only_real          -- S_ont from the Phase 3b real rule packs
  detector_only_full          -- Phase 6 full-scale unsupervised detector
  combined_real_without_sgen  -- (w_det*S_det + w_ont*S_ont')/(w_det+w_ont)
  legacy_baseline             -- legacy ICD-prefix ontology (pure), the "legacy rules" baseline

Only aggregate outputs are written; per-record scores go to an ignored/ dir.

  python scripts/run_phase7_final_evaluation.py
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
OUT = PROJECT_ROOT / "artifacts" / "phase7"
DEFAULT_CONFIG = "configs/phase6_detector_full.yaml"
DEFAULT_RUN = "phase6_detector_full_gpu"


def run(config_path: str = DEFAULT_CONFIG, run_id: str = DEFAULT_RUN) -> dict[str, Any]:
    import numpy as np

    from src.evaluation.calibration import apply_threshold, select_best_f1_threshold
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
    ckpt = PROJECT_ROOT / cfg.output_dir / run_id / "checkpoints"
    detector_available = (ckpt / "detector_unsup.pt").exists()

    real = OntologyAwareScorer.from_processed_dir(
        PROCESSED_ONT, ontology_mode="real", weights=ScoreWeights()
    )
    legacy = OntologyAwareScorer(ontology_mode="legacy", weights=ScoreWeights())
    val = load_records(cfg.split_path("val"), cfg.sequence_key, cfg.label_key)
    test = load_records(cfg.split_path("test"), cfg.sequence_key, cfg.label_key)
    val_y = [r["label"] for r in val]
    test_y = np.array([r["label"] for r in test])

    test_ont_real = ontology_scores(real, test)
    val_ont_real = ontology_scores(real, val)
    test_ont_legacy = ontology_scores(legacy, test)

    variant_scores: dict[str, tuple[list[float], list[float]]] = {
        "ontology_only_real": (val_ont_real, test_ont_real),
        "legacy_baseline": (ontology_scores(legacy, val), test_ont_legacy),
    }
    if detector_available:
        detector = UnsupervisedSequenceDetector.load(ckpt, device=cfg.resolved_device())
        val_det = detector_scores(detector, val, cfg.batch_size)
        test_det = detector_scores(detector, test, cfg.batch_size)
        lo, hi = minmax_fit(val_det)
        val_det_n = minmax_apply(val_det, lo, hi)
        test_det_n = minmax_apply(test_det, lo, hi)
        variant_scores["detector_only_full"] = (val_det_n, test_det_n)
        variant_scores["combined_real_without_sgen"] = (
            combined_scores(val_det_n, val_ont_real, cfg.scoring_weights),
            combined_scores(test_det_n, test_ont_real, cfg.scoring_weights),
        )

    rows: list[dict[str, Any]] = []
    for name, (vsc, tsc) in variant_scores.items():
        s = np.array(list(map(float, tsc)))
        boot = bootstrap_auc_ap(test_y, s, n_boot=1000, seed=cfg.seed)
        thr = select_best_f1_threshold(val_y, vsc)  # VAL only
        applied = apply_threshold(test_y, list(map(float, tsc)), thr.threshold)
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
                "val_threshold": round(thr.threshold, 6),
                "test_precision": round(applied["precision"], 4),
                "test_recall": round(applied["recall"], 4),
                "test_f1": round(applied["f1"], 4),
            }
        )

    def _pair(a_name, b_name):
        _, a = variant_scores[a_name]
        _, b = variant_scores[b_name]
        d = paired_bootstrap_diff(
            roc_auc,
            test_y,
            np.array(a, dtype=float),
            np.array(b, dtype=float),
            n_boot=1000,
            seed=cfg.seed,
        )
        return {
            "a": a_name,
            "b": b_name,
            "observed_diff": round(d["observed_diff"], 4),
            "ci": [round(d["ci_low"], 4), round(d["ci_high"], 4)],
            "p_value": round(d["p_value"], 4),
            "significant": bool(d["ci_low"] > 0 or d["ci_high"] < 0),
        }

    pairs = [("ontology_only_real", "legacy_baseline")]
    if detector_available:
        pairs += [
            ("ontology_only_real", "detector_only_full"),
            ("combined_real_without_sgen", "ontology_only_real"),
            ("combined_real_without_sgen", "detector_only_full"),
            ("combined_real_without_sgen", "legacy_baseline"),
        ]
    stat_tests = [_pair(a, b) for a, b in pairs]

    auc = {r["variant"]: r["roc_auc"] for r in rows}
    best = max(auc, key=auc.get)
    result = {
        "phase": 7,
        "benchmark": "benchmark-v2 (non-circular; FINAL benchmark)",
        "old_circular_benchmark_used_as_final_evidence": False,
        "detector_run": run_id if detector_available else None,
        "detector_available": detector_available,
        "sgen_included_in_core": False,
        "w_gen": 0.0,
        "score_equation": "S_cal = (w_det*S_det + w_ont*S_ont') / (w_det + w_ont), w_gen=0",
        "n_val": len(val),
        "n_test": len(test),
        "test_anomaly_rate": round(float(test_y.mean()), 4),
        "threshold_protocol": "best-F1 threshold selected on VAL, applied to TEST; no test tuning",
        "variant_metrics": rows,
        "strongest_variant": best,
        "sgen_diagnostic_reference": {
            "status": "excluded_from_core (Phase 5 remove_from_core)",
            "diagnostic_roc_auc": 0.4868,
            "source": "artifacts/phase5/phase5_summary.json",
        },
        "answers": {
            "real_ontology_beats_legacy": auc["ontology_only_real"]
            > auc["legacy_baseline"],
            "detector_improves_over_ontology_only": detector_available
            and auc.get("detector_only_full", 0) > auc["ontology_only_real"],
            "combined_improves_over_ontology_only": detector_available
            and auc.get("combined_real_without_sgen", 0) > auc["ontology_only_real"],
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (OUT / "final_stat_tests.json").write_text(
        json.dumps({"paired_bootstrap_roc_auc": stat_tests}, indent=2), encoding="utf-8"
    )
    _write_eval_md(result, OUT / "final_evaluation.md")
    _write_stats_md(stat_tests, OUT / "final_stat_tests.md")
    with (OUT / "final_score_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "variant",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "average_precision",
                "test_f1",
                "test_precision",
                "test_recall",
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
                    r["test_f1"],
                    r["test_precision"],
                    r["test_recall"],
                ]
            )

    print(f"[phase7][final] strongest={best} detector_available={detector_available}")
    for r in rows:
        print(
            f"[phase7][final]   {r['variant']}: ROC-AUC={r['roc_auc']} CI={r['roc_auc_ci']} AP={r['average_precision']} F1={r['test_f1']}"
        )
    for t in stat_tests:
        print(
            f"[phase7][stat]   {t['a']} vs {t['b']}: diff={t['observed_diff']} CI={t['ci']} p={t['p_value']} sig={t['significant']}"
        )
    return result


def _write_eval_md(r: dict[str, Any], path: Path) -> None:
    md = [
        "# Phase 7 -- Final Evaluation (benchmark-v2)\n",
        f"**Benchmark:** {r['benchmark']} | **Sgen in core:** {r['sgen_included_in_core']} "
        f"(w_gen={r['w_gen']}) | **strongest:** `{r['strongest_variant']}`\n",
        f"Score equation: `{r['score_equation']}`\n",
        "| variant | ROC-AUC | 95% CI | AP | F1 (val-thr) | P | R |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for v in r["variant_metrics"]:
        md.append(
            f"| {v['variant']} | {v['roc_auc']} | {v['roc_auc_ci']} | {v['average_precision']} | {v['test_f1']} | {v['test_precision']} | {v['test_recall']} |"
        )
    md.append(f"\n**Answers:** {r['answers']}")
    md.append(
        f"\n> Sgen excluded from core (Phase 5 remove_from_core; diagnostic ROC-AUC {r['sgen_diagnostic_reference']['diagnostic_roc_auc']})."
    )
    path.write_text("\n".join(md), encoding="utf-8")


def _write_stats_md(stat_tests: list[dict[str, Any]], path: Path) -> None:
    md = [
        "# Phase 7 -- Statistical Tests (paired bootstrap ROC-AUC diff)\n",
        "| comparison (A vs B) | Δ(A−B) | 95% CI | p | significant |",
        "|---|---:|---|---:|---|",
    ]
    for t in stat_tests:
        md.append(
            f"| {t['a']} vs {t['b']} | {t['observed_diff']} | {t['ci']} | {t['p_value']} | {t['significant']} |"
        )
    path.write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 final evaluation.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--run-id", default=DEFAULT_RUN)
    args = ap.parse_args(argv)
    run(args.config, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
