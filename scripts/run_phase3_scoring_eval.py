"""
scripts/run_phase3_scoring_eval.py
==================================
Phase 3 -- Scoring-variant evaluation harness.

Two modes:
  * benchmark-v2 present -> VALID non-circular evaluation. Trains a (smoke-scale)
    unsupervised detector on the clean train split, selects a threshold on val,
    and reports test-set detection metrics with bootstrap CIs for each scoring
    variant. Marked smoke-scale (full-scale training deferred), NOT circular.
  * benchmark-v2 absent -> diagnostic_only_old_circular_benchmark (never final
    evidence).

Variants: detector_only, ontology_only_real, combined_real_ontology,
combined_legacy_ontology. (supervised_synthetic_baseline intentionally omitted as
circular.)

Outputs: artifacts/phase3/phase3_scoring_eval.{json,md}, score_variant_table.csv
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
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase3"
V2_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_v2"
V1_DEFAULT = PROJECT_ROOT / "data" / "processed" / "mimiciv_val_synth_anomaly.pkl"


def _load(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
    return df.to_dict(orient="records")


def _seq(rec: dict[str, Any]) -> list[str]:
    seq = rec.get("model_visible_sequence", rec.get("codes", []))
    return [str(t) for t in seq] if isinstance(seq, (list, tuple)) else []


def _scorer_row(rec: dict[str, Any]) -> dict[str, Any]:
    # Re-key into a row the OntologyAwareScorer understands ("codes" + gender).
    return {
        "codes": _seq(rec),
        "gender": rec.get("gender"),
        "age_group": rec.get("age_group"),
    }


def _minmax(values: list[float], lo: float, hi: float) -> list[float]:
    rng = hi - lo
    if rng <= 0:
        return [0.0 for _ in values]
    return [max(0.0, min(1.0, (v - lo) / rng)) for v in values]


def run_v2(
    out_dir: Path, ont_dir: Path, detector_epochs: int, train_cap: int
) -> dict[str, Any]:
    import numpy as np

    from src.evaluation.calibration import apply_threshold, select_best_f1_threshold
    from src.evaluation.stats import average_precision, bootstrap_auc_ap, roc_auc
    from src.models.detector_unsup import UnsupervisedSequenceDetector
    from src.scoring.ontology_aware import (
        OntologyAwareScorer,
        ScoreWeights,
        compute_calibrated_score,
    )
    from src.training.train_detector_unsup import run_training

    train = _load(V2_DIR / "train.pkl")
    val = _load(V2_DIR / "val.pkl")
    test = _load(V2_DIR / "test.pkl")

    # --- train smoke-scale unsupervised detector on CLEAN train normals only ---
    train_normals = [_seq(r) for r in train if int(r.get("label", 0)) == 0 and _seq(r)]
    if train_cap and len(train_normals) > train_cap:
        train_normals = train_normals[:train_cap]
    det_dir = out_dir / "detector_unsup_v2"
    run_training(
        train_normals,
        out_dir=det_dir,
        epochs=detector_epochs,
        batch_size=32,
        seed=42,
        embed_dim=64,
        hidden_dim=64,
    )
    detector = UnsupervisedSequenceDetector.load(det_dir)

    # --- real ontology scorer ---
    scorer = OntologyAwareScorer.from_processed_dir(
        ont_dir, ontology_mode="real", weights=ScoreWeights()
    )
    legacy = OntologyAwareScorer(ontology_mode="legacy", weights=ScoreWeights())

    def detector_scores(rows: list[dict[str, Any]]) -> list[float]:
        return detector.anomaly_scores([_seq(r) for r in rows])

    def ont_scores(rows: list[dict[str, Any]], scr: OntologyAwareScorer) -> list[float]:
        out = []
        for r in rows:
            try:
                out.append(scr.score(_scorer_row(r), s_det=0.0)["s_ont"])
            except Exception:
                out.append(0.0)
        return out

    val_y = np.array([int(r.get("label", 0)) for r in val])
    test_y = np.array([int(r.get("label", 0)) for r in test])

    val_sdet = detector_scores(val)
    test_sdet = detector_scores(test)
    lo, hi = min(val_sdet), max(val_sdet)
    val_sdet_n = _minmax(val_sdet, lo, hi)
    test_sdet_n = _minmax(test_sdet, lo, hi)

    val_sont_real = ont_scores(val, scorer)
    test_sont_real = ont_scores(test, scorer)
    test_sont_legacy = ont_scores(test, legacy)

    w = ScoreWeights()
    val_comb_real = [
        compute_calibrated_score(d, o, weights=w)
        for d, o in zip(val_sdet_n, val_sont_real)
    ]
    test_comb_real = [
        compute_calibrated_score(d, o, weights=w)
        for d, o in zip(test_sdet_n, test_sont_real)
    ]
    test_comb_legacy = [
        compute_calibrated_score(d, o, weights=w)
        for d, o in zip(test_sdet_n, test_sont_legacy)
    ]

    variants = {
        "detector_only": (test_sdet_n, None),
        "ontology_only_real": (test_sont_real, None),
        "combined_real_ontology": (test_comb_real, (val_comb_real, val_y)),
        "combined_legacy_ontology": (test_comb_legacy, None),
    }

    rows: list[dict[str, Any]] = []
    for name, (scores, val_sel) in variants.items():
        scores = list(map(float, scores))
        metrics = bootstrap_auc_ap(test_y, scores, n_boot=500, seed=42)
        row: dict[str, Any] = {
            "variant": name,
            "test_roc_auc": round(roc_auc(test_y, np.array(scores)), 4),
            "test_roc_auc_ci": [
                round(metrics["roc_auc"]["ci_low"], 4),
                round(metrics["roc_auc"]["ci_high"], 4),
            ],
            "test_average_precision": round(
                average_precision(test_y, np.array(scores)), 4
            ),
            "test_ap_ci": [
                round(metrics["average_precision"]["ci_low"], 4),
                round(metrics["average_precision"]["ci_high"], 4),
            ],
        }
        if val_sel is not None:
            val_scores, val_labels = val_sel
            thr = select_best_f1_threshold(val_labels, val_scores)
            applied = apply_threshold(test_y, scores, thr.threshold)
            row["threshold_selected_on_val"] = round(thr.threshold, 4)
            row["test_precision"] = round(applied["precision"], 4)
            row["test_recall"] = round(applied["recall"], 4)
            row["test_f1"] = round(applied["f1"], 4)
        rows.append(row)

    return {
        "mode": "valid_v2_benchmark",
        "detector": "unsupervised_next_token_gru (smoke-scale)",
        "detector_train_normals": len(train_normals),
        "detector_epochs": detector_epochs,
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "test_anomaly_rate": round(float(test_y.mean()), 4),
        "variant_metrics": rows,
        "threshold_protocol": "selected on val (best-F1), applied to test; no test-set tuning",
        "scale_note": "SMOKE-SCALE detector (small model, few epochs, capped train). Valid non-circular benchmark; metrics are preliminary, not final paper SOTA.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3 scoring-variant evaluation.")
    ap.add_argument("--ontology-dir", default=str(PROCESSED_ONT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--detector-epochs", type=int, default=3)
    ap.add_argument("--train-cap", type=int, default=8000)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ont_dir = Path(args.ontology_dir)

    v2_ok = (
        (V2_DIR / "val.pkl").exists()
        and (V2_DIR / "test.pkl").exists()
        and (V2_DIR / "train.pkl").exists()
    )
    triviality = PROJECT_ROOT / "artifacts" / "diagnostics" / "rf2_triviality_v2.json"
    triviality_pass = None
    if triviality.exists():
        try:
            tv = json.loads(triviality.read_text(encoding="utf-8"))["verdict"]
            triviality_pass = tv["strongest_discriminative_power"] < 0.80
        except Exception:
            triviality_pass = None

    if v2_ok:
        result = run_v2(out_dir, ont_dir, args.detector_epochs, args.train_cap)
        status = "valid_v2_benchmark"
        final_evidence = False  # smoke-scale; full training deferred
        report = {
            "phase": 3,
            "status": status,
            "benchmark": {"benchmark": "v2", "circular": False, "dir": str(V2_DIR)},
            "benchmark_v2_triviality_pass": triviality_pass,
            "real_ontology_available": (ont_dir / "snomed_hierarchy.json").exists(),
            "final_paper_evidence_claimable": final_evidence,
            "evidence_level": "smoke_scale_valid_non_circular",
            **result,
            "warnings": [
                "Detector is SMOKE-SCALE (small model / few epochs / capped train). Metrics are "
                "preliminary non-circular evidence, NOT final paper SOTA; full-scale training is a "
                "later experiment phase.",
            ],
        }
        variant_rows = result["variant_metrics"]
    else:
        status = "diagnostic_only_old_circular_benchmark"
        report = {
            "phase": 3,
            "status": status,
            "benchmark": {"benchmark": "v1", "circular": True},
            "final_paper_evidence_claimable": False,
            "reason": "benchmark-v2 absent; complete Phase 1b first.",
            "warnings": ["Benchmark-v2 not found; no valid metrics computed."],
        }
        variant_rows = []

    (out_dir / "phase3_scoring_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "variant",
        "test_roc_auc",
        "test_roc_auc_ci",
        "test_average_precision",
        "test_ap_ci",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    with (out_dir / "score_variant_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in variant_rows:
            writer.writerow(row)

    md = [
        "# Phase 3 Scoring-Variant Evaluation\n",
        f"**Status:** `{status}`  ",
        f"**Benchmark-v2 triviality pass (<0.80):** {triviality_pass}  ",
        f"**Final paper evidence:** {report['final_paper_evidence_claimable']} "
        f"({report.get('evidence_level', 'n/a')})\n",
    ]
    if variant_rows:
        md.append(
            "## Test-set detection metrics (non-circular benchmark-v2, smoke-scale detector)\n"
        )
        md.append("| variant | ROC-AUC | ROC-AUC 95% CI | AP | F1 (val-thr) |")
        md.append("|---|---:|---|---:|---:|")
        for r in variant_rows:
            md.append(
                f"| {r['variant']} | {r['test_roc_auc']} | {r.get('test_roc_auc_ci')} | {r['test_average_precision']} | {r.get('test_f1', '-')} |"
            )
        md.append("")
    for w_ in report.get("warnings", []):
        md.append(f"> ⚠️ {w_}")
    (out_dir / "phase3_scoring_eval.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[phase3] status={status} triviality_pass={triviality_pass}")
    for r in variant_rows:
        print(
            f"[phase3] {r['variant']}: ROC-AUC={r['test_roc_auc']} CI={r.get('test_roc_auc_ci')} AP={r['test_average_precision']}"
        )
    print(f"[phase3] wrote {out_dir / 'phase3_scoring_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
