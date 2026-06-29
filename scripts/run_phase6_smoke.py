"""
scripts/run_phase6_smoke.py
===========================
Phase 6 -- SMOKE pipeline: train (CPU, tiny) -> evaluate detector -> combined eval.

Fast end-to-end sanity check of the whole Phase 6 infrastructure. Used by tests.
Sgen is disabled throughout (w_gen = 0).

  python scripts/run_phase6_smoke.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SMOKE_CONFIG = "configs/phase6_detector_smoke.yaml"


def pipeline(config_path: str, run_id: str | None = None) -> dict[str, Any]:
    """Train -> detector eval -> combined eval. Updates the experiment index."""
    from scripts.run_phase6_combined_eval import run as combined_run
    from scripts.run_phase6_evaluate_detector import run as detector_run
    from scripts.run_phase6_train_detector import run as train_run
    from src.experiments.config import ExperimentConfig
    from src.experiments.tracking import git_commit, now_iso, record_run

    cfg = ExperimentConfig.from_file(config_path)
    rid = run_id or f"{cfg.experiment_name}_seed{cfg.seed}"

    train_summary = train_run(config_path, run_id=rid)
    det_eval = detector_run(config_path, run_id=rid)
    comb_eval = combined_run(config_path, run_id=rid)

    comb_auc = {v["variant"]: v["roc_auc"] for v in comb_eval["variant_metrics"]}
    record_run(
        {
            "run_id": rid,
            "timestamp": now_iso(),
            "git_commit": git_commit(),
            "config_path": config_path,
            "seed": cfg.seed,
            "dataset": cfg.data_dir,
            "n_train": train_summary["n_train_normal"],
            "n_val": det_eval["n_val"],
            "n_test": det_eval["n_test"],
            "device": train_summary["device"],
            "epochs_completed": train_summary["epochs_run"],
            "best_epoch": train_summary["best_epoch"],
            "checkpoint_path": train_summary["best_checkpoint"],
            "evidence_level": cfg.evidence_level,
            "detector_metrics": {
                "roc_auc": det_eval["test_roc_auc"],
                "average_precision": det_eval["test_average_precision"],
                "f1": det_eval["test_f1"],
            },
            "combined_metrics": comb_auc,
            "status": "evaluated",
            "notes": "Sgen disabled (w_gen=0); val-only calibration; ontology-centered.",
        }
    )
    return {
        "train": train_summary,
        "detector_eval": det_eval,
        "combined_eval": comb_eval,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 smoke pipeline.")
    ap.add_argument("--config", default=DEFAULT_SMOKE_CONFIG)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    out = pipeline(args.config, run_id=args.run_id)
    de = out["detector_eval"]
    ce = {v["variant"]: v["roc_auc"] for v in out["combined_eval"]["variant_metrics"]}
    print(f"[phase6][smoke] detector test ROC-AUC={de['test_roc_auc']} | combined={ce}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
