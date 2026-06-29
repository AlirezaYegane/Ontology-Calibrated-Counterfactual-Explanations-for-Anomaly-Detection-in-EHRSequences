"""
scripts/run_phase6_full_gpu.py
==============================
Phase 6 -- FULL GPU training pipeline.

Runs full-scale training on GPU if available (falls back to CPU with a warning;
does NOT hard-fail if no GPU). Does NOT use Sgen (w_gen = 0). Validation-only
calibration. Checkpoints/vocab go to IGNORED subdirectories.

  python scripts/run_phase6_full_gpu.py
  python scripts/run_phase6_full_gpu.py --config configs/phase6_detector_h200.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG = "configs/phase6_detector_full.yaml"


def main(argv: list[str] | None = None) -> int:
    from scripts.run_phase6_smoke import pipeline
    from src.experiments.config import ExperimentConfig

    ap = argparse.ArgumentParser(description="Phase 6 full GPU pipeline.")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.from_file(args.config)
    dev = cfg.resolved_device()
    if dev != "cuda":
        print(
            "[phase6][full-gpu] WARNING: no CUDA device detected; this will run on "
            "CPU and may be slow. (Not failing — H200/GPU is not a hard dependency.)"
        )
    print(
        f"[phase6][full-gpu] device={dev}; full-scale training; Sgen disabled (w_gen=0)."
    )
    out = pipeline(args.config, run_id=args.run_id)
    de = out["detector_eval"]
    ce = {v["variant"]: v["roc_auc"] for v in out["combined_eval"]["variant_metrics"]}
    print(
        f"[phase6][full-gpu] detector test ROC-AUC={de['test_roc_auc']} | combined={ce}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
