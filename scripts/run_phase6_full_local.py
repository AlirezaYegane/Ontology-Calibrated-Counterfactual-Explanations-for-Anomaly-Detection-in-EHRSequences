"""
scripts/run_phase6_full_local.py
================================
Phase 6 -- FULL LOCAL training pipeline (CPU).

WARNING: this runs FULL-SCALE training and may take a long time on CPU.
It does NOT use Sgen (w_gen = 0). Checkpoints/vocab are written to the run's
IGNORED subdirectories and are never committed.

  python scripts/run_phase6_full_local.py
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

    ap = argparse.ArgumentParser(description="Phase 6 full LOCAL (CPU) pipeline.")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--run-id", default="phase6_detector_full_local")
    args = ap.parse_args(argv)

    print(
        "[phase6][full-local] WARNING: full-scale training on CPU may take a long "
        "time. Sgen is disabled (w_gen=0). Checkpoints go to ignored paths."
    )
    out = pipeline(args.config, run_id=args.run_id)
    de = out["detector_eval"]
    ce = {v["variant"]: v["roc_auc"] for v in out["combined_eval"]["variant_metrics"]}
    print(
        f"[phase6][full-local] detector test ROC-AUC={de['test_roc_auc']} | combined={ce}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
