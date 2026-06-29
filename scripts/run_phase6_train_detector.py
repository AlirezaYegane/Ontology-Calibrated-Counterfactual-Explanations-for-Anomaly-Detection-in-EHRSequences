"""
scripts/run_phase6_train_detector.py
=====================================
Phase 6 -- Full-scale unsupervised detector training driven by a config.

Trains ONLY on benchmark-v2 clean-normal train sequences. Validation is used for
model selection / early stopping; test is never touched here. Checkpoints + vocab
go to the run's ignored subdirectories; aggregate summaries are committable.

  python scripts/run_phase6_train_detector.py --config configs/phase6_detector_smoke.yaml
  python scripts/run_phase6_train_detector.py --config configs/phase6_detector_full.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_split(
    path: Path, seq_key: str, label_key: str
) -> tuple[list[list[str]], list[int]]:
    import pandas as pd

    rows = pd.read_pickle(path).to_dict(orient="records")
    seqs, labels = [], []
    for r in rows:
        s = r.get(seq_key, r.get("codes", []))
        seqs.append([str(t) for t in s] if isinstance(s, (list, tuple)) else [])
        labels.append(int(r.get(label_key, 0)))
    return seqs, labels


def run(
    config_path: str, run_id: str | None = None, max_epochs: int | None = None
) -> dict[str, Any]:
    from src.experiments.config import ExperimentConfig
    from src.experiments.tracking import git_commit, now_iso, record_run
    from src.training.train_detector_unsup import train_detector_full

    cfg = ExperimentConfig.from_file(config_path)
    rid = run_id or f"{cfg.experiment_name}_seed{cfg.seed}"
    out_dir = PROJECT_ROOT / cfg.output_dir / rid

    train_seqs, train_labels = _load_split(
        cfg.split_path("train"), cfg.sequence_key, cfg.label_key
    )
    val_seqs, val_labels = _load_split(
        cfg.split_path("val"), cfg.sequence_key, cfg.label_key
    )

    # detector trains on NORMAL train sequences only (no anomaly labels)
    train_normals = [s for s, y in zip(train_seqs, train_labels) if int(y) == 0 and s]
    if cfg.train_cap and len(train_normals) > cfg.train_cap:
        train_normals = train_normals[: cfg.train_cap]

    epochs = max_epochs or cfg.epochs
    summary = train_detector_full(
        train_normals,
        val_seqs,
        val_labels,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        device=cfg.resolved_device(),
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_metric=cfg.early_stopping_metric,
        resume=cfg.resume,
        config_snapshot=cfg.to_dict(),
    )

    record_run(
        {
            "run_id": rid,
            "timestamp": now_iso(),
            "git_commit": git_commit(),
            "config_path": config_path,
            "seed": cfg.seed,
            "dataset": cfg.data_dir,
            "n_train": summary["n_train_normal"],
            "n_val": summary["n_val"],
            "n_test": None,
            "device": summary["device"],
            "epochs_completed": summary["epochs_run"],
            "best_epoch": summary["best_epoch"],
            "checkpoint_path": str(out_dir / "checkpoints"),
            "evidence_level": cfg.evidence_level,
            "detector_metrics": {"val_roc_auc": summary["best_metric_value"]},
            "combined_metrics": None,
            "status": "trained",
            "notes": "Sgen disabled (w_gen=0); val-only model selection.",
        }
    )

    print(
        f"[phase6][train] run_id={rid} device={summary['device']} "
        f"best_epoch={summary['best_epoch']} best_{cfg.early_stopping_metric}="
        f"{summary['best_metric_value']} epochs_run={summary['epochs_run']}"
    )
    print(f"[phase6][train] checkpoints (ignored): {out_dir / 'checkpoints'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 detector training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override epochs (e.g. resume cap).",
    )
    args = ap.parse_args(argv)
    summary = run(args.config, run_id=args.run_id, max_epochs=args.max_epochs)
    print(
        json.dumps(
            {k: summary[k] for k in ("best_epoch", "best_metric_value", "epochs_run")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
