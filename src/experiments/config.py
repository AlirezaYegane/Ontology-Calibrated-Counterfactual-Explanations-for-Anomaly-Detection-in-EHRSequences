"""
src/experiments/config.py
=========================
Phase 6 -- Experiment configuration for reproducible full-scale training.

A single dataclass describes a detector training/evaluation experiment. Configs
load from YAML or JSON; defaults keep the Sgen-free core invariant: ``sgen_enabled``
is always False and ``w_gen`` is always 0.0 (validated on construction). Scoring
weights are explicit (``w_det`` / ``w_ont`` / ``w_gen=0``), matching the Phase 3b/5
calibrated score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_V2 = "data/processed/benchmark_v2"


@dataclass(frozen=True)
class ScoringWeights:
    w_det: float = 0.70
    w_ont: float = 0.30
    w_gen: float = 0.0  # Sgen removed from core (Phase 5); MUST stay 0 in Phase 6


@dataclass
class ExperimentConfig:
    """Reproducible experiment description (detector training + evaluation)."""

    experiment_name: str = "phase6_detector"
    seed: int = 42
    # data
    data_dir: str = _V2
    train_split: str = "train.pkl"
    val_split: str = "val.pkl"
    test_split: str = "test.pkl"
    sequence_key: str = "model_visible_sequence"
    label_key: str = "label"
    # model
    model_type: str = "unsupervised_next_token_gru"
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    max_len: int = 256
    # training
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 4
    early_stopping_metric: str = "val_roc_auc"  # val-only model selection
    train_cap: int = 0  # 0 = use all normal train sequences
    # runtime
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    num_workers: int = 0
    # outputs / policy
    output_dir: str = "artifacts/phase6/runs"
    checkpoint_policy: str = "best_and_last"  # best.pt (val) + last.pt (resume)
    calibration_policy: str = "val_best_f1"  # threshold selected on val only
    resume: bool = True
    # scoring
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    sgen_enabled: bool = False  # Sgen removed from core (Phase 5) -- never enable here
    evidence_level: str = "smoke"  # smoke | full_local | full_gpu | h200

    def __post_init__(self) -> None:
        # Hard invariants: Sgen stays out of the core in every Phase 6 config.
        if self.sgen_enabled:
            raise ValueError(
                "sgen_enabled must be False in Phase 6 (Sgen removed from core in Phase 5)."
            )
        if float(self.scoring_weights.w_gen) != 0.0:
            raise ValueError(
                f"w_gen must be 0.0 in Phase 6, got {self.scoring_weights.w_gen}."
            )
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be auto|cpu|cuda, got {self.device!r}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1.")

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def split_path(self, which: str) -> Path:
        name = {
            "train": self.train_split,
            "val": self.val_split,
            "test": self.test_split,
        }[which]
        return PROJECT_ROOT / self.data_dir / name

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        sw = d.pop("scoring_weights", None)
        cfg = cls(**d)
        if sw is not None:
            object.__setattr__(cfg, "scoring_weights", ScoringWeights(**sw))
            cfg.__post_init__()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data or {})


__all__ = ["ExperimentConfig", "ScoringWeights", "PROJECT_ROOT"]
