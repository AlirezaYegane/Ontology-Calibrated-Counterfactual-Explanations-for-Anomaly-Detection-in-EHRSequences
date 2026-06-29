"""Phase 6 -- Sgen-exclusion + data-safety (gitignore) invariants."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.experiments.config import ExperimentConfig
from src.scoring.ontology_aware import ScoreWeights, compute_calibrated_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "configs/phase6_detector_smoke.yaml",
    "configs/phase6_detector_full.yaml",
    "configs/phase6_detector_h200.yaml",
]


def test_default_scoring_weights_keep_wgen_zero() -> None:
    assert ScoreWeights().w_gen == 0.0


def test_all_phase6_configs_keep_sgen_disabled() -> None:
    for path in CONFIGS:
        cfg = ExperimentConfig.from_file(path)
        assert cfg.sgen_enabled is False
        assert cfg.scoring_weights.w_gen == 0.0
        assert cfg.evidence_level in ("smoke", "full_local", "full_gpu", "h200")


def test_sgen_not_silently_included_in_scal() -> None:
    # Even if an s_gen is supplied, with w_gen=0 it must NOT change S_cal.
    base = compute_calibrated_score(s_det=0.4, s_ont=0.5, weights=ScoreWeights())
    with_gen = compute_calibrated_score(
        s_det=0.4, s_ont=0.5, s_gen=0.99, weights=ScoreWeights()
    )
    assert base == with_gen


def _ignored(rel: str) -> bool:
    out = subprocess.run(
        ["git", "check-ignore", rel], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return out.returncode == 0


def test_checkpoint_vocab_per_record_paths_are_gitignored() -> None:
    assert _ignored("artifacts/phase6/runs/somerun/checkpoints/detector_unsup.pt")
    assert _ignored("artifacts/phase6/runs/somerun/checkpoints/last.pt")
    assert _ignored("artifacts/phase6/runs/somerun/vocab/vocab.json")
    assert _ignored("artifacts/phase6/runs/somerun/ignored/per_record_scores.csv")
    assert _ignored("artifacts/phase6/runs/somerun/checkpoints/anything.pkl")


def test_aggregate_summaries_are_not_ignored() -> None:
    # the committable aggregate artifacts must NOT be ignored
    assert not _ignored("artifacts/phase6/experiment_index.json")
    assert not _ignored("artifacts/phase6/runs/somerun/train_summary.json")
    assert not _ignored("artifacts/phase6/runs/somerun/combined_eval.json")
