"""Phase 6 -- config loading + invariants (Sgen disabled, w_gen=0)."""

from __future__ import annotations

import pytest

from src.experiments.config import ExperimentConfig, ScoringWeights

CONFIGS = [
    "configs/phase6_detector_smoke.yaml",
    "configs/phase6_detector_full.yaml",
    "configs/phase6_detector_h200.yaml",
]


@pytest.mark.parametrize("path", CONFIGS)
def test_configs_load_and_validate(path) -> None:
    cfg = ExperimentConfig.from_file(path)
    assert cfg.experiment_name
    assert cfg.epochs >= 1 and cfg.batch_size >= 1
    assert cfg.resolved_device() in ("cpu", "cuda")


@pytest.mark.parametrize("path", CONFIGS)
def test_all_configs_disable_sgen(path) -> None:
    cfg = ExperimentConfig.from_file(path)
    assert cfg.sgen_enabled is False
    assert cfg.scoring_weights.w_gen == 0.0


def test_sgen_enabled_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(sgen_enabled=True)


def test_nonzero_wgen_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(scoring_weights=ScoringWeights(w_gen=0.1))


def test_bad_device_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(device="tpu")


def test_roundtrip_dict() -> None:
    cfg = ExperimentConfig.from_file("configs/phase6_detector_smoke.yaml")
    cfg2 = ExperimentConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    assert cfg2.scoring_weights.w_gen == 0.0
