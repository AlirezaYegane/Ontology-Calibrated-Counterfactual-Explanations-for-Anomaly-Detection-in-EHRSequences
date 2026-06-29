"""Phase 6 -- experiment index tracking (no PHI; update-by-run_id)."""

from __future__ import annotations

import json

import pytest

from src.experiments.tracking import record_run


def _entry(run_id="run1", **kw):
    e = {
        "run_id": run_id,
        "timestamp": "2026-06-24T00:00:00Z",
        "git_commit": "abc1234",
        "config_path": "configs/phase6_detector_smoke.yaml",
        "seed": 42,
        "dataset": "data/processed/benchmark_v2",
        "n_train": 100,
        "n_val": 50,
        "n_test": 60,
        "device": "cpu",
        "epochs_completed": 2,
        "best_epoch": 1,
        "checkpoint_path": "artifacts/phase6/runs/run1/checkpoints",
        "evidence_level": "smoke",
        "detector_metrics": {"roc_auc": 0.43},
        "combined_metrics": {"combined_real_without_sgen": 0.66},
        "status": "evaluated",
        "notes": "Sgen disabled.",
    }
    e.update(kw)
    return e


def test_record_run_writes_index(tmp_path) -> None:
    record_run(_entry(), index_dir=tmp_path)
    jp = tmp_path / "experiment_index.json"
    mp = tmp_path / "experiment_index.md"
    assert jp.exists() and mp.exists()
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["n_runs"] == 1
    assert data["runs"][0]["run_id"] == "run1"


def test_record_run_updates_by_run_id(tmp_path) -> None:
    record_run(_entry(status="trained"), index_dir=tmp_path)
    record_run(_entry(status="evaluated"), index_dir=tmp_path)  # same run_id
    data = json.loads((tmp_path / "experiment_index.json").read_text(encoding="utf-8"))
    assert data["n_runs"] == 1  # updated, not duplicated
    assert data["runs"][0]["status"] == "evaluated"


def test_record_run_rejects_phi_keys(tmp_path) -> None:
    with pytest.raises(ValueError):
        record_run(_entry(per_record=[1, 2, 3]), index_dir=tmp_path)
    with pytest.raises(ValueError):
        record_run(_entry(subject_id="x"), index_dir=tmp_path)


def test_index_has_no_raw_sequences(tmp_path) -> None:
    record_run(_entry(), index_dir=tmp_path)
    text = (tmp_path / "experiment_index.json").read_text(encoding="utf-8").lower()
    for forbidden in ("dx_10_", "med_", "subject_id", "per_record"):
        assert forbidden not in text
