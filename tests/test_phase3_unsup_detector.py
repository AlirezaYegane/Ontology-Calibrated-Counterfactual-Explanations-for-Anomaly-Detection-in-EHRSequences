"""Phase 3 -- smoke tests for the unsupervised detector scaffold."""

from __future__ import annotations

from pathlib import Path

from src.models.detector_unsup import (
    UnsupDetectorConfig,
    UnsupervisedSequenceDetector,
    build_vocab,
)
from src.training.train_detector_unsup import filter_normal_sequences, run_training

NORMAL = [
    ["DX_10_I10", "MED_ASPIRIN", "DX_10_E119"],
    ["DX_10_E119", "MED_METFORMIN"],
    ["DX_10_I10", "MED_ASPIRIN"],
    ["DX_10_N179", "MED_FUROSEMIDE", "DX_10_I509"],
] * 4


def test_build_vocab_has_specials() -> None:
    vocab = build_vocab(NORMAL)
    for sp in ("<pad>", "<unk>", "<bos>", "<eos>"):
        assert sp in vocab
    assert vocab["<pad>"] == 0


def test_forward_and_score_shapes() -> None:
    vocab = build_vocab(NORMAL)
    det = UnsupervisedSequenceDetector(
        UnsupDetectorConfig(vocab_size=len(vocab), embed_dim=16, hidden_dim=16), vocab
    )
    scores = det.anomaly_scores(NORMAL[:3])
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_filter_normal_sequences_excludes_anomalies() -> None:
    rows = [
        {"codes": ["DX_10_I10"], "is_synthetic_anomaly": 0},
        {"codes": ["DX_10_O09"], "is_synthetic_anomaly": 1},  # anomaly -> excluded
        {"codes": ["DX_10_E119"]},  # missing label -> treated normal
    ]
    normal = filter_normal_sequences(rows)
    assert ["DX_10_I10"] in normal
    assert ["DX_10_E119"] in normal
    assert ["DX_10_O09"] not in normal


def test_tiny_training_and_save_load(tmp_path: Path) -> None:
    out = tmp_path / "unsup"
    summary = run_training(
        NORMAL,
        out_dir=out,
        epochs=1,
        batch_size=4,
        seed=123,
        embed_dim=16,
        hidden_dim=16,
    )
    assert summary["trained_on"] == "normal_sequences_only"
    assert summary["final_train_loss"] is not None
    assert (out / "detector_unsup.pt").exists()
    assert (out / "vocab.json").exists()
    assert (out / "metrics.jsonl").exists()

    det = UnsupervisedSequenceDetector.load(out)
    scores = det.anomaly_scores(NORMAL[:2])
    assert len(scores) == 2


def test_training_deterministic(tmp_path: Path) -> None:
    s1 = run_training(
        NORMAL,
        out_dir=tmp_path / "a",
        epochs=1,
        batch_size=4,
        seed=7,
        embed_dim=16,
        hidden_dim=16,
    )
    s2 = run_training(
        NORMAL,
        out_dir=tmp_path / "b",
        epochs=1,
        batch_size=4,
        seed=7,
        embed_dim=16,
        hidden_dim=16,
    )
    assert s1["final_train_loss"] == s2["final_train_loss"]


def test_training_requires_normal_data() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_training([], out_dir="unused")
