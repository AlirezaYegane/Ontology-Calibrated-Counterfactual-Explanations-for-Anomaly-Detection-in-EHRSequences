"""Phase 6 -- smoke training on a tiny synthetic fixture (no licensed data)."""

from __future__ import annotations

from src.training.train_detector_unsup import train_detector_full

# tiny synthetic corpus: normal sequences share a pattern; "anomalies" differ.
NORMALS = [["A", "B", "C", "D"], ["A", "B", "C"], ["A", "B", "D"], ["A", "C", "D"]] * 8
VAL_SEQS = [["A", "B", "C"], ["A", "B", "D"], ["Z", "Z", "Z"], ["Q", "Q"]]
VAL_LABELS = [0, 0, 1, 1]


def test_train_full_creates_artifacts(tmp_path) -> None:
    summary = train_detector_full(
        NORMALS,
        VAL_SEQS,
        VAL_LABELS,
        out_dir=tmp_path,
        epochs=3,
        batch_size=8,
        embed_dim=16,
        hidden_dim=16,
        seed=42,
        early_stopping_patience=5,
        resume=False,
    )
    assert summary["n_train_normal"] == len(NORMALS)
    assert summary["best_epoch"] >= 0
    assert (tmp_path / "train_metrics.jsonl").exists()
    assert (tmp_path / "train_summary.json").exists()
    # best checkpoint + vocab live under the (ignored) checkpoints/ dir
    assert (tmp_path / "checkpoints" / "detector_unsup.pt").exists()
    assert (tmp_path / "checkpoints" / "last.pt").exists()
    # per-epoch metrics logged
    lines = (tmp_path / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1


def test_training_is_deterministic(tmp_path) -> None:
    a = train_detector_full(
        NORMALS,
        VAL_SEQS,
        VAL_LABELS,
        out_dir=tmp_path / "a",
        epochs=2,
        batch_size=8,
        embed_dim=16,
        hidden_dim=16,
        seed=7,
        resume=False,
    )
    b = train_detector_full(
        NORMALS,
        VAL_SEQS,
        VAL_LABELS,
        out_dir=tmp_path / "b",
        epochs=2,
        batch_size=8,
        embed_dim=16,
        hidden_dim=16,
        seed=7,
        resume=False,
    )
    assert a["final_val_metrics"]["train_loss"] == b["final_val_metrics"]["train_loss"]


def test_resume_continues_training(tmp_path) -> None:
    s1 = train_detector_full(
        NORMALS,
        VAL_SEQS,
        VAL_LABELS,
        out_dir=tmp_path,
        epochs=2,
        batch_size=8,
        embed_dim=16,
        hidden_dim=16,
        seed=42,
        resume=True,
    )
    assert s1["epochs_run"] == 2
    # resume with more epochs -> continues from where it stopped
    s2 = train_detector_full(
        NORMALS,
        VAL_SEQS,
        VAL_LABELS,
        out_dir=tmp_path,
        epochs=4,
        batch_size=8,
        embed_dim=16,
        hidden_dim=16,
        seed=42,
        resume=True,
    )
    assert s2["epochs_run"] == 4
    n_lines = len(
        (tmp_path / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert n_lines == 4  # 2 + 2 appended


def test_empty_training_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError):
        train_detector_full([], VAL_SEQS, VAL_LABELS, out_dir=tmp_path, epochs=1)
