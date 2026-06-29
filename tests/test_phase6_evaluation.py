"""Phase 6 -- evaluation building blocks: val-only calibration + Sgen-free combined."""

from __future__ import annotations

from src.evaluation.calibration import apply_threshold, select_best_f1_threshold
from src.experiments.config import ScoringWeights
from src.experiments.eval_common import combined_scores, minmax_apply, minmax_fit
from src.models.detector_unsup import UnsupervisedSequenceDetector
from src.training.train_detector_unsup import train_detector_full

NORMALS = [["A", "B", "C", "D"], ["A", "B", "C"], ["A", "B", "D"]] * 10
VAL_SEQS = [["A", "B", "C"], ["A", "B", "D"], ["Z", "Z", "Z"], ["Q", "Q"]]
VAL_LABELS = [0, 0, 1, 1]


def test_threshold_selected_on_val_applied_to_test() -> None:
    val_labels = [0, 0, 1, 1]
    val_scores = [0.1, 0.2, 0.8, 0.9]
    thr = select_best_f1_threshold(val_labels, val_scores)
    assert thr.selected_on == "validation"
    applied = apply_threshold([0, 1], [0.15, 0.85], thr.threshold)
    assert applied["applied_on"] == "test"


def test_combined_excludes_sgen_and_matches_formula() -> None:
    weights = ScoringWeights(w_det=0.7, w_ont=0.3, w_gen=0.0)
    det = [0.5, 0.2]
    ont = [1.0, 0.0]
    out = combined_scores(det, ont, weights)
    # S_cal = (0.7*det + 0.3*normalize_sont(ont)) / (0.7+0.3); sgen never enters
    import math

    for d, o, got in zip(det, ont, out):
        ont_norm = 1.0 - math.exp(-max(o, 0.0))
        expected = (0.7 * d + 0.3 * ont_norm) / 1.0
        assert abs(got - expected) < 1e-9


def test_trained_detector_scores_load_and_run(tmp_path) -> None:
    train_detector_full(
        NORMALS,
        VAL_SEQS,
        VAL_LABELS,
        out_dir=tmp_path,
        epochs=2,
        batch_size=8,
        embed_dim=16,
        hidden_dim=16,
        seed=42,
        resume=False,
    )
    det = UnsupervisedSequenceDetector.load(tmp_path / "checkpoints")
    scores = det.anomaly_scores(VAL_SEQS, batch_size=4)
    assert len(scores) == len(VAL_SEQS)
    assert all(isinstance(s, float) for s in scores)


def test_minmax_is_fit_on_val_only() -> None:
    val = [0.0, 2.0, 4.0]
    lo, hi = minmax_fit(val)
    # a test value beyond the val range is clamped to [0,1] (no test-set refit)
    assert minmax_apply([6.0], lo, hi) == [1.0]
    assert minmax_apply([-1.0], lo, hi) == [0.0]
