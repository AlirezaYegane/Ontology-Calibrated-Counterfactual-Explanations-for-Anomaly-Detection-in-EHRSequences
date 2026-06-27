"""Phase 5 -- Sgen must stay OUT of the default calibrated score (w_gen = 0)."""

from __future__ import annotations

from src.scoring.ontology_aware import (
    ScoreWeights,
    compute_calibrated_score,
)


def test_default_w_gen_is_zero() -> None:
    assert ScoreWeights().w_gen == 0.0


def test_sgen_not_included_when_w_gen_zero() -> None:
    # Supplying s_gen with the DEFAULT weights (w_gen=0) must not change S_cal.
    without = compute_calibrated_score(s_det=0.3, s_ont=0.8, weights=ScoreWeights())
    with_gen = compute_calibrated_score(
        s_det=0.3, s_ont=0.8, s_gen=0.99, weights=ScoreWeights()
    )
    assert without == with_gen  # s_gen ignored at w_gen=0


def test_sgen_only_enters_when_explicitly_weighted() -> None:
    # Only an explicit w_gen>0 AND a supplied s_gen changes the score.
    base = compute_calibrated_score(s_det=0.3, s_ont=0.8, weights=ScoreWeights())
    weighted = compute_calibrated_score(
        s_det=0.3,
        s_ont=0.8,
        s_gen=0.99,
        weights=ScoreWeights(w_det=0.7, w_ont=0.3, w_gen=0.3),
    )
    assert weighted != base


def test_phase3_phase4_default_scoring_unbroken() -> None:
    # ontology-disabled -> S_cal == S_det (the documented invariant must hold).
    s = compute_calibrated_score(
        s_det=0.42, s_ont=0.0, weights=ScoreWeights(), include_ont=False
    )
    assert abs(s - 0.42) < 1e-9


def test_decision_keeps_w_gen_zero_unless_keep_main() -> None:
    # Mirror the gate->config contract: only keep_main would put Sgen in core.
    from src.evaluation.generative_gate import GenerativeGateInputs, decide_sgen_gate

    res = decide_sgen_gate(
        GenerativeGateInputs(
            model_available=True,
            sgen_roc_auc=0.4868,
            sgen_roc_auc_ci=(0.4633, 0.5109),
            mode_collapse=True,
        )
    )
    sgen_in_core = res.decision == "keep_main"
    assert sgen_in_core is False
    assert ScoreWeights().w_gen == 0.0
