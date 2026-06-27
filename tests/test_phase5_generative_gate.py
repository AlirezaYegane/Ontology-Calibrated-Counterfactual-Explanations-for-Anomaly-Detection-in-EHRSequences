"""Phase 5 -- tests for the Sgen decision gate (strict evidence requirements)."""

from __future__ import annotations

from src.evaluation.generative_gate import GenerativeGateInputs, decide_sgen_gate


def test_keep_main_only_under_full_strict_evidence() -> None:
    strong = GenerativeGateInputs(
        model_available=True,
        sgen_roc_auc=0.72,
        sgen_roc_auc_ci=(0.66, 0.78),
        adds_signal_beyond_ont_det=True,
        combined_with_sgen_improves=True,
        leakage_detected=False,
        protocol_valid=True,
        mode_collapse=False,
    )
    assert decide_sgen_gate(strong).decision == "keep_main"


def test_keep_main_denied_if_any_condition_fails() -> None:
    base = dict(
        model_available=True,
        sgen_roc_auc=0.72,
        sgen_roc_auc_ci=(0.66, 0.78),
        adds_signal_beyond_ont_det=True,
        combined_with_sgen_improves=True,
        leakage_detected=False,
        protocol_valid=True,
        mode_collapse=False,
    )
    # drop combined improvement -> not keep_main
    assert (
        decide_sgen_gate(
            GenerativeGateInputs(**{**base, "combined_with_sgen_improves": False})
        ).decision
        != "keep_main"
    )
    # invalid protocol -> not keep_main
    assert (
        decide_sgen_gate(
            GenerativeGateInputs(**{**base, "protocol_valid": False})
        ).decision
        != "keep_main"
    )
    # CI does not clear chance -> not keep_main
    assert (
        decide_sgen_gate(
            GenerativeGateInputs(**{**base, "sgen_roc_auc_ci": (0.49, 0.80)})
        ).decision
        != "keep_main"
    )
    # leakage -> not keep_main
    assert (
        decide_sgen_gate(
            GenerativeGateInputs(**{**base, "leakage_detected": True})
        ).decision
        != "keep_main"
    )


def test_diagnostic_only_for_weak_but_running() -> None:
    weak = GenerativeGateInputs(
        model_available=True,
        sgen_roc_auc=0.535,  # above near-random, below meaningful
        sgen_roc_auc_ci=(0.51, 0.56),
        adds_signal_beyond_ont_det=False,
        combined_with_sgen_improves=False,
        leakage_detected=False,
        protocol_valid=False,  # old-data checkpoint
        mode_collapse=False,
    )
    assert decide_sgen_gate(weak).decision == "diagnostic_only"


def test_remove_from_core_for_near_random() -> None:
    near_random = GenerativeGateInputs(
        model_available=True,
        sgen_roc_auc=0.4868,
        sgen_roc_auc_ci=(0.46, 0.51),
        adds_signal_beyond_ont_det=False,
        combined_with_sgen_improves=False,
        mode_collapse=True,
    )
    assert decide_sgen_gate(near_random).decision == "remove_from_core"


def test_remove_from_core_for_mode_collapse_even_if_auc_ok() -> None:
    collapsed = GenerativeGateInputs(
        model_available=True,
        sgen_roc_auc=0.6,
        sgen_roc_auc_ci=(0.55, 0.65),
        adds_signal_beyond_ont_det=True,
        combined_with_sgen_improves=True,
        protocol_valid=True,
        mode_collapse=True,
    )
    assert decide_sgen_gate(collapsed).decision == "remove_from_core"


def test_blocked_when_no_model() -> None:
    assert (
        decide_sgen_gate(GenerativeGateInputs(model_available=False)).decision
        == "blocked_no_valid_model"
    )
    # model present but no AUC computed
    assert (
        decide_sgen_gate(
            GenerativeGateInputs(model_available=True, sgen_roc_auc=None)
        ).decision
        == "blocked_no_valid_model"
    )


def test_actual_phase5_evidence_yields_remove_from_core() -> None:
    # The real benchmark-v2 result from run_phase5_generative_eval.
    actual = GenerativeGateInputs(
        model_available=True,
        sgen_roc_auc=0.4868,
        sgen_roc_auc_ci=(0.4633, 0.5109),
        adds_signal_beyond_ont_det=False,
        combined_with_sgen_improves=False,
        leakage_detected=False,
        protocol_valid=False,
        mode_collapse=True,
    )
    assert decide_sgen_gate(actual).decision == "remove_from_core"
