"""
src/evaluation/generative_gate.py
=================================
Phase 5 -- Formal Sgen / generative decision gate.

A pure, testable decision function that maps evaluation evidence to one of four
decisions about the diffusion / generative-surprise (`Sgen`) component:

  * ``keep_main``            -- include Sgen in the main calibrated score.
  * ``diagnostic_only``      -- Sgen runs but is weak / non-improving / from a
                                diagnostic-only (e.g. old-data) checkpoint.
  * ``remove_from_core``     -- Sgen is near-random or worse, or mode-collapsed,
                                or harms the paper's clarity.
  * ``blocked_no_valid_model`` -- no usable trained model/checkpoint exists.

The thresholds are intentionally strict: ``keep_main`` requires *every* condition
to hold. This module performs NO I/O and reads NO labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Strict thresholds for keep_main.
MEANINGFUL_AUC = 0.55  # Sgen ROC-AUC must clear this to be "meaningfully > 0.50"
CHANCE = 0.50
# Below this, Sgen is "near-random or worse" -> remove_from_core.
NEAR_RANDOM_AUC = 0.52


@dataclass(frozen=True)
class GenerativeGateInputs:
    """Evidence fed to the gate (all derived from a VALID benchmark eval)."""

    model_available: bool
    sgen_roc_auc: float | None = None
    sgen_roc_auc_ci: tuple[float, float] | None = None  # (low, high)
    adds_signal_beyond_ont_det: bool = False
    combined_with_sgen_improves: bool = False
    leakage_detected: bool = False
    protocol_valid: bool = False  # trained on the right (benchmark-v2 normal-only) data
    mode_collapse: bool = False


@dataclass
class GenerativeGateResult:
    decision: str
    reasons: list[str] = field(default_factory=list)
    criteria: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": self.reasons,
            "criteria": self.criteria,
        }


def decide_sgen_gate(inp: GenerativeGateInputs) -> GenerativeGateResult:
    """Return the Sgen decision under strict, explicit criteria."""
    reasons: list[str] = []

    # 0) No model at all -> blocked.
    if not inp.model_available:
        return GenerativeGateResult(
            decision="blocked_no_valid_model",
            reasons=["no usable trained generative model/checkpoint available"],
            criteria={"model_available": False},
        )
    if inp.sgen_roc_auc is None:
        return GenerativeGateResult(
            decision="blocked_no_valid_model",
            reasons=["model present but Sgen could not be evaluated (no ROC-AUC)"],
            criteria={"model_available": True, "sgen_roc_auc": None},
        )

    auc = float(inp.sgen_roc_auc)
    ci_low = inp.sgen_roc_auc_ci[0] if inp.sgen_roc_auc_ci else None
    ci_clears_chance = ci_low is not None and ci_low > CHANCE

    keep_conditions = {
        "auc_meaningfully_above_chance": auc >= MEANINGFUL_AUC,
        "ci_clears_chance": bool(ci_clears_chance),
        "adds_signal_beyond_ont_det": inp.adds_signal_beyond_ont_det,
        "combined_with_sgen_improves": inp.combined_with_sgen_improves,
        "no_leakage": not inp.leakage_detected,
        "protocol_valid": inp.protocol_valid,
        "no_mode_collapse": not inp.mode_collapse,  # collapse disqualifies keep_main
    }

    # 1) keep_main only if ALL strict conditions hold.
    if all(keep_conditions.values()):
        reasons.append(
            f"Sgen ROC-AUC {auc:.4f} (CI low {ci_low}) clears chance, adds signal "
            "beyond ontology+detector, improves the combined score, no leakage, valid protocol."
        )
        return GenerativeGateResult(
            decision="keep_main", reasons=reasons, criteria=keep_conditions
        )

    # 2) remove_from_core: near-random/worse, or mode-collapsed.
    if auc <= NEAR_RANDOM_AUC or inp.mode_collapse:
        if auc <= NEAR_RANDOM_AUC:
            reasons.append(
                f"Sgen ROC-AUC {auc:.4f} is near-random or worse (<= {NEAR_RANDOM_AUC})."
            )
        if inp.mode_collapse:
            reasons.append("generation is mode-collapsed.")
        reasons.append("keeping Sgen would weaken the paper's clarity/strength.")
        return GenerativeGateResult(
            decision="remove_from_core",
            reasons=reasons,
            criteria={**keep_conditions, "mode_collapse": inp.mode_collapse},
        )

    # 3) diagnostic_only: runs, not near-random, but doesn't meet keep_main
    #    (weak, non-improving, or from an invalid/old-data protocol).
    failed = [k for k, v in keep_conditions.items() if not v]
    reasons.append(
        f"Sgen runs (ROC-AUC {auc:.4f}) but fails keep_main conditions: {failed}. "
        "Demote to diagnostic/appendix; keep out of the core calibrated score."
    )
    return GenerativeGateResult(
        decision="diagnostic_only", reasons=reasons, criteria=keep_conditions
    )


__all__ = ["GenerativeGateInputs", "GenerativeGateResult", "decide_sgen_gate"]
