"""
src/experiments/eval_common.py
==============================
Phase 6 -- Shared evaluation helpers (detector / ontology / combined scoring).

All scoring uses ONLY model-visible fields. ``label`` / ``anomaly_type`` are used
for evaluation bucketing only, never as scorer inputs. The combined score is the
Sgen-free calibrated score (``w_gen = 0``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_records(path: Path, seq_key: str, label_key: str) -> list[dict[str, Any]]:
    import pandas as pd

    rows = pd.read_pickle(path).to_dict(orient="records")
    out = []
    for r in rows:
        s = r.get(seq_key, r.get("codes", []))
        out.append(
            {
                "seq": [str(t) for t in s] if isinstance(s, (list, tuple)) else [],
                "gender": r.get("gender"),
                "age_group": r.get("age_group"),
                "label": int(r.get(label_key, 0)),
                "anomaly_type": str(r.get("anomaly_type") or "anomaly"),
            }
        )
    return out


def detector_scores(
    detector: Any, records: list[dict[str, Any]], batch_size: int = 64
) -> list[float]:
    return detector.anomaly_scores([r["seq"] for r in records], batch_size=batch_size)


def ontology_scores(scorer: Any, records: list[dict[str, Any]]) -> list[float]:
    out = []
    for r in records:
        try:
            row = {
                "codes": r["seq"],
                "gender": r["gender"],
                "age_group": r["age_group"],
            }
            out.append(float(scorer.score(row, s_det=0.0)["s_ont"]))
        except Exception:
            out.append(0.0)
    return out


def minmax_fit(values: list[float]) -> tuple[float, float]:
    return (min(values), max(values)) if values else (0.0, 1.0)


def minmax_apply(values: list[float], lo: float, hi: float) -> list[float]:
    rng = hi - lo
    if rng <= 0:
        return [0.0 for _ in values]
    return [max(0.0, min(1.0, (v - lo) / rng)) for v in values]


def combined_scores(
    s_det_norm: list[float], s_ont: list[float], weights: Any
) -> list[float]:
    """Sgen-free calibrated score (w_gen=0): S_cal = (w_det*S_det + w_ont*S_ont')/Σ."""
    from src.scoring.ontology_aware import ScoreWeights, compute_calibrated_score

    w = ScoreWeights(w_det=weights.w_det, w_ont=weights.w_ont, w_gen=0.0)
    return [
        compute_calibrated_score(d, o, s_gen=None, weights=w, include_ont=True)
        for d, o in zip(s_det_norm, s_ont)
    ]


__all__ = [
    "load_records",
    "detector_scores",
    "ontology_scores",
    "minmax_fit",
    "minmax_apply",
    "combined_scores",
]
