"""
src/evaluation/stats.py
=======================
Phase 3 -- Statistical comparison utilities: bootstrap CIs, paired bootstrap
differences, and a bootstrap p-value. NumPy-based; uses sklearn metrics when
available with a NumPy fallback so tests don't hard-require sklearn.

These let later phases attach confidence intervals and significance to every
ablation delta (no "improvement" claimed without a CI / p-value).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from sklearn.metrics import average_precision_score as _sk_ap
    from sklearn.metrics import roc_auc_score as _sk_auc

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _sk_auc = _sk_ap = None
    _HAVE_SKLEARN = False


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC (sklearn if present, else Mann-Whitney U with tie handling)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both classes required for AUC.")
    if _HAVE_SKLEARN:
        return float(_sk_auc(y_true, scores))
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        return 0.0
    if _HAVE_SKLEARN:
        return float(_sk_ap(y_true, scores))
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    cum_tp = np.cumsum(y)
    precision = cum_tp / (np.arange(len(y)) + 1.0)
    recall = cum_tp / y.sum()
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    scores,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for a metric. Safe on degenerate resamples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    try:
        point = float(metric_fn(y, s))
    except Exception:
        # degenerate (e.g. single-class) -> no valid metric
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot_valid": 0,
        }
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boots.append(float(metric_fn(yb, sb)))
        except Exception:
            continue
    if not boots:
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot_valid": 0,
        }
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot_valid": len(boots)}


def bootstrap_auc_ap(
    y_true, scores, n_boot: int = 1000, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Convenience: bootstrap CIs for both ROC-AUC and AP."""
    return {
        "roc_auc": bootstrap_ci(roc_auc, y_true, scores, n_boot=n_boot, seed=seed),
        "average_precision": bootstrap_ci(
            average_precision, y_true, scores, n_boot=n_boot, seed=seed
        ),
    }


def paired_bootstrap_diff(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    scores_a,
    scores_b,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of metric(A) - metric(B) on the SAME resampled records.

    Returns the observed difference, a CI, and a two-sided bootstrap p-value for
    H0: difference == 0.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    n = len(y)
    observed = float(metric_fn(y, a) - metric_fn(y, b))
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            diffs.append(float(metric_fn(yb, a[idx]) - metric_fn(yb, b[idx])))
        except Exception:
            continue
    if not diffs:
        return {
            "observed_diff": observed,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n_boot_valid": 0,
        }
    arr = np.asarray(diffs)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    # two-sided bootstrap p-value: fraction of resamples on the opposite side of 0
    p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
    p = min(1.0, p)
    return {
        "observed_diff": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "n_boot_valid": len(diffs),
    }


__all__ = [
    "roc_auc",
    "average_precision",
    "bootstrap_ci",
    "bootstrap_auc_ap",
    "paired_bootstrap_diff",
]
