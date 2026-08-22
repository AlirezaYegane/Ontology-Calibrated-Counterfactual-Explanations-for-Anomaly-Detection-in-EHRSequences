from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ensure_validation_only_paths(
    train_path: str,
    val_path: str,
) -> None:
    forbidden = {"test.pkl", "benchmark_test.pkl"}

    train_name = str(train_path).replace("\\", "/").split("/")[-1].lower()
    val_name = str(val_path).replace("\\", "/").split("/")[-1].lower()

    if train_name in forbidden:
        raise ValueError("Test data cannot be used as training data.")

    if val_name in forbidden:
        raise ValueError(
            "Paper Phase 2 Day 2 is validation-only. Do not pass test.pkl."
        )


def discrimination_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> float | None:
    if x.nunique(dropna=True) < 2:
        return None
    if y.nunique(dropna=True) < 2:
        return None

    value = x.corr(y, method="spearman")

    if pd.isna(value):
        return None

    return float(value)


def topk_mean(
    values: np.ndarray,
    k: int,
) -> float:
    if len(values) == 0:
        return 0.0

    k_eff = max(1, min(int(k), len(values)))
    return float(np.mean(np.sort(values)[-k_eff:]))


def aggregate_upper_scores(
    values: np.ndarray,
    quantiles: list[float],
    top_ks: list[int],
) -> dict[str, float]:
    if len(values) == 0:
        output = {
            "worst": 0.0,
            "mean": 0.0,
        }

        for q in quantiles:
            output[f"q{int(round(q * 100))}"] = 0.0

        for k in top_ks:
            output[f"topk{k}"] = 0.0

        return output

    output = {
        "worst": float(np.max(values)),
        "mean": float(np.mean(values)),
    }

    for q in quantiles:
        output[f"q{int(round(q * 100))}"] = float(
            np.quantile(values, q)
        )

    for k in top_ks:
        output[f"topk{k}"] = topk_mean(values, k)

    return output


def evaluate_candidate(
    *,
    candidate_id: str,
    model: str,
    score_family: str,
    aggregation: str,
    parameters: dict[str, Any],
    is_extreme: bool,
    labels: np.ndarray,
    scores: np.ndarray,
    sequence_lengths: np.ndarray,
    candidate_pair_counts: np.ndarray,
    length_bins: np.ndarray,
    anomaly_types: np.ndarray,
) -> dict[str, Any]:
    overall = discrimination_metrics(labels, scores)

    frame = pd.DataFrame(
        {
            "label": labels,
            "score": scores,
            "sequence_length": sequence_lengths,
            "candidate_pair_count": candidate_pair_counts,
            "length_bin": length_bins,
            "anomaly_type": anomaly_types,
        }
    )

    rho_length = safe_spearman(
        frame["score"],
        frame["sequence_length"],
    )
    rho_pairs = safe_spearman(
        frame["score"],
        frame["candidate_pair_count"],
    )

    abs_values = [
        abs(value)
        for value in (rho_length, rho_pairs)
        if value is not None
    ]

    max_abs_bias = max(abs_values) if abs_values else None

    length_metrics: dict[str, Any] = {}

    for bin_name in ("short", "medium", "long"):
        subset = frame[frame["length_bin"] == bin_name]

        payload: dict[str, Any] = {
            "n": int(len(subset)),
            "n_anomaly": int(subset["label"].sum()),
            "anomaly_rate": (
                float(subset["label"].mean())
                if len(subset)
                else None
            ),
        }

        if len(subset) > 0 and subset["label"].nunique() == 2:
            payload.update(
                discrimination_metrics(
                    subset["label"].to_numpy(dtype=int),
                    subset["score"].to_numpy(dtype=float),
                )
            )
        else:
            payload["pr_auc"] = None
            payload["roc_auc"] = None

        length_metrics[bin_name] = payload

    family_metrics: dict[str, Any] = {}

    for family, subset in frame.groupby("anomaly_type"):
        family_metrics[str(family)] = {
            "n": int(len(subset)),
            "n_anomaly": int(subset["label"].sum()),
            "mean": float(subset["score"].mean()),
            "median": float(subset["score"].median()),
        }

    return {
        "candidate_id": candidate_id,
        "model": model,
        "score_family": score_family,
        "aggregation": aggregation,
        "parameters": parameters,
        "is_extreme": bool(is_extreme),
        "overall": overall,
        "bias": {
            "spearman_vs_sequence_length": rho_length,
            "spearman_vs_candidate_pair_count": rho_pairs,
            "max_abs_spearman": max_abs_bias,
        },
        "length_bins": length_metrics,
        "anomaly_family_breakdown": family_metrics,
    }


def selection_key(
    candidate: dict[str, Any],
) -> tuple[float, float, float, str]:
    """
    Pre-run Day-2 operational policy.

    1) maximise validation PR-AUC
    2) minimise observed max absolute length/pair correlation
    3) maximise ROC-AUC
    4) stable candidate-id tie break

    Extreme max/worst candidates are excluded before this is called.
    """
    pr_auc = float(candidate["overall"]["pr_auc"])
    roc_auc = float(candidate["overall"]["roc_auc"])

    bias = candidate["bias"]["max_abs_spearman"]
    bias_value = float(bias) if bias is not None else float("inf")

    return (
        -pr_auc,
        bias_value,
        -roc_auc,
        str(candidate["candidate_id"]),
    )


def select_best_non_extreme(
    candidates: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["model"] == model
        and not candidate["is_extreme"]
    ]

    if not eligible:
        raise ValueError(
            f"No non-extreme candidates available for {model}."
        )

    return sorted(eligible, key=selection_key)[0]


def select_best_extreme(
    candidates: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["model"] == model
        and candidate["is_extreme"]
    ]

    if not eligible:
        raise ValueError(
            f"No extreme candidates available for {model}."
        )

    return sorted(
        eligible,
        key=lambda candidate: (
            -float(candidate["overall"]["pr_auc"]),
            -float(candidate["overall"]["roc_auc"]),
            str(candidate["candidate_id"]),
        ),
    )[0]


def compare_selected_to_extreme(
    selected: dict[str, Any],
    extreme: dict[str, Any],
) -> dict[str, Any]:
    selected_bias = selected["bias"]["max_abs_spearman"]
    extreme_bias = extreme["bias"]["max_abs_spearman"]

    bias_delta = None

    if selected_bias is not None and extreme_bias is not None:
        bias_delta = float(selected_bias - extreme_bias)

    return {
        "selected_candidate_id": selected["candidate_id"],
        "extreme_candidate_id": extreme["candidate_id"],
        "pr_auc_delta_selected_minus_extreme": float(
            selected["overall"]["pr_auc"]
            - extreme["overall"]["pr_auc"]
        ),
        "roc_auc_delta_selected_minus_extreme": float(
            selected["overall"]["roc_auc"]
            - extreme["overall"]["roc_auc"]
        ),
        "max_abs_bias_selected": selected_bias,
        "max_abs_bias_extreme": extreme_bias,
        "max_abs_bias_delta_selected_minus_extreme": bias_delta,
        "bias_reduced": (
            bool(bias_delta < 0.0)
            if bias_delta is not None
            else None
        ),
    }
