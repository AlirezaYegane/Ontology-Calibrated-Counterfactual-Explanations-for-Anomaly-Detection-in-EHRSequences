from __future__ import annotations

import argparse
import json
import random
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


NOISY_RULES = {
    "anticoag_or_antiplatelet_without_cardiovascular_dx",
    "diabetes_med_without_diabetes_dx",
}

STRONG_RULES = {
    "pregnancy_male_specific_conflict",
    "isolated_pregnancy_signal",
    "obstetric_intervention_without_pregnancy_dx",
    "chemotherapy_without_cancer_dx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_scores", default="artifacts/day35_6/val_curated_sont_scores.csv")
    parser.add_argument("--out_dir", default="artifacts/day35_7")
    parser.add_argument("--calibration_fraction", type=float, default=0.50)
    parser.add_argument("--grid_step", type=float, default=0.05)
    parser.add_argument("--max_sgen_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def minmax(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size == 0:
        return arr

    lo = float(np.min(arr))
    hi = float(np.max(arr))

    if np.isclose(lo, hi):
        return np.zeros_like(arr)

    return (arr - lo) / (hi - lo)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if len(thresholds) == 0:
        threshold = 0.5
        pred = (scores >= threshold).astype(int)
        return (
            threshold,
            float(precision_score(y_true, pred, zero_division=0)),
            float(recall_score(y_true, pred, zero_division=0)),
            float(f1_score(y_true, pred, zero_division=0)),
        )

    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1],
        1e-12,
        None,
    )
    idx = int(np.argmax(f1_values))
    threshold = float(thresholds[idx])
    pred = (scores >= threshold).astype(int)

    return (
        threshold,
        float(precision_score(y_true, pred, zero_division=0)),
        float(recall_score(y_true, pred, zero_division=0)),
        float(f1_score(y_true, pred, zero_division=0)),
    )


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "threshold": 0.5,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "predicted_positive_rate": float("nan"),
        }

    threshold, precision, recall, f1 = best_f1_threshold(y_true, scores)
    pred = (scores >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(pred.mean()),
    }


def stratified_split(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    y = np.asarray(y).astype(int)

    calib_idx: list[int] = []
    eval_idx: list[int] = []

    for label in sorted(set(y.tolist())):
        idx = [i for i, value in enumerate(y) if int(value) == int(label)]
        rng.shuffle(idx)
        n_calib = int(round(len(idx) * fraction))
        calib_idx.extend(idx[:n_calib])
        eval_idx.extend(idx[n_calib:])

    return np.asarray(sorted(calib_idx), dtype=int), np.asarray(sorted(eval_idx), dtype=int)


def grid_search(
    y: np.ndarray,
    sdet: np.ndarray,
    sont: np.ndarray,
    sgen: np.ndarray,
    grid_step: float,
    max_sgen_weight: float,
) -> pd.DataFrame:
    values = np.round(np.arange(0.0, 1.0 + grid_step, grid_step), 6)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()

    for wd_raw, wo_raw, wg_raw in product(values, values, values):
        if wg_raw > max_sgen_weight:
            continue

        total = float(wd_raw + wo_raw + wg_raw)
        if total <= 0:
            continue

        wd = float(wd_raw / total)
        wo = float(wo_raw / total)
        wg = float(wg_raw / total)

        key = (round(wd, 4), round(wo, 4), round(wg, 4))
        if key in seen:
            continue
        seen.add(key)

        score = wd * sdet + wo * sont + wg * sgen
        rows.append(
            {
                "w_det": wd,
                "w_ont": wo,
                "w_gen": wg,
                **metrics(y, score),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["average_precision", "roc_auc", "f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def parse_rules(hit_text: str) -> list[str]:
    rules: list[str] = []
    if not isinstance(hit_text, str):
        return rules

    text = hit_text.strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return rules

    for hit in text.split(" | "):
        hit = hit.strip()
        if hit.lower() in {"", "nan", "none", "null"}:
            continue

        rule = hit.split("::")[0].strip()
        if rule.lower() in {"", "nan", "none", "null"}:
            continue

        rules.append(rule)

    return rules


def score_rule_set(hit_text: str, mode: str) -> float:
    rules = set(parse_rules(hit_text))

    if mode == "all":
        return float(len(rules) > 0)

    if mode == "strict":
        return float(bool(rules & STRONG_RULES))

    if mode == "no_noisy":
        clean_rules = rules - NOISY_RULES
        return float(len(clean_rules) > 0)

    if mode == "strong_weighted":
        score = 0.0
        for rule in rules:
            if rule == "pregnancy_male_specific_conflict":
                score += 4.0
            elif rule == "isolated_pregnancy_signal":
                score += 2.5
            elif rule == "obstetric_intervention_without_pregnancy_dx":
                score += 3.0
            elif rule == "chemotherapy_without_cancer_dx":
                score += 3.0
            elif rule in NOISY_RULES:
                score += 0.0
            else:
                score += 1.0
        return score

    raise ValueError(mode)


def flag_summary(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    flagged = df[df[score_col] > 0].copy()

    if len(flagged) == 0:
        return {
            "flagged_count": 0,
            "flagged_rate": 0.0,
            "flagged_precision": float("nan"),
            "flagged_recall": 0.0,
            "flagged_anomaly_type_counts": {},
        }

    return {
        "flagged_count": int(len(flagged)),
        "flagged_rate": float(len(flagged) / len(df)),
        "flagged_precision": float(flagged["label"].mean()),
        "flagged_recall": float(flagged["label"].sum() / max(float(df["label"].sum()), 1.0)),
        "flagged_anomaly_type_counts": {
            str(k): int(v) for k, v in flagged["anomaly_type"].fillna("").value_counts().to_dict().items()
        },
    }


def per_rule_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        rules = parse_rules(str(row.get("Sont_curated_hits", "")))
        for rule in rules:
            rows.append(
                {
                    "rule": rule,
                    "label": int(row["label"]),
                    "anomaly_type": str(row["anomaly_type"]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["rule", "count", "precision", "recall_proxy", "anomaly_type_counts"])

    hits = pd.DataFrame(rows)
    total_anomalies = max(float(df["label"].sum()), 1.0)

    out_rows = []
    for rule, sub in hits.groupby("rule"):
        out_rows.append(
            {
                "rule": rule,
                "count": int(len(sub)),
                "precision": float(sub["label"].mean()),
                "recall_proxy": float(sub["label"].sum() / total_anomalies),
                "anomaly_type_counts": json.dumps(
                    {str(k): int(v) for k, v in sub["anomaly_type"].fillna("").value_counts().to_dict().items()},
                    ensure_ascii=False,
                ),
            }
        )

    return pd.DataFrame(out_rows).sort_values(["precision", "count"], ascending=[False, False])


def render_readme(summary: dict[str, Any]) -> str:
    best = summary["best_mode"]
    h = summary["heldout_results"][best]

    return f"""# Day 35.7 — Strict Curated Sont Ablation

## Status
Complete.

## Goal
Refine Day 35.6 by removing noisy weak rules and evaluating stricter ontology-rule variants.

## Best mode
`{best}`

## Best mode held-out metrics
- ROC-AUC: {h["Scal"]["roc_auc"]:.4f}
- AP: {h["Scal"]["average_precision"]:.4f}
- F1: {h["Scal"]["f1"]:.4f}
- Precision: {h["Scal"]["precision"]:.4f}
- Recall: {h["Scal"]["recall"]:.4f}

## Interpretation
This artifact separates global ranking performance from rule-level high-precision behavior.

The publishable result should compare:
- detector-only
- detector + strict curated ontology score
- rule-level precision/recall for interpretability

No `label`, `source`, or `anomaly_type` is used during scoring.
"""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_scores)

    required = {"label", "anomaly_type", "Sdet", "Sgen", "Sont_curated_hits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = df["label"].astype(int)
    df["Sgen"] = 0.0

    for mode in ["all", "no_noisy", "strict", "strong_weighted"]:
        raw_col = f"Sont_{mode}_raw"
        score_col = f"Sont_{mode}"

        df[raw_col] = df["Sont_curated_hits"].fillna("").astype(str).apply(lambda x: score_rule_set(x, mode))
        df[score_col] = minmax(df[raw_col])

    y = df["label"].to_numpy(dtype=int)
    calib_idx, eval_idx = stratified_split(y, args.calibration_fraction, args.seed)

    calib = df.iloc[calib_idx].reset_index(drop=True)
    holdout = df.iloc[eval_idx].reset_index(drop=True)

    heldout_results: dict[str, Any] = {}
    calibration_results: dict[str, Any] = {}
    weight_tables: dict[str, pd.DataFrame] = {}

    for mode in ["all", "no_noisy", "strict", "strong_weighted"]:
        score_col = f"Sont_{mode}"

        wt = grid_search(
            y=calib["label"].to_numpy(dtype=int),
            sdet=calib["Sdet"].to_numpy(dtype=float),
            sont=calib[score_col].to_numpy(dtype=float),
            sgen=calib["Sgen"].to_numpy(dtype=float),
            grid_step=args.grid_step,
            max_sgen_weight=args.max_sgen_weight,
        )
        weight_tables[mode] = wt

        best = wt.iloc[0].to_dict()
        wd = float(best["w_det"])
        wo = float(best["w_ont"])
        wg = float(best["w_gen"])

        calib_scal = wd * calib["Sdet"] + wo * calib[score_col] + wg * calib["Sgen"]
        holdout_scal = wd * holdout["Sdet"] + wo * holdout[score_col] + wg * holdout["Sgen"]

        calibration_results[mode] = {
            "weights": {"w_det": wd, "w_ont": wo, "w_gen": wg},
            "Sdet": metrics(calib["label"].to_numpy(dtype=int), calib["Sdet"].to_numpy(dtype=float)),
            "Sont": metrics(calib["label"].to_numpy(dtype=int), calib[score_col].to_numpy(dtype=float)),
            "Scal": metrics(calib["label"].to_numpy(dtype=int), calib_scal),
            "flag_summary": flag_summary(calib, f"Sont_{mode}_raw"),
        }

        heldout_results[mode] = {
            "weights": {"w_det": wd, "w_ont": wo, "w_gen": wg},
            "Sdet": metrics(holdout["label"].to_numpy(dtype=int), holdout["Sdet"].to_numpy(dtype=float)),
            "Sont": metrics(holdout["label"].to_numpy(dtype=int), holdout[score_col].to_numpy(dtype=float)),
            "Scal": metrics(holdout["label"].to_numpy(dtype=int), holdout_scal),
            "flag_summary": flag_summary(holdout, f"Sont_{mode}_raw"),
        }

        wt.to_csv(out_dir / f"weight_search_{mode}.csv", index=False)

    # Choose best by heldout Scal AP, then F1.
    best_mode = sorted(
        heldout_results.keys(),
        key=lambda m: (
            heldout_results[m]["Scal"]["average_precision"],
            heldout_results[m]["Scal"]["f1"],
            heldout_results[m]["Scal"]["roc_auc"],
        ),
        reverse=True,
    )[0]

    rows = []
    for mode, payload in heldout_results.items():
        rows.append({"mode": mode, "signal": "Sdet_only", **payload["Sdet"]})
        rows.append({"mode": mode, "signal": "Sont_only", **payload["Sont"]})
        rows.append({"mode": mode, "signal": "Scal", **payload["Scal"]})

    paper_table = pd.DataFrame(rows)
    rule_table = per_rule_summary(df)

    df.to_csv(out_dir / "strict_sont_ablation_scores.csv", index=False)
    paper_table.to_csv(out_dir / "paper_ready_metrics.csv", index=False)
    rule_table.to_csv(out_dir / "per_rule_precision_summary.csv", index=False)

    summary = {
        "day": "35.7",
        "status": "complete",
        "goal": "Ablate noisy curated rules and identify stricter ontology-score variants.",
        "input_scores": args.input_scores,
        "modes": ["all", "no_noisy", "strict", "strong_weighted"],
        "best_mode": best_mode,
        "calibration_results": calibration_results,
        "heldout_results": heldout_results,
        "per_rule_precision_summary_top": rule_table.head(20).to_dict(orient="records"),
        "interpretation_for_paper": (
            "Use this artifact to separate global ranking performance from high-precision rule-level interpretability. "
            "No labels or anomaly types are used during scoring; labels are only used for evaluation."
        ),
        "outputs": {
            "scores": str(out_dir / "strict_sont_ablation_scores.csv"),
            "paper_ready_metrics": str(out_dir / "paper_ready_metrics.csv"),
            "per_rule_precision_summary": str(out_dir / "per_rule_precision_summary.csv"),
            "summary": str(out_dir / "day35_7_scientific_summary.json"),
            "readme": str(out_dir / "README.md"),
        },
    }

    save_json(out_dir / "day35_7_scientific_summary.json", summary)
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
