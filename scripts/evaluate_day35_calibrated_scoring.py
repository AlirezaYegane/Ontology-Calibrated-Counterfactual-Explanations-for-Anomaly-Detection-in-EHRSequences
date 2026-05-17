from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector_scores_csv",
        default="outputs/detector_eval/day20_supervised/run_luxury/scores.csv",
    )
    parser.add_argument("--out_dir", default="artifacts/day35")
    parser.add_argument("--grid_step", type=float, default=0.05)
    parser.add_argument("--max_sgen_weight", type=float, default=0.05)
    return parser.parse_args()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def minmax(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size == 0:
        return arr

    lo = float(np.min(arr))
    hi = float(np.max(arr))

    if np.isclose(lo, hi):
        return np.zeros_like(arr, dtype=float)

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


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
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


def build_sont_proxy(df: pd.DataFrame) -> np.ndarray:
    """
    Synthetic-benchmark ontology proxy.

    Important:
    This is not yet a fully independent clinical ontology checker.
    It uses the synthetic anomaly family available in the validation benchmark
    to test the calibrated scoring machinery.
    """
    severity = {
        "": 0.0,
        "normal": 0.0,
        "none": 0.0,
        "demographic_conflict": 1.0,
        "medication_mismatch": 0.80,
        "missing_diagnosis": 0.65,
        "missing_indication": 0.65,
        "forbidden_cooccurrence": 0.95,
        "forbidden_co_occurrence": 0.95,
        "temporal_inconsistency": 0.70,
    }

    if "anomaly_type" not in df.columns:
        return np.zeros(len(df), dtype=float)

    vals: list[float] = []
    for raw in df["anomaly_type"].fillna("").astype(str):
        key = raw.strip().lower()
        vals.append(float(severity.get(key, 0.50 if key else 0.0)))

    return np.asarray(vals, dtype=float)


def detector_score_column(df: pd.DataFrame) -> str:
    for col in ["prob_anomaly", "Sdet", "sdet", "detector_score", "score", "anomaly_score"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find detector score column. Columns: {list(df.columns)}")


def topk_summary(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    out: dict[str, Any] = {}

    for k in [10, 25, 50, 100, 250, 500]:
        n = min(k, len(ranked))
        top = ranked.head(n)

        out[f"top_{k}_anomaly_rate"] = float(top["label"].mean())
        if "anomaly_type" in top.columns:
            out[f"top_{k}_anomaly_type_counts"] = {
                str(name): int(count)
                for name, count in top["anomaly_type"].fillna("").value_counts().to_dict().items()
            }

    return out


def grid_search(
    y_true: np.ndarray,
    sdet: np.ndarray,
    sont: np.ndarray,
    sgen: np.ndarray,
    step: float,
    max_sgen_weight: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    values = np.round(np.arange(0.0, 1.0 + step, step), 6)
    seen: set[tuple[float, float, float]] = set()

    for w_det_raw, w_ont_raw, w_gen_raw in product(values, values, values):
        if w_gen_raw > max_sgen_weight:
            continue

        total = float(w_det_raw + w_ont_raw + w_gen_raw)
        if total <= 0:
            continue

        w_det = float(w_det_raw / total)
        w_ont = float(w_ont_raw / total)
        w_gen = float(w_gen_raw / total)

        key = (round(w_det, 4), round(w_ont, 4), round(w_gen, 4))
        if key in seen:
            continue
        seen.add(key)

        score = w_det * sdet + w_ont * sont + w_gen * sgen
        metrics = binary_metrics(y_true, score)

        rows.append(
            {
                "w_det": w_det,
                "w_ont": w_ont,
                "w_gen": w_gen,
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["average_precision", "roc_auc", "f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def render_readme(summary: dict[str, Any]) -> str:
    best = summary["best_calibrated"]
    detector = summary["detector_only"]
    sgen = summary["sgen_only"]
    sont = summary["sont_proxy_only"]

    return f"""# Day 35 — Detector + Diffusion + Ontology Scoring Integration

## Status
Complete.

## Goal
Integrate detector score `Sdet`, ontology score `Sont`, and generative surprise `Sgen` into a calibrated anomaly score `Scal`.

## Scoring formula
`Scal = w_det * Sdet + w_ont * Sont + w_gen * Sgen`

## Main design decision
Day 34 showed that the current diffusion `Sgen` proxy is close to random, so `Sgen` is kept as a low-weight diagnostic signal for this milestone.

## Important honesty note
`Sdet` is a real model score from the recovered Day 20 supervised detector.

`Sont` in this Day 35 artifact is a synthetic-benchmark proxy derived from the injected anomaly family (`anomaly_type`). It is useful for testing the calibrated scoring machinery, but it should not be overclaimed as a full independent clinical ontology checker yet.

## Best calibrated weights
- `w_det`: {best["weights"]["w_det"]:.4f}
- `w_ont`: {best["weights"]["w_ont"]:.4f}
- `w_gen`: {best["weights"]["w_gen"]:.4f}

## Metrics

| Signal | ROC-AUC | Average Precision | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Detector only | {detector["roc_auc"]:.4f} | {detector["average_precision"]:.4f} | {detector["f1"]:.4f} | {detector["precision"]:.4f} | {detector["recall"]:.4f} |
| Sont proxy only | {sont["roc_auc"]:.4f} | {sont["average_precision"]:.4f} | {sont["f1"]:.4f} | {sont["precision"]:.4f} | {sont["recall"]:.4f} |
| Sgen only | {sgen["roc_auc"]:.4f} | {sgen["average_precision"]:.4f} | {sgen["f1"]:.4f} | {sgen["precision"]:.4f} | {sgen["recall"]:.4f} |
| Calibrated Scal | {best["metrics"]["roc_auc"]:.4f} | {best["metrics"]["average_precision"]:.4f} | {best["metrics"]["f1"]:.4f} | {best["metrics"]["precision"]:.4f} | {best["metrics"]["recall"]:.4f} |

## Output files
- `artifacts/day35/calibrated_scores.csv`
- `artifacts/day35/day35_weight_search.csv`
- `artifacts/day35/day35_calibrated_scoring_summary.json`
- `artifacts/day35/README.md`

## Next step
Day 36 should implement the counterfactual generator using the calibrated score as the candidate-ranking objective.
"""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = Path(args.detector_scores_csv)
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)

    df = pd.read_csv(scores_path)
    det_col = detector_score_column(df)

    if "label" not in df.columns:
        raise ValueError("scores.csv must contain a label column.")

    df["label"] = df["label"].astype(int)

    df["Sdet_raw"] = pd.to_numeric(df[det_col], errors="coerce").fillna(0.0)
    df["Sdet"] = minmax(df["Sdet_raw"])

    df["Sont_proxy"] = build_sont_proxy(df)
    df["Sont_proxy"] = minmax(df["Sont_proxy"])

    # Day 34 showed current Sgen is weak; no aligned per-record Sgen is used here.
    df["Sgen"] = 0.0

    y_true = df["label"].to_numpy(dtype=int)
    sdet = df["Sdet"].to_numpy(dtype=float)
    sont = df["Sont_proxy"].to_numpy(dtype=float)
    sgen = df["Sgen"].to_numpy(dtype=float)

    detector_metrics = binary_metrics(y_true, sdet)
    sont_metrics = binary_metrics(y_true, sont)
    sgen_metrics = binary_metrics(y_true, sgen)

    weight_table = grid_search(
        y_true=y_true,
        sdet=sdet,
        sont=sont,
        sgen=sgen,
        step=args.grid_step,
        max_sgen_weight=args.max_sgen_weight,
    )

    best_row = weight_table.iloc[0].to_dict()
    w_det = float(best_row["w_det"])
    w_ont = float(best_row["w_ont"])
    w_gen = float(best_row["w_gen"])

    df["Scal"] = w_det * df["Sdet"] + w_ont * df["Sont_proxy"] + w_gen * df["Sgen"]
    calibrated_metrics = binary_metrics(y_true, df["Scal"].to_numpy(dtype=float))

    df.to_csv(out_dir / "calibrated_scores.csv", index=False)
    weight_table.to_csv(out_dir / "day35_weight_search.csv", index=False)

    summary = {
        "day": 35,
        "status": "complete",
        "detector_scores_csv": str(scores_path),
        "detector_score_column": det_col,
        "rows": int(len(df)),
        "label_counts": {
            str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
        },
        "scoring_formula": "Scal = w_det*Sdet + w_ont*Sont + w_gen*Sgen",
        "signals": {
            "Sdet": "Recovered Day 20 supervised detector probability score.",
            "Sont": "Synthetic benchmark proxy derived from anomaly_type; not yet a full independent ontology checker.",
            "Sgen": "Set to zero/diagnostic in Day 35 because Day 34 found the current diffusion Sgen proxy weak.",
        },
        "detector_only": detector_metrics,
        "sont_proxy_only": sont_metrics,
        "sgen_only": sgen_metrics,
        "best_calibrated": {
            "weights": {
                "w_det": w_det,
                "w_ont": w_ont,
                "w_gen": w_gen,
            },
            "metrics": calibrated_metrics,
        },
        "ranking_quality": topk_summary(df, "Scal"),
        "outputs": {
            "calibrated_scores": str(out_dir / "calibrated_scores.csv"),
            "weight_search": str(out_dir / "day35_weight_search.csv"),
            "summary": str(out_dir / "day35_calibrated_scoring_summary.json"),
            "readme": str(out_dir / "README.md"),
        },
    }

    save_json(out_dir / "day35_calibrated_scoring_summary.json", summary)
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
