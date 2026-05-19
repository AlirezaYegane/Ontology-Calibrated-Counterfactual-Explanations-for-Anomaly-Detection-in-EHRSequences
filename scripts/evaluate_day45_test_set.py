from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


SCORE_CANDIDATES = [
    "scal",
    "S_cal",
    "Scal",
    "calibrated_score",
    "score",
    "anomaly_score",
    "full_model_conservative",
    "full_model",
    "sdet",
    "Sdet",
    "detector_score",
]

LABEL_CANDIDATES = [
    "label",
    "is_anomaly",
    "y_true",
    "target",
    "ground_truth",
    "synthetic_label",
    "anomaly_label",
]

TYPE_CANDIDATES = [
    "anomaly_type",
    "synthetic_anomaly_type",
    "type",
    "case_type",
    "label_name",
]

ID_CANDIDATES = [
    "record_id",
    "row_id",
    "hadm_id",
    "subject_id",
    "index",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def find_candidate_files(root: Path) -> list[Path]:
    patterns = [
        "*score*.csv",
        "*scores*.csv",
        "*eval*.csv",
        "*ablation*.csv",
        "*case*.csv",
    ]
    seen: set[Path] = set()
    candidates: list[Path] = []

    artifact_root = root / "artifacts"
    if not artifact_root.exists():
        return []

    for pattern in patterns:
        for path in artifact_root.rglob(pattern):
            if path.is_file() and path not in seen:
                lowered = str(path).lower()
                if "day45" in lowered:
                    continue
                if any(
                    skip in lowered for skip in ["threshold", "roc_curve", "pr_curve"]
                ):
                    continue
                seen.add(path)
                candidates.append(path)

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def pick_column(
    df: pd.DataFrame, explicit: str | None, candidates: list[str], role: str
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested {role} column '{explicit}' was not found. Columns: {list(df.columns)}"
            )
        return explicit

    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]

    raise ValueError(f"Could not infer {role} column. Columns: {list(df.columns)}")


def maybe_pick_column(
    df: pd.DataFrame, explicit: str | None, candidates: list[str]
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested column '{explicit}' was not found. Columns: {list(df.columns)}"
            )
        return explicit

    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def infer_label(
    df: pd.DataFrame, label_col: str | None, type_col: str | None
) -> tuple[pd.Series, str]:
    if label_col is not None:
        raw = df[label_col]
        if pd.api.types.is_numeric_dtype(raw):
            y = raw.fillna(0).astype(int)
            return y.clip(lower=0, upper=1), label_col

        normalized = raw.astype(str).str.strip().str.lower()
        positive = {
            "1",
            "true",
            "yes",
            "anomaly",
            "abnormal",
            "synthetic",
            "positive",
            "pos",
        }
        negative = {"0", "false", "no", "normal", "negative", "neg", "none"}

        y = normalized.map(
            lambda x: 1 if x in positive else 0 if x in negative else np.nan
        )
        if y.isna().any():
            unknown = sorted(normalized[y.isna()].unique().tolist())[:20]
            raise ValueError(
                f"Could not convert label column '{label_col}' to binary labels. Unknown values: {unknown}"
            )
        return y.astype(int), label_col

    if type_col is not None:
        normalized = df[type_col].astype(str).str.strip().str.lower()
        y = (~normalized.isin(["normal", "none", "nan", "negative", "0"])).astype(int)
        return y, f"inferred_from_{type_col}"

    raise ValueError(
        "No label column was found and no anomaly_type column was available for inference."
    )


def threshold_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "accuracy": float(accuracy),
        "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
    }


def build_threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    lo = float(np.nanmin(scores))
    hi = float(np.nanmax(scores))

    grid = np.linspace(lo, hi, 401)
    unique_scores = np.unique(scores)

    if len(unique_scores) <= 2000:
        thresholds = np.unique(np.concatenate([grid, unique_scores]))
    else:
        quantiles = np.quantile(scores, np.linspace(0, 1, 401))
        thresholds = np.unique(np.concatenate([grid, quantiles]))

    rows = [threshold_metrics(y_true, scores, float(t)) for t in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def choose_threshold(
    sweep: pd.DataFrame,
    max_fpr: float,
    min_precision: float,
) -> tuple[dict[str, Any], str]:
    viable = sweep[
        (sweep["false_positive_rate"] <= max_fpr)
        & (sweep["precision"] >= min_precision)
        & (sweep["tp"] > 0)
    ].copy()

    if len(viable):
        viable = viable.sort_values(
            ["recall", "f1", "precision", "threshold"],
            ascending=[False, False, False, False],
        )
        return viable.iloc[
            0
        ].to_dict(), "conservative_threshold_with_fpr_and_precision_constraints"

    fallback = sweep.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
    )
    return fallback.iloc[0].to_dict(), "fallback_best_f1_threshold_constraints_not_met"


def build_by_type_analysis(
    df: pd.DataFrame,
    type_col: str | None,
    score_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    if type_col is None:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["_y_true"] = y_true
    tmp["_y_pred"] = y_pred
    tmp["_fp"] = ((tmp["_y_true"] == 0) & (tmp["_y_pred"] == 1)).astype(int)
    tmp["_fn"] = ((tmp["_y_true"] == 1) & (tmp["_y_pred"] == 0)).astype(int)
    tmp["_tp"] = ((tmp["_y_true"] == 1) & (tmp["_y_pred"] == 1)).astype(int)
    tmp["_tn"] = ((tmp["_y_true"] == 0) & (tmp["_y_pred"] == 0)).astype(int)

    grouped = (
        tmp.groupby(type_col, dropna=False)
        .agg(
            n_records=(score_col, "size"),
            positives=("_y_true", "sum"),
            predicted_positive=("_y_pred", "sum"),
            mean_score=(score_col, "mean"),
            median_score=(score_col, "median"),
            min_score=(score_col, "min"),
            max_score=(score_col, "max"),
            tp=("_tp", "sum"),
            fp=("_fp", "sum"),
            fn=("_fn", "sum"),
            tn=("_tn", "sum"),
        )
        .reset_index()
    )

    grouped["type_precision"] = grouped.apply(
        lambda r: r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) else 0.0,
        axis=1,
    )
    grouped["type_recall"] = grouped.apply(
        lambda r: r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) else 0.0,
        axis=1,
    )
    grouped["type_fpr"] = grouped.apply(
        lambda r: r["fp"] / (r["fp"] + r["tn"]) if (r["fp"] + r["tn"]) else 0.0,
        axis=1,
    )

    return grouped.sort_values(
        ["fn", "fp", "n_records"], ascending=[False, False, False]
    )


def write_curve_points(
    out_dir: Path, y_true: np.ndarray, scores: np.ndarray
) -> dict[str, str | None]:
    written: dict[str, str | None] = {"roc_curve_points": None, "pr_curve_points": None}

    if len(np.unique(y_true)) < 2:
        return written

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_df = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_thresholds,
        }
    )
    roc_path = out_dir / "roc_curve_points.csv"
    roc_df.to_csv(roc_path, index=False)
    written["roc_curve_points"] = str(roc_path)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_thresholds_padded = np.append(pr_thresholds, np.nan)
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": pr_thresholds_padded,
        }
    )
    pr_path = out_dir / "pr_curve_points.csv"
    pr_df.to_csv(pr_path, index=False)
    written["pr_curve_points"] = str(pr_path)

    return written


def write_optional_plots(
    out_dir: Path, y_true: np.ndarray, scores: np.ndarray
) -> dict[str, str | None]:
    plot_paths: dict[str, str | None] = {"roc_curve_png": None, "pr_curve_png": None}

    if len(np.unique(y_true)) < 2:
        return plot_paths

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Day 45 ROC Curve")
        plt.tight_layout()
        roc_png = out_dir / "roc_curve.png"
        plt.savefig(roc_png, dpi=160)
        plt.close()
        plot_paths["roc_curve_png"] = str(roc_png)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Day 45 Precision-Recall Curve")
        plt.tight_layout()
        pr_png = out_dir / "pr_curve.png"
        plt.savefig(pr_png, dpi=160)
        plt.close()
        plot_paths["pr_curve_png"] = str(pr_png)
    except Exception as exc:  # pragma: no cover
        plot_paths["plot_error"] = str(exc)

    return plot_paths


def write_markdown_report(
    out_dir: Path,
    input_path: Path,
    summary: dict[str, Any],
    score_col: str,
    label_source: str,
    type_col: str | None,
) -> None:
    metrics = summary["global_metrics"]
    threshold = summary["selected_threshold"]

    lines = [
        "# Day 45 — Comprehensive Test Set Evaluation",
        "",
        "## Status",
        "",
        "Complete.",
        "",
        "## Goal",
        "",
        "Evaluate the final calibrated anomaly score on a held-out/test score artifact, inspect threshold sensitivity, and select a conservative operating threshold with controlled false-positive rate.",
        "",
        "## Input",
        "",
        f"- Input scores: `{input_path}`",
        f"- Score column: `{score_col}`",
        f"- Label source: `{label_source}`",
        f"- Type column: `{type_col or 'not available'}`",
        "",
        "## Global metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Records | {summary['n_records']} |",
        f"| Positives | {summary['n_positive']} |",
        f"| Negatives | {summary['n_negative']} |",
        f"| ROC-AUC | {metrics.get('roc_auc')} |",
        f"| Average Precision | {metrics.get('average_precision')} |",
        "",
        "## Selected operating threshold",
        "",
        f"- Selection strategy: `{summary['threshold_selection_strategy']}`",
        f"- Threshold: `{threshold['threshold']:.6f}`",
        f"- Precision: `{threshold['precision']:.6f}`",
        f"- Recall: `{threshold['recall']:.6f}`",
        f"- F1: `{threshold['f1']:.6f}`",
        f"- False positive rate: `{threshold['false_positive_rate']:.6f}`",
        f"- False negative rate: `{threshold['false_negative_rate']:.6f}`",
        f"- Predicted positive rate: `{threshold['predicted_positive_rate']:.6f}`",
        "",
        "## Error-analysis artifacts",
        "",
        "- `threshold_sensitivity.csv`",
        "- `roc_curve_points.csv`",
        "- `pr_curve_points.csv`",
        "- `error_analysis_by_type.csv`",
        "- `false_positive_examples.csv`",
        "- `false_negative_examples.csv`",
        "- `flagged_records_preview.csv`",
        "",
        "## Paper-ready interpretation",
        "",
        "The selected threshold provides a conservative operating point for the ontology-calibrated anomaly score. This report should be used to support the evaluation section by documenting global discrimination, precision-recall trade-offs, false-positive behavior, and anomaly-type-specific failure modes.",
        "",
    ]

    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_scores:
        input_path = Path(args.input_scores)
        if not input_path.is_absolute():
            input_path = project_root / input_path
    else:
        candidates = find_candidate_files(project_root)
        if not candidates:
            raise FileNotFoundError(
                "No candidate score CSV files were found under artifacts/. Use --input_scores."
            )
        input_path = candidates[0]

    if not input_path.exists():
        raise FileNotFoundError(f"Input score file not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input score file is empty: {input_path}")

    score_col = pick_column(df, args.score_col, SCORE_CANDIDATES, "score")
    type_col = maybe_pick_column(df, args.type_col, TYPE_CANDIDATES)
    label_col = maybe_pick_column(df, args.label_col, LABEL_CANDIDATES)
    id_col = maybe_pick_column(df, args.id_col, ID_CANDIDATES)

    y_series, label_source = infer_label(df, label_col, type_col)
    scores_series = pd.to_numeric(df[score_col], errors="coerce")

    valid = scores_series.notna() & y_series.notna()
    df = df.loc[valid].copy()
    y_true = y_series.loc[valid].astype(int).to_numpy()
    scores = scores_series.loc[valid].astype(float).to_numpy()

    if len(df) == 0:
        raise ValueError("No valid rows remained after dropping missing labels/scores.")

    if len(np.unique(y_true)) >= 2:
        roc_auc = float(roc_auc_score(y_true, scores))
        avg_precision = float(average_precision_score(y_true, scores))
    else:
        roc_auc = None
        avg_precision = None

    sweep = build_threshold_sweep(y_true, scores)
    selected, selection_strategy = choose_threshold(
        sweep=sweep,
        max_fpr=args.max_fpr,
        min_precision=args.min_precision,
    )
    threshold = float(selected["threshold"])
    y_pred = (scores >= threshold).astype(int)

    sweep_path = out_dir / "threshold_sensitivity.csv"
    sweep.to_csv(sweep_path, index=False)

    by_type = build_by_type_analysis(df, type_col, score_col, y_true, y_pred)
    by_type_path = out_dir / "error_analysis_by_type.csv"
    by_type.to_csv(by_type_path, index=False)

    eval_df = df.copy()
    eval_df["_score"] = scores
    eval_df["_y_true"] = y_true
    eval_df["_y_pred"] = y_pred
    eval_df["_false_positive"] = (
        (eval_df["_y_true"] == 0) & (eval_df["_y_pred"] == 1)
    ).astype(int)
    eval_df["_false_negative"] = (
        (eval_df["_y_true"] == 1) & (eval_df["_y_pred"] == 0)
    ).astype(int)

    sort_cols = ["_score"]
    fp_df = (
        eval_df[eval_df["_false_positive"] == 1]
        .sort_values(sort_cols, ascending=False)
        .head(args.preview_rows)
    )
    fn_df = (
        eval_df[eval_df["_false_negative"] == 1]
        .sort_values(sort_cols, ascending=True)
        .head(args.preview_rows)
    )
    flagged_df = (
        eval_df[eval_df["_y_pred"] == 1]
        .sort_values(sort_cols, ascending=False)
        .head(args.preview_rows)
    )

    fp_path = out_dir / "false_positive_examples.csv"
    fn_path = out_dir / "false_negative_examples.csv"
    flagged_path = out_dir / "flagged_records_preview.csv"

    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    flagged_df.to_csv(flagged_path, index=False)

    if args.save_full_predictions:
        eval_df.to_csv(out_dir / "scored_records_with_predictions.csv", index=False)

    curve_files = write_curve_points(out_dir, y_true, scores)
    plot_files = write_optional_plots(out_dir, y_true, scores)

    selected_metrics = threshold_metrics(y_true, scores, threshold)

    hardest_type = None
    if not by_type.empty and "fn" in by_type.columns:
        positives_only = by_type[by_type["positives"] > 0].copy()
        if len(positives_only):
            hardest_type = str(
                positives_only.sort_values(
                    ["type_recall", "fn"], ascending=[True, False]
                ).iloc[0][type_col]
            )

    summary = {
        "day": 45,
        "status": "complete",
        "task": "Comprehensive Test Set Evaluation",
        "input_scores": str(input_path),
        "output_dir": str(out_dir),
        "n_records": int(len(df)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "score_column": score_col,
        "label_source": label_source,
        "type_column": type_col,
        "id_column": id_col,
        "global_metrics": {
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
        },
        "threshold_constraints": {
            "max_fpr": float(args.max_fpr),
            "min_precision": float(args.min_precision),
        },
        "threshold_selection_strategy": selection_strategy,
        "selected_threshold": selected_metrics,
        "hardest_anomaly_type_by_recall": hardest_type,
        "artifacts": {
            "threshold_sensitivity": str(sweep_path),
            "error_analysis_by_type": str(by_type_path),
            "false_positive_examples": str(fp_path),
            "false_negative_examples": str(fn_path),
            "flagged_records_preview": str(flagged_path),
            **curve_files,
            **plot_files,
        },
        "paper_note": (
            "Day 45 evaluates the calibrated anomaly score across the held-out/test score artifact, "
            "selecting a conservative threshold using false-positive and precision constraints."
        ),
    }

    summary_path = out_dir / "day45_test_set_metrics.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_markdown_report(
        out_dir=out_dir,
        input_path=input_path,
        summary=_json_safe(summary),
        score_col=score_col,
        label_source=label_source,
        type_col=type_col,
    )

    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 45 comprehensive test-set evaluation."
    )
    parser.add_argument("--project_root", default=".", help="Project root directory.")
    parser.add_argument(
        "--input_scores", default=None, help="CSV containing final scores and labels."
    )
    parser.add_argument(
        "--out_dir", default="artifacts/day45", help="Output directory."
    )
    parser.add_argument(
        "--score_col", default=None, help="Score column. Auto-detected if omitted."
    )
    parser.add_argument(
        "--label_col",
        default=None,
        help="Binary label column. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--type_col",
        default=None,
        help="Anomaly type column. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--id_col", default=None, help="Record id column. Auto-detected if omitted."
    )
    parser.add_argument(
        "--max_fpr",
        type=float,
        default=0.10,
        help="Maximum acceptable false-positive rate.",
    )
    parser.add_argument(
        "--min_precision",
        type=float,
        default=0.80,
        help="Minimum acceptable precision.",
    )
    parser.add_argument(
        "--preview_rows",
        type=int,
        default=50,
        help="Rows to keep for FP/FN/flagged previews.",
    )
    parser.add_argument(
        "--save_full_predictions",
        action="store_true",
        help="Write full per-record predictions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
