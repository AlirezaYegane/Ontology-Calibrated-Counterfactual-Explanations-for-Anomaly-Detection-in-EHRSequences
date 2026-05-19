from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


Y_CANDIDATES = ("y_true", "label", "target", "is_anomaly", "anomaly_label")
EXCLUDE_NUMERIC = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "icustay_id",
    "row_id",
    "index",
    "fold",
    "seed",
}


def infer_y_column(df: pd.DataFrame) -> str:
    for col in Y_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not infer label column. Expected one of {Y_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def infer_score_columns(df: pd.DataFrame, y_col: str) -> list[str]:
    score_cols: list[str] = []
    for col in df.columns:
        if col == y_col:
            continue
        if col.lower() in EXCLUDE_NUMERIC:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            score_cols.append(col)

    if not score_cols:
        raise ValueError(
            "No numeric variant score columns found. "
            "Expected columns such as detector_only, generative_only, no_ontology, full_model, etc."
        )
    return score_cols


def roc_auc_score_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute tie-aware average precision.

    This follows the precision-recall step formulation used for ranking scores.
    It handles tied/constant scores correctly. For a constant score vector, AP
    equals the positive-class prevalence instead of depending on row order.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    distinct_value_indices = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs].astype(float)
    fps = (1 + threshold_idxs - tps).astype(float)

    precision = tps / (tps + fps)
    recall = tps / n_pos

    previous_recall = np.r_[0.0, recall[:-1]]
    ap = np.sum((recall - previous_recall) * precision)
    return float(ap)


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    quantiles = np.linspace(0.0, 1.0, 201)
    thresholds = np.unique(np.quantile(y_score, quantiles))
    thresholds = np.unique(np.concatenate([thresholds, np.array([0.5])]))

    rows = [threshold_metrics(y_true, y_score, float(t)) for t in thresholds]
    rows = sorted(rows, key=lambda r: (r["f1"], r["balanced_accuracy"], r["precision"]), reverse=True)
    return rows[0]


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    if metric == "roc_auc":
        fn = roc_auc_score_safe
    elif metric == "average_precision":
        fn = average_precision_safe
    else:
        raise ValueError(metric)

    point = fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        v = fn(y_b, y_score[idx])
        if not np.isnan(v):
            vals.append(v)

    if not vals:
        return point, float("nan"), float("nan")

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def summarize_by_label(y_true: np.ndarray, df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = sorted(np.unique(y_true).tolist())

    for variant in score_cols:
        scores = df[variant].astype(float).to_numpy()
        for label in labels:
            s = scores[y_true == label]
            rows.append(
                {
                    "variant": variant,
                    "label": int(label),
                    "count": int(len(s)),
                    "mean": float(np.mean(s)),
                    "std": float(np.std(s)),
                    "median": float(np.median(s)),
                    "p10": float(np.quantile(s, 0.10)),
                    "p90": float(np.quantile(s, 0.90)),
                    "min": float(np.min(s)),
                    "max": float(np.max(s)),
                }
            )

    return pd.DataFrame(rows)


def make_metrics_table(
    df: pd.DataFrame,
    y_col: str,
    score_cols: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    y_true = df[y_col].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []

    for i, variant in enumerate(score_cols):
        y_score = df[variant].astype(float).fillna(0.0).to_numpy()

        roc, roc_lo, roc_hi = bootstrap_ci(
            y_true, y_score, metric="roc_auc", n_boot=n_boot, seed=seed + i
        )
        ap, ap_lo, ap_hi = bootstrap_ci(
            y_true, y_score, metric="average_precision", n_boot=n_boot, seed=seed + 1000 + i
        )
        th = best_threshold_by_f1(y_true, y_score)

        rows.append(
            {
                "variant": variant,
                "roc_auc": roc,
                "roc_auc_ci_low": roc_lo,
                "roc_auc_ci_high": roc_hi,
                "average_precision": ap,
                "average_precision_ci_low": ap_lo,
                "average_precision_ci_high": ap_hi,
                "best_threshold": th["threshold"],
                "precision": th["precision"],
                "recall": th["recall"],
                "specificity": th["specificity"],
                "f1": th["f1"],
                "balanced_accuracy": th["balanced_accuracy"],
                "predicted_positive_rate": th["predicted_positive_rate"],
                "score_mean": float(np.mean(y_score)),
                "score_std": float(np.std(y_score)),
                "score_min": float(np.min(y_score)),
                "score_max": float(np.max(y_score)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["roc_auc", "average_precision", "f1"], ascending=False).reset_index(drop=True)
    return out


def make_pairwise_deltas(metrics_df: pd.DataFrame) -> pd.DataFrame:
    best = metrics_df.iloc[0]
    rows: list[dict[str, Any]] = []

    for _, row in metrics_df.iterrows():
        rows.append(
            {
                "reference_variant": str(best["variant"]),
                "comparison_variant": str(row["variant"]),
                "delta_roc_auc_vs_reference": float(row["roc_auc"] - best["roc_auc"]),
                "delta_average_precision_vs_reference": float(row["average_precision"] - best["average_precision"]),
                "delta_f1_vs_reference": float(row["f1"] - best["f1"]),
                "delta_recall_vs_reference": float(row["recall"] - best["recall"]),
                "delta_precision_vs_reference": float(row["precision"] - best["precision"]),
            }
        )

    return pd.DataFrame(rows)


def fmt(x: Any) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def render_markdown_table(metrics_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Day 46 — Paper-Ready Ablation Metrics")
    lines.append("")
    lines.append("| Variant | ROC-AUC | 95% CI | AP | 95% CI | F1 | Precision | Recall | Threshold |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in metrics_df.iterrows():
        roc_ci = f"[{fmt(r['roc_auc_ci_low'])}, {fmt(r['roc_auc_ci_high'])}]"
        ap_ci = f"[{fmt(r['average_precision_ci_low'])}, {fmt(r['average_precision_ci_high'])}]"
        lines.append(
            f"| `{r['variant']}` | {fmt(r['roc_auc'])} | {roc_ci} | "
            f"{fmt(r['average_precision'])} | {ap_ci} | {fmt(r['f1'])} | "
            f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['best_threshold'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_interpretation(metrics_df: pd.DataFrame, deltas_df: pd.DataFrame) -> str:
    best = metrics_df.iloc[0]
    variant_names = set(metrics_df["variant"].astype(str))

    lines: list[str] = []
    lines.append("# Day 46 — Results Interpretation for Manuscript")
    lines.append("")
    lines.append("## Main result")
    lines.append("")
    lines.append(
        f"The strongest variant in this run is `{best['variant']}`, "
        f"with ROC-AUC={fmt(best['roc_auc'])}, AP={fmt(best['average_precision'])}, "
        f"F1={fmt(best['f1'])}, precision={fmt(best['precision'])}, and recall={fmt(best['recall'])}."
    )
    lines.append("")

    if "generative_only" in variant_names:
        gen = metrics_df[metrics_df["variant"] == "generative_only"].iloc[0]
        lines.append("## Generative-only signal")
        lines.append("")
        if float(gen["roc_auc"]) < 0.55:
            lines.append(
                "The `generative_only` variant remains weak as a standalone anomaly discriminator. "
                "This supports reporting the generative component conservatively as an auxiliary or diagnostic signal, "
                "rather than claiming that it independently drives anomaly detection performance."
            )
        else:
            lines.append(
                "The `generative_only` variant shows measurable discrimination, but it should still be interpreted "
                "relative to detector and ontology-calibrated variants before making a strong claim."
            )
        lines.append(
            f"In this run, `generative_only` achieved ROC-AUC={fmt(gen['roc_auc'])} and AP={fmt(gen['average_precision'])}."
        )
        lines.append("")

    if "detector_only" in variant_names:
        det = metrics_df[metrics_df["variant"] == "detector_only"].iloc[0]
        lines.append("## Detector-only baseline")
        lines.append("")
        lines.append(
            f"The `detector_only` baseline achieved ROC-AUC={fmt(det['roc_auc'])} and AP={fmt(det['average_precision'])}. "
            "This is the key statistical baseline for judging whether ontology calibration adds useful information."
        )
        lines.append("")

    if "no_ontology" in variant_names:
        no_ont = metrics_df[metrics_df["variant"] == "no_ontology"].iloc[0]
        lines.append("## Ontology ablation")
        lines.append("")
        lines.append(
            f"The `no_ontology` variant achieved ROC-AUC={fmt(no_ont['roc_auc'])} and AP={fmt(no_ont['average_precision'])}. "
            "This variant is important because it isolates how much performance remains when explicit ontology calibration is removed."
        )
        lines.append("")

    lines.append("## Paper-safe interpretation")
    lines.append("")
    lines.append(
        "A conservative manuscript claim should focus on comparative evidence: which scoring component contributes signal, "
        "which component fails as a standalone discriminator, and how ontology-aware calibration changes performance relative to ablated variants. "
        "Avoid overstating the diffusion/generative component if the ablation shows weak discriminative value."
    )
    lines.append("")
    lines.append("## Recommended manuscript wording")
    lines.append("")
    lines.append(
        "> The ablation study indicates that anomaly discrimination is primarily driven by the detector and ontology-calibrated scoring components, "
        "while the current generative-only score provides limited standalone discrimination. We therefore treat the generative score as an auxiliary "
        "diagnostic signal in the present implementation and report ontology/detector contributions transparently through ablation results."
    )
    lines.append("")

    return "\n".join(lines)


def write_plots(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for metric, ylabel, filename in [
        ("roc_auc", "ROC-AUC", "day46_roc_auc_by_variant.png"),
        ("average_precision", "Average Precision", "day46_average_precision_by_variant.png"),
        ("f1", "Best F1", "day46_f1_by_variant.png"),
    ]:
        fig = plt.figure(figsize=(9, 5))
        plt.bar(metrics_df["variant"].astype(str), metrics_df[metric].astype(float))
        plt.ylabel(ylabel)
        plt.xlabel("Variant")
        plt.title(f"Day 46 — {ylabel} by Ablation Variant")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--out_dir", default="artifacts/day46")
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scores_path = Path(args.scores_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_path)
    y_col = infer_y_column(df)
    score_cols = infer_score_columns(df, y_col)

    df = df.dropna(subset=[y_col]).copy()
    df[y_col] = df[y_col].astype(int)

    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    y_true = df[y_col].astype(int).to_numpy()

    metrics_df = make_metrics_table(df, y_col, score_cols, n_boot=args.n_boot, seed=args.seed)
    deltas_df = make_pairwise_deltas(metrics_df)
    dist_df = summarize_by_label(y_true, df, score_cols)

    metrics_df.to_csv(out_dir / "day46_variant_metrics.csv", index=False)
    deltas_df.to_csv(out_dir / "day46_pairwise_deltas.csv", index=False)
    dist_df.to_csv(out_dir / "day46_score_distribution_by_label.csv", index=False)

    (out_dir / "day46_paper_results_table.md").write_text(
        render_markdown_table(metrics_df),
        encoding="utf-8",
    )
    (out_dir / "day46_result_interpretation.md").write_text(
        render_interpretation(metrics_df, deltas_df),
        encoding="utf-8",
    )

    write_plots(metrics_df, out_dir)

    summary = {
        "day": 46,
        "title": "Paper-ready ablation evidence pack",
        "status": "complete",
        "input_scores_csv": str(scores_path),
        "rows": int(len(df)),
        "label_column": y_col,
        "variants": score_cols,
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "best_variant": str(metrics_df.iloc[0]["variant"]),
        "best_roc_auc": float(metrics_df.iloc[0]["roc_auc"]),
        "best_average_precision": float(metrics_df.iloc[0]["average_precision"]),
        "best_f1": float(metrics_df.iloc[0]["f1"]),
        "outputs": {
            "metrics": str(out_dir / "day46_variant_metrics.csv"),
            "deltas": str(out_dir / "day46_pairwise_deltas.csv"),
            "distribution": str(out_dir / "day46_score_distribution_by_label.csv"),
            "paper_table": str(out_dir / "day46_paper_results_table.md"),
            "interpretation": str(out_dir / "day46_result_interpretation.md"),
            "roc_plot": str(out_dir / "day46_roc_auc_by_variant.png"),
            "ap_plot": str(out_dir / "day46_average_precision_by_variant.png"),
            "f1_plot": str(out_dir / "day46_f1_by_variant.png"),
        },
    }

    (out_dir / "day46_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

