from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


LABEL_CANDIDATES = [
    "label",
    "y_true",
    "target",
    "is_anomaly",
    "anomaly_label",
    "ground_truth",
    "true_label",
]

EXCLUDE_COLUMNS = {
    "label",
    "y_true",
    "target",
    "is_anomaly",
    "anomaly_label",
    "ground_truth",
    "true_label",
    "row_id",
    "record_id",
    "source_row_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "icustay_id",
    "index",
}

LOWER_IS_BETTER_HINTS = {
    "rank",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def infer_label(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        y = pd.to_numeric(s, errors="coerce").fillna(0)
        return (y > 0).astype(int)

    text = s.astype(str).str.strip().str.lower()
    positives = {"1", "true", "anomaly", "anomalous", "positive", "abnormal", "synthetic_anomaly"}
    negatives = {"0", "false", "normal", "negative", "benign", "control", "non_anomaly", "non-anomaly"}

    out = []
    for item in text:
        if item in positives or ("anomaly" in item and item not in negatives):
            out.append(1)
        else:
            out.append(0)
    return pd.Series(out, index=s.index, dtype=int)


def is_numeric_score_column(df: pd.DataFrame, col: str, label_col: str) -> bool:
    col_l = col.lower()
    if col == label_col or col_l in EXCLUDE_COLUMNS:
        return False
    if col_l.endswith("_id") or col_l.endswith("id"):
        return False
    if any(hint in col_l for hint in LOWER_IS_BETTER_HINTS):
        return False

    values = pd.to_numeric(df[col], errors="coerce")
    if values.notna().mean() < 0.95:
        return False
    if values.nunique(dropna=True) < 2:
        return False
    return True


def threshold_at_positive_rate(scores: pd.Series, y: pd.Series) -> float:
    pos_rate = float(y.mean())
    pos_rate = min(max(pos_rate, 1e-6), 1 - 1e-6)
    return float(np.quantile(scores.to_numpy(dtype=float), 1.0 - pos_rate))


def evaluate_variant(name: str, y: pd.Series, raw_scores: pd.Series) -> dict[str, Any]:
    scores = pd.to_numeric(raw_scores, errors="coerce")
    scores = scores.fillna(scores.median()).astype(float)

    if y.nunique() < 2:
        raise ValueError("Need both normal and anomaly labels for evaluation.")

    threshold = threshold_at_positive_rate(scores, y)
    pred = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )

    return {
        "variant": name,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def markdown_table(df: pd.DataFrame) -> str:
    cols = ["variant", "roc_auc", "average_precision", "precision", "recall", "f1"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def latex_table(df: pd.DataFrame) -> str:
    cols = ["variant", "roc_auc", "average_precision", "precision", "recall", "f1"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Day 41 ablation results using Day 40 variant scores.}",
        "\\label{tab:day41_ablation_day40}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Variant & ROC-AUC & AP & Precision & Recall & F1 \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{str(row['variant']).replace('_', '\\_')} & "
            f"{row['roc_auc']:.4f} & "
            f"{row['average_precision']:.4f} & "
            f"{row['precision']:.4f} & "
            f"{row['recall']:.4f} & "
            f"{row['f1']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_scores", default="artifacts/day40/variant_scores.csv")
    parser.add_argument("--out_dir", default="artifacts/day41")
    args = parser.parse_args()

    input_path = Path(args.input_scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    label_col = first_existing(list(df.columns), LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    y = infer_label(df[label_col])

    if y.nunique() < 2:
        raise ValueError(
            f"Input label column {label_col!r} has only one class. "
            "Use the full Day 40 variant_scores.csv, not a sequential head sample."
        )

    score_cols = [
        col for col in df.columns
        if is_numeric_score_column(df, col, label_col)
    ]

    if not score_cols:
        raise ValueError(
            "No numeric variant score columns detected. "
            f"Columns were: {list(df.columns)}"
        )

    results = [evaluate_variant(col, y, df[col]) for col in score_cols]
    result_df = pd.DataFrame(results).sort_values(
        ["roc_auc", "average_precision", "f1"],
        ascending=False,
    )

    result_csv = out_dir / "day41_ablation_results.csv"
    result_json = out_dir / "day41_ablation_summary.json"
    result_md = out_dir / "day41_ablation_results.md"
    result_tex = out_dir / "day41_ablation_tables.tex"

    result_df.to_csv(result_csv, index=False)

    best = result_df.iloc[0].to_dict()

    full_auc = None
    no_ont_auc = None
    no_gen_auc = None

    variant_lower = {str(v).lower(): i for i, v in enumerate(result_df["variant"])}

    for key in ["full_model_conservative", "full_model", "full", "calibrated", "full_score"]:
        if key in variant_lower:
            full_auc = float(result_df.iloc[variant_lower[key]]["roc_auc"])
            break

    for key in ["no_ontology", "without_ontology"]:
        if key in variant_lower:
            no_ont_auc = float(result_df.iloc[variant_lower[key]]["roc_auc"])
            break

    for key in ["no_generative", "without_generative", "no_sgen"]:
        if key in variant_lower:
            no_gen_auc = float(result_df.iloc[variant_lower[key]]["roc_auc"])
            break

    interpretation = []

    if full_auc is not None and no_ont_auc is not None:
        interpretation.append(f"Full model minus no-ontology ROC-AUC delta: {full_auc - no_ont_auc:.4f}.")
    if full_auc is not None and no_gen_auc is not None:
        interpretation.append(f"Full model minus no-generative ROC-AUC delta: {full_auc - no_gen_auc:.4f}.")
    if full_auc is None:
        interpretation.append("No explicit full_model-like column was detected; use the best available Day 40 variant conservatively.")
    elif str(best["variant"]).lower() in {"full_model_conservative", "full_model", "full", "calibrated", "full_score"}:
        interpretation.append("The full ontology-calibrated variant is the top-ranked setting by ROC-AUC.")
    else:
        interpretation.append("The full ontology-calibrated variant is not the top-ranked setting by ROC-AUC; report this as an honest ablation finding.")

    summary = {
        "generated_at": now_utc(),
        "input_scores": str(input_path),
        "rows": int(len(df)),
        "label_column": label_col,
        "score_columns": score_cols,
        "best_variant": best["variant"],
        "best_roc_auc": best["roc_auc"],
        "best_average_precision": best["average_precision"],
        "interpretation": interpretation,
        "day34_policy_note": "Raw diffusion Sgen should be reported conservatively because Day 34 found weak standalone Sgen separation.",
        "results": result_df.to_dict(orient="records"),
    }

    result_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# Day 41 — Ablation Study Results",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Input scores: `{summary['input_scores']}`",
        f"- Rows evaluated: **{summary['rows']}**",
        f"- Label column: `{label_col}`",
        f"- Evaluated variants: **{len(score_cols)}**",
        f"- Best variant: **{summary['best_variant']}**",
        f"- Best ROC-AUC: **{summary['best_roc_auc']:.4f}**",
        f"- Best Average Precision: **{summary['best_average_precision']:.4f}**",
        "",
        "## Main Comparative Table",
        "",
        markdown_table(result_df),
        "",
        "## Interpretation",
        "",
    ]
    md.extend([f"- {item}" for item in interpretation])
    md.extend([
        "",
        "## Scientific Reporting Note",
        "",
        "This Day 41 run uses the Day 40 ablation score artifact directly. Raw diffusion `Sgen` should be interpreted conservatively because the earlier Day 34 evaluation found weak standalone generative separation.",
        "",
        "## Generated Artifacts",
        "",
        "- `artifacts/day41/day41_ablation_results.csv`",
        "- `artifacts/day41/day41_ablation_summary.json`",
        "- `artifacts/day41/day41_ablation_tables.tex`",
    ])

    result_md.write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    result_tex.write_text(latex_table(result_df), encoding="utf-8")

    print("Day 41 Day40-variant ablation complete.")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    run()

