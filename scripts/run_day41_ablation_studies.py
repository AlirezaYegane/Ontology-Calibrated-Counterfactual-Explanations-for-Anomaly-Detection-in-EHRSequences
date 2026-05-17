from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for Day 41 metrics. "
        "Install it with: python -m pip install scikit-learn"
    ) from exc


Y_COLS = [
    "y_true",
    "label",
    "target",
    "is_anomaly",
    "anomaly_label",
    "ground_truth",
    "true_label",
]

VARIANT_COLS = ["variant", "ablation", "model_variant", "setting"]

SCORE_COLS = [
    "score",
    "scal",
    "s_cal",
    "calibrated_score",
    "anomaly_score",
    "combined_score",
]

SDET_COLS = [
    "sdet",
    "Sdet",
    "s_det",
    "detector_score",
    "supervised_score",
    "model_score",
    "classifier_score",
    "anomaly_score",
]

SONT_COLS = [
    "sont",
    "Sont",
    "s_ont",
    "ontology_score",
    "strict_sont",
    "curated_sont",
    "violation_score",
    "ontology_violation_score",
]

SGEN_COLS = [
    "sgen",
    "Sgen",
    "s_gen",
    "generative_score",
    "diffusion_score",
    "denoising_error",
    "generative_surprise",
]

VAE_COLS = [
    "vae_score",
    "svae",
    "s_vae",
    "vae_reconstruction_error",
    "vae_surprise",
]

TYPE_COLS = [
    "anomaly_type",
    "type",
    "injection_type",
    "synthetic_type",
    "case_type",
]

RUNTIME_COLS = [
    "runtime_sec",
    "runtime_seconds",
    "latency_sec",
    "elapsed_sec",
    "time_sec",
    "runtime_ms",
    "latency_ms",
]

BEFORE_COLS = [
    "score_before",
    "scal_before",
    "s_cal_before",
    "before_score",
    "original_score",
]

AFTER_COLS = [
    "score_after",
    "scal_after",
    "s_cal_after",
    "after_score",
    "counterfactual_score",
]

EDIT_COLS = [
    "edit_count",
    "num_edits",
    "n_edits",
    "edits",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    direct = [c for c in candidates if c in df.columns]
    if direct:
        return direct[0]

    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(c.lower())
        if hit is not None:
            return str(hit)
    return None


def as_float_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        raise ValueError(f"Column {s.name!r} cannot be converted to numeric scores.")
    return out.fillna(out.median())


def robust_minmax(s: pd.Series) -> pd.Series:
    x = as_float_series(s).astype(float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - lo) / (hi - lo)


def infer_y_true(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        y = pd.to_numeric(series, errors="coerce").fillna(0)
        unique = set(y.astype(int).unique().tolist())
        if unique.issubset({0, 1}):
            return y.astype(int)
        return (y > 0).astype(int)

    text = series.astype(str).str.strip().str.lower()

    negative = {
        "0",
        "false",
        "normal",
        "negative",
        "non_anomaly",
        "non-anomaly",
        "benign",
        "control",
    }
    positive = {
        "1",
        "true",
        "anomaly",
        "anomalous",
        "positive",
        "synthetic_anomaly",
        "abnormal",
    }

    out = []
    for item in text:
        if item in negative or item.startswith("normal"):
            out.append(0)
        elif item in positive or "anomaly" in item or item not in negative:
            out.append(1)
        else:
            out.append(0)

    y = pd.Series(out, index=series.index, dtype=int)
    if y.nunique() < 2:
        raise ValueError(
            f"Could not infer both normal and anomaly labels from column {series.name!r}."
        )
    return y


def fair_threshold(scores: pd.Series, y_true: pd.Series) -> float:
    positive_rate = float(y_true.mean())
    positive_rate = min(max(positive_rate, 1e-6), 1 - 1e-6)
    return float(np.quantile(scores.to_numpy(dtype=float), 1.0 - positive_rate))


def evaluate_variant(
    *,
    variant: str,
    y_true: pd.Series,
    scores: pd.Series,
    anomaly_type: pd.Series | None = None,
    runtime: pd.Series | None = None,
    score_before: pd.Series | None = None,
    score_after: pd.Series | None = None,
    edit_count: pd.Series | None = None,
) -> dict[str, Any]:
    score = as_float_series(scores).astype(float)
    y = y_true.astype(int)

    if y.nunique() < 2:
        raise ValueError("Evaluation requires both normal and anomaly rows.")

    threshold = fair_threshold(score, y)
    pred = (score >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )

    result: dict[str, Any] = {
        "variant": variant,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if runtime is not None:
        rt = as_float_series(runtime).astype(float)
        if "ms" in str(runtime.name).lower():
            rt = rt / 1000.0
        result["mean_runtime_sec"] = float(rt.mean())
        result["median_runtime_sec"] = float(rt.median())

    if score_before is not None and score_after is not None:
        before = as_float_series(score_before).astype(float)
        after = as_float_series(score_after).astype(float)
        delta = before - after
        result["mean_delta_score"] = float(delta.mean())
        result["median_delta_score"] = float(delta.median())
        result["pct_improved"] = float((delta > 0).mean())

    if edit_count is not None:
        edits = pd.to_numeric(edit_count, errors="coerce").fillna(0)
        result["mean_edit_count"] = float(edits.mean())
        result["pct_one_edit_or_less"] = float((edits <= 1).mean())
        result["pct_two_edits_or_less"] = float((edits <= 2).mean())

    if anomaly_type is not None:
        breakdown: dict[str, dict[str, float]] = {}
        for t, idx in anomaly_type.groupby(anomaly_type).groups.items():
            idx_list = list(idx)
            if len(idx_list) == 0:
                continue
            breakdown[str(t)] = {
                "n": float(len(idx_list)),
                "mean_score": float(score.loc[idx_list].mean()),
                "median_score": float(score.loc[idx_list].median()),
            }
        result["type_breakdown"] = breakdown

    return result


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            value = row[c]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def latex_table(df: pd.DataFrame) -> str:
    cols = [
        "variant",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "mean_runtime_sec",
    ]
    cols = [c for c in cols if c in df.columns]
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Day 41 ablation results.}")
    lines.append("\\label{tab:day41_ablation}")
    lines.append("\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}")
    lines.append("\\toprule")
    header = []
    for c in cols:
        header.append(c.replace("_", "\\_"))
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def discover_csvs(root: Path) -> list[Path]:
    search_roots = [
        root / "artifacts" / "day40",
        root / "artifacts" / "day39",
        root / "artifacts" / "day38",
        root / "artifacts" / "day37",
        root / "artifacts" / "day36",
        root / "artifacts" / "day36_repair_ready",
        root / "artifacts" / "day35_7",
        root / "artifacts" / "day35_6",
        root / "artifacts" / "day35_5",
        root / "artifacts" / "day35",
    ]
    found: list[Path] = []
    for d in search_roots:
        if d.exists():
            found.extend(sorted(d.rglob("*.csv")))
    return found


def score_candidate_file(path: Path) -> tuple[int, str]:
    try:
        df = pd.read_csv(path, nrows=50)
    except Exception:
        return (0, "unreadable")

    y_col = first_col(df, Y_COLS)
    variant_col = first_col(df, VARIANT_COLS)
    score_col = first_col(df, SCORE_COLS)
    sdet_col = first_col(df, SDET_COLS)
    sont_col = first_col(df, SONT_COLS)
    sgen_col = first_col(df, SGEN_COLS)

    score = 0
    reasons = []
    if y_col:
        score += 5
        reasons.append(f"label={y_col}")
    if variant_col and score_col:
        score += 8
        reasons.append(f"long_format={variant_col}+{score_col}")
    for name, col in [("sdet", sdet_col), ("sont", sont_col), ("sgen", sgen_col)]:
        if col:
            score += 3
            reasons.append(f"{name}={col}")

    return score, ", ".join(reasons) if reasons else "no recognized schema"


def choose_input(root: Path, provided: str | None) -> Path:
    if provided:
        path = Path(provided)
        if not path.exists():
            raise FileNotFoundError(f"Input score file does not exist: {path}")
        return path

    candidates = []
    for p in discover_csvs(root):
        score, reason = score_candidate_file(p)
        if score > 0:
            candidates.append((score, reason, p))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    if not candidates:
        raise FileNotFoundError(
            "No suitable CSV score file found automatically. "
            "Pass --input_scores manually, preferably a Day 40 row-level ablation CSV."
        )

    print("Auto-selected input:", candidates[0][2])
    print("Reason:", candidates[0][1])
    print()
    print("Top candidate files:")
    for score, reason, path in candidates[:8]:
        print(f"- score={score:02d} | {path} | {reason}")

    return candidates[0][2]


def build_component_variants(df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, str]]:
    sdet_col = first_col(df, SDET_COLS)
    sont_col = first_col(df, SONT_COLS)
    sgen_col = first_col(df, SGEN_COLS)
    vae_col = first_col(df, VAE_COLS)
    full_existing_col = first_col(df, ["full_model_score", "full_score", "scal", "s_cal", "calibrated_score"])

    components: dict[str, pd.Series] = {}
    used: dict[str, str] = {}

    if sdet_col:
        components["sdet"] = robust_minmax(df[sdet_col])
        used["sdet"] = sdet_col
    if sont_col:
        components["sont"] = robust_minmax(df[sont_col])
        used["sont"] = sont_col
    if sgen_col:
        components["sgen"] = robust_minmax(df[sgen_col])
        used["sgen"] = sgen_col
    if vae_col:
        components["vae"] = robust_minmax(df[vae_col])
        used["vae"] = vae_col

    variants: dict[str, pd.Series] = {}

    if full_existing_col:
        variants["existing_full_score"] = robust_minmax(df[full_existing_col])
        used["existing_full_score"] = full_existing_col

    if "sdet" in components:
        variants["detector_only"] = components["sdet"]

    if "sont" in components:
        variants["ontology_only"] = components["sont"]

    if "sgen" in components:
        variants["generative_only"] = components["sgen"]

    if "sdet" in components and "sgen" in components:
        variants["no_ontology"] = 0.90 * components["sdet"] + 0.10 * components["sgen"]

    if "sdet" in components and "sont" in components:
        variants["no_generative"] = 0.55 * components["sdet"] + 0.45 * components["sont"]

    if all(k in components for k in ["sdet", "sont", "sgen"]):
        variants["full_model"] = (
            0.55 * components["sdet"]
            + 0.40 * components["sont"]
            + 0.05 * components["sgen"]
        )

    if all(k in components for k in ["sdet", "sont", "vae"]):
        variants["vae_replacement"] = (
            0.55 * components["sdet"]
            + 0.40 * components["sont"]
            + 0.05 * components["vae"]
        )

    if not variants:
        raise ValueError(
            "Could not build ablation variants. Need either long-format "
            "`variant + score` columns or component columns such as Sdet/Sont/Sgen."
        )

    return variants, used


def run(args: argparse.Namespace) -> None:
    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = choose_input(root, args.input_scores)
    df = pd.read_csv(input_path)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    y_col = first_col(df, Y_COLS)
    if y_col is None:
        raise ValueError(f"No label column found. Expected one of: {Y_COLS}")

    y_true = infer_y_true(df[y_col])

    anomaly_type_col = first_col(df, TYPE_COLS)
    runtime_col = first_col(df, RUNTIME_COLS)
    before_col = first_col(df, BEFORE_COLS)
    after_col = first_col(df, AFTER_COLS)
    edit_col = first_col(df, EDIT_COLS)

    anomaly_type = df[anomaly_type_col] if anomaly_type_col else None
    runtime = df[runtime_col] if runtime_col else None
    before = df[before_col] if before_col else None
    after = df[after_col] if after_col else None
    edits = df[edit_col] if edit_col else None

    results: list[dict[str, Any]] = []
    score_frame = pd.DataFrame({"y_true": y_true})

    variant_col = first_col(df, VARIANT_COLS)
    long_score_col = first_col(df, SCORE_COLS)
    used_columns: dict[str, str] = {"label": y_col}

    if variant_col and long_score_col:
        used_columns["variant"] = variant_col
        used_columns["score"] = long_score_col

        for variant_name, group in df.groupby(variant_col):
            group_y = infer_y_true(group[y_col])
            group_type = group[anomaly_type_col] if anomaly_type_col else None
            group_runtime = group[runtime_col] if runtime_col else None
            group_before = group[before_col] if before_col else None
            group_after = group[after_col] if after_col else None
            group_edits = group[edit_col] if edit_col else None

            results.append(
                evaluate_variant(
                    variant=str(variant_name),
                    y_true=group_y,
                    scores=group[long_score_col],
                    anomaly_type=group_type,
                    runtime=group_runtime,
                    score_before=group_before,
                    score_after=group_after,
                    edit_count=group_edits,
                )
            )
    else:
        variants, used = build_component_variants(df)
        used_columns.update(used)
        for variant_name, scores in variants.items():
            score_frame[variant_name] = scores
            results.append(
                evaluate_variant(
                    variant=variant_name,
                    y_true=y_true,
                    scores=scores,
                    anomaly_type=anomaly_type,
                    runtime=runtime,
                    score_before=before,
                    score_after=after,
                    edit_count=edits,
                )
            )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=["roc_auc", "average_precision", "f1"],
        ascending=False,
    ).reset_index(drop=True)

    result_csv = out_dir / "day41_ablation_results.csv"
    result_json = out_dir / "day41_ablation_summary.json"
    result_md = out_dir / "day41_ablation_results.md"
    result_tex = out_dir / "day41_ablation_tables.tex"
    score_csv = out_dir / "day41_variant_scores.csv"

    result_df.drop(columns=["type_breakdown"], errors="ignore").to_csv(result_csv, index=False)
    score_frame.to_csv(score_csv, index=False)

    best = result_df.iloc[0].to_dict()
    summary: dict[str, Any] = {
        "generated_at": now_utc(),
        "input_scores": str(input_path),
        "rows": int(len(df)),
        "label_column": y_col,
        "used_columns": used_columns,
        "best_variant": best.get("variant"),
        "best_roc_auc": best.get("roc_auc"),
        "best_average_precision": best.get("average_precision"),
        "variants": result_df["variant"].tolist(),
        "day34_policy_note": (
            "Sgen is treated as a low-weight/diagnostic component because prior "
            "checkpoint-aligned Day 34 evaluation found raw diffusion Sgen weak."
        ),
        "results": result_df.to_dict(orient="records"),
    }

    full_row = result_df[result_df["variant"].astype(str).eq("full_model")]
    no_ont_row = result_df[result_df["variant"].astype(str).eq("no_ontology")]
    no_gen_row = result_df[result_df["variant"].astype(str).eq("no_generative")]

    interpretation = []
    if not full_row.empty and not no_ont_row.empty:
        delta = float(full_row.iloc[0]["roc_auc"] - no_ont_row.iloc[0]["roc_auc"])
        summary["delta_full_minus_no_ontology_auc"] = delta
        interpretation.append(
            f"Full model vs no-ontology ROC-AUC delta: {delta:.4f}."
        )
    if not full_row.empty and not no_gen_row.empty:
        delta = float(full_row.iloc[0]["roc_auc"] - no_gen_row.iloc[0]["roc_auc"])
        summary["delta_full_minus_no_generative_auc"] = delta
        interpretation.append(
            f"Full model vs no-generative ROC-AUC delta: {delta:.4f}."
        )

    if full_row.empty:
        interpretation.append(
            "No explicit `full_model` variant was available; interpret the best available setting conservatively."
        )
    elif str(best.get("variant")) == "full_model":
        interpretation.append(
            "The full ontology-calibrated setting ranked highest under ROC-AUC in this run."
        )
    else:
        interpretation.append(
            "The full setting did not rank first in this run; report this as a mixed or negative ablation finding, not as a failure."
        )

    summary["interpretation"] = interpretation

    result_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Day 41 — Ablation Study Results",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Input scores: `{summary['input_scores']}`",
        f"- Rows evaluated: **{summary['rows']}**",
        f"- Best variant: **{summary['best_variant']}**",
        f"- Best ROC-AUC: **{summary['best_roc_auc']:.4f}**",
        f"- Best Average Precision: **{summary['best_average_precision']:.4f}**",
        "",
        "## Main Comparative Table",
        "",
        markdown_table(
            result_df.drop(columns=["type_breakdown"], errors="ignore"),
            [
                "variant",
                "roc_auc",
                "average_precision",
                "precision",
                "recall",
                "f1",
                "mean_delta_score",
                "pct_improved",
                "mean_edit_count",
                "mean_runtime_sec",
            ],
        ),
        "",
        "## Interpretation",
        "",
    ]
    md_lines.extend([f"- {item}" for item in interpretation])
    md_lines.extend(
        [
            "",
            "## Scientific Reporting Note",
            "",
            (
                "Because the earlier checkpoint-aligned generative evaluation showed weak raw "
                "`Sgen` separation, this ablation treats `Sgen` as a low-weight auxiliary "
                "component rather than the dominant anomaly signal."
            ),
            "",
            "## Generated Artifacts",
            "",
            "- `artifacts/day41/day41_ablation_results.csv`",
            "- `artifacts/day41/day41_ablation_summary.json`",
            "- `artifacts/day41/day41_ablation_tables.tex`",
            "- `artifacts/day41/day41_variant_scores.csv`",
        ]
    )
    result_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    result_tex.write_text(latex_table(result_df), encoding="utf-8")

    print()
    print("Day 41 ablation study complete.")
    print(f"- {result_csv}")
    print(f"- {result_json}")
    print(f"- {result_md}")
    print(f"- {result_tex}")
    print(f"- {score_csv}")
    print()
    print(result_df.drop(columns=["type_breakdown"], errors="ignore").to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--input_scores", default=None)
    parser.add_argument("--out_dir", default="artifacts/day41")
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
