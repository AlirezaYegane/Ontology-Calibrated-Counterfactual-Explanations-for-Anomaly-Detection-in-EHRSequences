from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_CANDIDATES = [
    "label",
    "y_true",
    "target",
    "is_anomaly",
    "is_synthetic_anomaly",
    "synthetic_anomaly",
    "anomaly_label",
]

DETECTOR_SCORE_CANDIDATES = [
    "sdet",
    "s_det",
    "detector_score",
    "det_score",
    "prob_anomaly",
    "anomaly_probability",
    "supervised_score",
    "model_score",
    "anomaly_score",
]

ONTOLOGY_SCORE_CANDIDATES = [
    "sont",
    "s_ont",
    "ontology_score",
    "ontology_violation_score",
    "ontology_violations",
    "violation_count",
    "num_violations",
    "rule_score",
]

GENERATIVE_SCORE_CANDIDATES = [
    "sgen",
    "s_gen",
    "generative_score",
    "diffusion_score",
    "diffusion_sgen",
    "denoising_error",
    "reconstruction_error",
    "gen_score",
]

VAE_SCORE_CANDIDATES = [
    "vae_score",
    "vae_sgen",
    "vae_reconstruction_error",
    "vae_proxy_score",
]

CALIBRATED_SCORE_CANDIDATES = [
    "scal",
    "s_cal",
    "calibrated_score",
    "final_score",
    "combined_score",
]

PREFERRED_INPUT_HINTS = [
    "day39",
    "day38",
    "day37",
    "day36",
    "day35",
    "repair_ready",
    "case",
    "counterfactual",
    "score",
]


@dataclass
class ColumnResolution:
    label_col: str
    detector_col: str | None
    ontology_col: str | None
    generative_col: str | None
    vae_col: str | None
    calibrated_col: str | None
    label_source: str
    warnings: list[str]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalise_score(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.min(arr)) if len(arr) else 0.0
    hi = float(np.max(arr)) if len(arr) else 0.0

    if math.isclose(lo, hi):
        return np.zeros_like(arr, dtype=float)

    return (arr - lo) / (hi - lo)


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}

    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

    for cand in candidates:
        pattern = re.compile(re.escape(cand), re.IGNORECASE)
        for col in columns:
            if pattern.search(col):
                return col

    return None


def infer_label_from_frame(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    direct = find_column(cols, LABEL_CANDIDATES)
    if direct is not None:
        return direct, "direct_label_column"

    if "source" in df.columns:
        labels = []
        for value in df["source"].astype(str).str.lower():
            labels.append(0 if "normal" in value and "anomaly" not in value else 1)
        df["__label_inferred_from_source"] = labels
        return "__label_inferred_from_source", "inferred_from_source"

    if "anomaly_type" in df.columns:
        normal_values = {"", "normal", "none", "nan", "null", "no_anomaly"}
        labels = []
        for value in df["anomaly_type"].fillna("").astype(str).str.lower().str.strip():
            labels.append(0 if value in normal_values else 1)
        df["__label_inferred_from_anomaly_type"] = labels
        return "__label_inferred_from_anomaly_type", "inferred_from_anomaly_type"

    raise ValueError(
        "Could not infer a binary label column. Expected one of "
        f"{LABEL_CANDIDATES}, or a usable source/anomaly_type column."
    )


def resolve_columns(df: pd.DataFrame) -> ColumnResolution:
    warnings: list[str] = []
    cols = list(df.columns)

    label_col, label_source = infer_label_from_frame(df)

    detector_col = find_column(cols, DETECTOR_SCORE_CANDIDATES)
    ontology_col = find_column(cols, ONTOLOGY_SCORE_CANDIDATES)
    generative_col = find_column(cols, GENERATIVE_SCORE_CANDIDATES)
    vae_col = find_column(cols, VAE_SCORE_CANDIDATES)
    calibrated_col = find_column(cols, CALIBRATED_SCORE_CANDIDATES)

    if detector_col is None and calibrated_col is not None:
        detector_col = calibrated_col
        warnings.append(
            "No explicit detector score found; using calibrated/final score as detector-like fallback."
        )

    if detector_col is None:
        warnings.append("No detector score column was found.")

    if ontology_col is None:
        warnings.append(
            "No ontology score column was found; Sont will be zero for ontology variants."
        )

    if generative_col is None:
        warnings.append(
            "No generative/diffusion score column was found; Sgen will be zero for generative variants."
        )

    if vae_col is None:
        warnings.append(
            "No true VAE score column was found. VAE replacement can only run as a proxy slot."
        )

    return ColumnResolution(
        label_col=label_col,
        detector_col=detector_col,
        ontology_col=ontology_col,
        generative_col=generative_col,
        vae_col=vae_col,
        calibrated_col=calibrated_col,
        label_source=label_source,
        warnings=warnings,
    )


def read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=1).columns)


def score_input_candidate(path: Path) -> int:
    name = path.as_posix().lower()
    score = 0

    for hint in PREFERRED_INPUT_HINTS:
        if hint in name:
            score += 2

    try:
        cols = [c.lower() for c in read_csv_header(path)]
    except Exception:
        return -999

    if any(c in cols for c in LABEL_CANDIDATES):
        score += 8
    if "source" in cols or "anomaly_type" in cols:
        score += 5
    if any(c in cols for c in DETECTOR_SCORE_CANDIDATES):
        score += 6
    if any(c in cols for c in ONTOLOGY_SCORE_CANDIDATES):
        score += 6
    if any(c in cols for c in GENERATIVE_SCORE_CANDIDATES):
        score += 4
    if any(c in cols for c in CALIBRATED_SCORE_CANDIDATES):
        score += 3

    return score


def auto_discover_input(artifacts_dir: Path) -> Path:
    csvs = sorted(artifacts_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {artifacts_dir}")

    ranked = sorted(
        ((score_input_candidate(path), path) for path in csvs),
        key=lambda item: (item[0], item[1].as_posix()),
        reverse=True,
    )

    best_score, best_path = ranked[0]
    if best_score <= 0:
        raise FileNotFoundError(
            "CSV files were found, but none looked like a score/evaluation file. "
            "Pass --input_scores explicitly."
        )

    return best_path


def binary_labels(series: pd.Series) -> np.ndarray:
    out = []
    for value in series:
        if isinstance(value, str):
            text = value.strip().lower()
            out.append(0 if text in {"0", "false", "normal", "no", "negative"} else 1)
        else:
            out.append(int(float(value) > 0))
    return np.asarray(out, dtype=int)


def roc_auc_score_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s)
    y_sorted = y[order]

    tp = 0
    precisions = []
    for i, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)

    return float(np.mean(precisions)) if precisions else float("nan")


def threshold_grid(scores: np.ndarray, max_thresholds: int = 500) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=float))
    if len(unique) <= max_thresholds:
        return unique

    qs = np.linspace(0, 1, max_thresholds)
    return np.unique(np.quantile(unique, qs))


def best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_thresholds: int = 500,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "predicted_positive_rate": 0.0,
    }

    for thr in threshold_grid(s, max_thresholds=max_thresholds):
        pred = (s >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        ppr = float(np.mean(pred))

        if f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive_rate": ppr,
            }

    return best


def precision_at_fraction(
    y_true: np.ndarray, scores: np.ndarray, fraction: float
) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    if len(y) == 0:
        return float("nan")

    k = max(1, int(math.ceil(len(y) * fraction)))
    order = np.argsort(-s)[:k]
    return float(np.mean(y[order]))


def cohen_d(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")

    pooled = math.sqrt((float(np.var(pos, ddof=1)) + float(np.var(neg, ddof=1))) / 2)
    if math.isclose(pooled, 0.0):
        return float("nan")

    return float((float(np.mean(pos)) - float(np.mean(neg))) / pooled)


def make_vae_proxy(
    df: pd.DataFrame, detector: np.ndarray, ontology: np.ndarray
) -> np.ndarray:
    """Conservative non-final proxy used only to keep the VAE replacement slot runnable.

    This is not a trained VAE. It allows the Day 40 framework to produce a comparable
    row while clearly marking it as a placeholder until a true VAE baseline is trained.
    """
    length_cols = [
        "sequence_length",
        "seq_len",
        "num_tokens",
        "n_tokens",
        "length",
        "edit_count",
        "num_edits",
    ]

    for col in length_cols:
        if col in df.columns:
            raw = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            median = float(np.median(raw)) if len(raw) else 0.0
            deviation = np.abs(raw - median)
            return normalise_score(deviation)

    # Final fallback: a small rank-stable mixture that does not use labels.
    return normalise_score(0.7 * detector + 0.3 * ontology)


def build_component_scores(
    df: pd.DataFrame,
    cols: ColumnResolution,
    allow_vae_proxy: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_true = binary_labels(df[cols.label_col])

    detector = (
        normalise_score(pd.to_numeric(df[cols.detector_col], errors="coerce"))
        if cols.detector_col is not None
        else np.zeros(len(df), dtype=float)
    )
    ontology = (
        normalise_score(pd.to_numeric(df[cols.ontology_col], errors="coerce"))
        if cols.ontology_col is not None
        else np.zeros(len(df), dtype=float)
    )
    generative = (
        normalise_score(pd.to_numeric(df[cols.generative_col], errors="coerce"))
        if cols.generative_col is not None
        else np.zeros(len(df), dtype=float)
    )

    vae_is_true = cols.vae_col is not None
    if cols.vae_col is not None:
        vae = normalise_score(pd.to_numeric(df[cols.vae_col], errors="coerce"))
    elif allow_vae_proxy:
        vae = make_vae_proxy(df, detector, ontology)
    else:
        vae = np.zeros(len(df), dtype=float)

    component_df = pd.DataFrame(
        {
            "label": y_true,
            "Sdet_norm": detector,
            "Sont_norm": ontology,
            "Sgen_norm": generative,
            "Svae_norm": vae,
        }
    )

    for optional_col in [
        "source",
        "anomaly_type",
        "case_id",
        "record_id",
        "hadm_id",
        "subject_id",
    ]:
        if optional_col in df.columns:
            component_df[optional_col] = df[optional_col].values

    meta = {
        "label_column": cols.label_col,
        "label_source": cols.label_source,
        "detector_column": cols.detector_col,
        "ontology_column": cols.ontology_col,
        "generative_column": cols.generative_col,
        "vae_column": cols.vae_col,
        "vae_is_true_trained_score": vae_is_true,
        "vae_proxy_used": bool(cols.vae_col is None and allow_vae_proxy),
        "warnings": cols.warnings,
    }
    return component_df, meta


def variant_scores(components: pd.DataFrame) -> dict[str, np.ndarray]:
    d = components["Sdet_norm"].to_numpy(dtype=float)
    o = components["Sont_norm"].to_numpy(dtype=float)
    g = components["Sgen_norm"].to_numpy(dtype=float)
    v = components["Svae_norm"].to_numpy(dtype=float)

    variants = {
        "full_model_conservative": 0.55 * d + 0.40 * o + 0.05 * g,
        "no_ontology": 0.90 * d + 0.10 * g,
        "no_generative": 0.60 * d + 0.40 * o,
        "detector_only": d,
        "ontology_only": o,
        "generative_only": g,
        "vae_replacement_slot": 0.55 * d + 0.40 * o + 0.05 * v,
    }
    return {name: normalise_score(score) for name, score in variants.items()}


def evaluate_variant(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    max_thresholds: int,
) -> dict[str, Any]:
    best = best_f1_threshold(y_true, scores, max_thresholds=max_thresholds)

    result = {
        "variant": name,
        "roc_auc": roc_auc_score_manual(y_true, scores),
        "average_precision": average_precision_manual(y_true, scores),
        "best_threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "predicted_positive_rate": best["predicted_positive_rate"],
        "precision_at_1pct": precision_at_fraction(y_true, scores, 0.01),
        "precision_at_5pct": precision_at_fraction(y_true, scores, 0.05),
        "precision_at_10pct": precision_at_fraction(y_true, scores, 0.10),
        "mean_score_normal": float(np.mean(scores[y_true == 0]))
        if np.any(y_true == 0)
        else float("nan"),
        "mean_score_anomaly": float(np.mean(scores[y_true == 1]))
        if np.any(y_true == 1)
        else float("nan"),
        "effect_size_cohen_d": cohen_d(y_true, scores),
    }
    return result


def render_markdown_report(
    summary: dict[str, Any],
    results_df: pd.DataFrame,
    meta: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append("# Day 40 — Ablation Framework Report")
    lines.append("")
    lines.append(f"- Generated at: `{summary['generated_at']}`")
    lines.append(f"- Input scores: `{summary['input_scores']}`")
    lines.append(f"- Rows evaluated: **{summary['rows']}**")
    lines.append(f"- Positive/anomaly rows: **{summary['positive_rows']}**")
    lines.append(f"- Negative/normal rows: **{summary['negative_rows']}**")
    lines.append("")
    lines.append("## Column Resolution")
    lines.append("")
    lines.append(f"- Label column: `{meta['label_column']}` ({meta['label_source']})")
    lines.append(f"- Detector score column: `{meta['detector_column']}`")
    lines.append(f"- Ontology score column: `{meta['ontology_column']}`")
    lines.append(f"- Generative score column: `{meta['generative_column']}`")
    lines.append(f"- VAE score column: `{meta['vae_column']}`")
    lines.append(f"- True VAE score used: **{meta['vae_is_true_trained_score']}**")
    lines.append(f"- VAE proxy used: **{meta['vae_proxy_used']}**")
    lines.append("")

    if meta["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in meta["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Main Ablation Results")
    lines.append("")
    lines.append(
        "| Variant | ROC-AUC | AP | F1 | Precision | Recall | P@5% | Effect Size |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    display_df = results_df.sort_values("average_precision", ascending=False)
    for row in display_df.itertuples(index=False):
        lines.append(
            f"| {row.variant} | "
            f"{row.roc_auc:.4f} | "
            f"{row.average_precision:.4f} | "
            f"{row.f1:.4f} | "
            f"{row.precision:.4f} | "
            f"{row.recall:.4f} | "
            f"{row.precision_at_5pct:.4f} | "
            f"{row.effect_size_cohen_d:.4f} |"
        )

    lines.append("")
    lines.append("## Scientific Interpretation")
    lines.append("")
    lines.append(
        "- `full_model_conservative` is the current paper-safe full system score. "
        "It gives high weight to detector and ontology, and only low weight to generative Sgen."
    )
    lines.append(
        "- `no_ontology` tests whether ontology violations are actually contributing beyond statistical scoring."
    )
    lines.append(
        "- `no_generative` tests whether the system remains strong without diffusion/Sgen."
    )
    lines.append(
        "- `detector_only`, `ontology_only`, and `generative_only` are single-component baselines."
    )
    lines.append(
        "- `vae_replacement_slot` is the framework slot for replacing diffusion with a simpler VAE-style generative score. "
        "If no real VAE score column exists, this row is marked as proxy and must not be claimed as a trained VAE baseline."
    )
    lines.append("")
    lines.append("## Paper Use")
    lines.append("")
    lines.append(
        "Use this table as the Day 40 ablation-framework artifact. "
        "Day 41 should run the same framework on the final selected evaluation split and, if possible, replace the VAE proxy with a trained VAE reconstruction score."
    )
    lines.append("")

    return "\n".join(lines)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 40 ablation framework for paper-ready component analysis."
    )
    parser.add_argument(
        "--input_scores",
        default=None,
        help="Optional CSV containing labels and component scores.",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts",
        help="Directory used for auto-discovery if input_scores is omitted.",
    )
    parser.add_argument(
        "--out_dir", default="artifacts/day40", help="Output directory."
    )
    parser.add_argument(
        "--allow_vae_proxy",
        action="store_true",
        help="Allow VAE replacement slot to use a clearly-marked proxy when no true VAE score exists.",
    )
    parser.add_argument("--max_thresholds", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = (
        Path(args.input_scores)
        if args.input_scores
        else auto_discover_input(Path(args.artifacts_dir))
    )
    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Input score file is empty: {input_path}")

    cols = resolve_columns(df)
    components, meta = build_component_scores(
        df, cols, allow_vae_proxy=args.allow_vae_proxy
    )

    y_true = components["label"].to_numpy(dtype=int)
    variants = variant_scores(components)

    variant_score_df = components.copy()
    results = []

    for variant_name, scores in variants.items():
        variant_score_df[variant_name] = scores
        results.append(
            evaluate_variant(
                name=variant_name,
                y_true=y_true,
                scores=scores,
                max_thresholds=args.max_thresholds,
            )
        )

    results_df = pd.DataFrame(results).sort_values("average_precision", ascending=False)

    summary = {
        "generated_at": now_utc_iso(),
        "input_scores": input_path.as_posix(),
        "out_dir": out_dir.as_posix(),
        "rows": int(len(df)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "column_resolution": meta,
        "best_variant_by_average_precision": str(results_df.iloc[0]["variant"]),
        "best_variant_by_f1": str(
            results_df.sort_values("f1", ascending=False).iloc[0]["variant"]
        ),
        "outputs": {
            "ablation_results_csv": (out_dir / "ablation_results.csv").as_posix(),
            "variant_scores_csv": (out_dir / "variant_scores.csv").as_posix(),
            "summary_json": (out_dir / "day40_ablation_summary.json").as_posix(),
            "report_md": (out_dir / "day40_ablation_report.md").as_posix(),
        },
    }

    results_df.to_csv(out_dir / "ablation_results.csv", index=False)
    variant_score_df.to_csv(out_dir / "variant_scores.csv", index=False)
    save_json(out_dir / "day40_ablation_summary.json", summary)

    report = render_markdown_report(summary=summary, results_df=results_df, meta=meta)
    (out_dir / "day40_ablation_report.md").write_text(report + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
