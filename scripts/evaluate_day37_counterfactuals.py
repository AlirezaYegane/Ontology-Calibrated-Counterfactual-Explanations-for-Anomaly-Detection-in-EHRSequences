from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalise_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    norm_to_real = {normalise_col(c): c for c in df.columns}
    for cand in candidates:
        key = normalise_col(cand)
        if key in norm_to_real:
            return norm_to_real[key]

    # softer contains-based fallback
    for cand in candidates:
        key = normalise_col(cand)
        for norm, real in norm_to_real.items():
            if key and key in norm:
                return real

    if required:
        raise ValueError(
            "Could not find required column. "
            f"Tried: {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def to_num_series(df: pd.DataFrame, col: str | None, default: float = np.nan) -> pd.Series:
    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def parse_json_like(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "[]"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def count_listish(value: Any) -> int:
    parsed = parse_json_like(value)
    if parsed is None:
        return 0
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        return len(parsed)
    text = str(parsed).strip()
    if not text:
        return 0

    for sep in ["||", ";", "|"]:
        if sep in text:
            return len([x for x in text.split(sep) if x.strip()])

    # common forms: "add X", "remove Y", "replace A with B"
    if text.lower() in {"no_edit", "none", "no action", "unchanged"}:
        return 0
    return 1


def infer_action_type(value: Any) -> str:
    parsed = parse_json_like(value)
    if parsed is None:
        return "no_edit"

    if isinstance(parsed, list):
        text = " ".join(str(x) for x in parsed).lower()
    elif isinstance(parsed, dict):
        text = json.dumps(parsed).lower()
    else:
        text = str(parsed).lower()

    has_add = "add" in text or "insert" in text
    has_remove = "remove" in text or "delete" in text or "drop" in text
    has_replace = "replace" in text or "swap" in text or "substitute" in text

    flags = [has_add, has_remove, has_replace]
    if sum(flags) > 1:
        return "mixed"
    if has_replace:
        return "replace"
    if has_remove:
        return "remove"
    if has_add:
        return "add"
    if text.strip() in {"", "none", "nan", "no_edit", "unchanged"}:
        return "no_edit"
    return "other"


def discover_input(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path

    roots = [Path("outputs"), Path("artifacts")]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.csv", "*.json", "*.jsonl"):
            for path in root.rglob(suffix):
                text = path.as_posix().lower()
                if any(k in text for k in ["counterfactual", "counterfactuals", "day36", "cf_"]):
                    if not any(k in text for k in ["summary", "readme", "breakdown"]):
                        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover Day 36 counterfactual output. "
            "Pass it manually with --input_path."
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            for key in ["records", "results", "items", "examples", "counterfactuals"]:
                if isinstance(obj.get(key), list):
                    return pd.DataFrame(obj[key])
            return pd.DataFrame([obj])
    raise ValueError(f"Unsupported input format: {path}")


def safe_rate(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.mean())


def quantiles(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--success_threshold", type=float, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    input_path = discover_input(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    if df.empty:
        raise ValueError(f"Input table is empty: {input_path}")

    before_score_col = pick_col(
        df,
        [
            # Day 36 repair outputs: primary score is ontology/violation repair score.
            "original_violation_score", "sont_before", "s_ont_before",
            "ontology_score_before", "original_sont",
            # Generic calibrated-score names, if future runs include them.
            "scal_before", "s_cal_before", "score_before", "before_score",
            "original_scal", "original_score", "calibrated_score_before",
            "scal_original", "s_cal_original",
            # Detector/input score is kept as a fallback only.
            "input_score",
        ],
        required=True,
    )
    after_score_col = pick_col(
        df,
        [
            # Day 36 repair outputs: primary score is ontology/violation repair score.
            "counterfactual_violation_score", "sont_after", "s_ont_after",
            "ontology_score_after", "counterfactual_sont",
            # Generic calibrated-score names, if future runs include them.
            "scal_after", "s_cal_after", "score_after", "after_score",
            "counterfactual_scal", "counterfactual_score", "cf_scal",
            "calibrated_score_after", "s_cal_star", "scal_star",
        ],
        required=True,
    )

    anomaly_type_col = pick_col(
        df,
        ["anomaly_type", "type", "issue_type", "violation_type", "source_anomaly_type"],
        required=False,
    )

    record_id_col = pick_col(
        df,
        ["record_id", "row_id", "hadm_id", "subject_id", "example_id", "id"],
        required=False,
    )

    action_col = pick_col(
        df,
        ["action", "actions", "edit_action", "edit_actions", "edits", "edit_sequence", "operation"],
        required=False,
    )

    edit_count_col = pick_col(
        df,
        ["edit_count", "n_edits", "num_edits", "number_of_edits"],
        required=False,
    )

    before_viol_col = pick_col(
        df,
        [
            "violations_before_count", "n_violations_before", "num_violations_before",
            "original_violation_score", "sont_before", "s_ont_before",
            "ontology_score_before", "original_sont",
            "violations_before",
        ],
        required=False,
    )
    after_viol_col = pick_col(
        df,
        [
            "violations_after_count", "n_violations_after", "num_violations_after",
            "counterfactual_violation_score", "sont_after", "s_ont_after",
            "ontology_score_after", "counterfactual_sont",
            "violations_after",
        ],
        required=False,
    )

    before_score = to_num_series(df, before_score_col)
    after_score = to_num_series(df, after_score_col)

    eval_df = pd.DataFrame()
    eval_df["source_row_index"] = np.arange(len(df))
    eval_df["record_id"] = df[record_id_col] if record_id_col else eval_df["source_row_index"]
    eval_df["anomaly_type"] = df[anomaly_type_col].astype(str) if anomaly_type_col else "unknown"
    eval_df["score_before"] = before_score
    eval_df["score_after"] = after_score
    eval_df["delta_scal"] = eval_df["score_before"] - eval_df["score_after"]
    eval_df["relative_reduction"] = eval_df["delta_scal"] / eval_df["score_before"].replace(0, np.nan)
    eval_df["score_reduced"] = eval_df["delta_scal"] > 0

    if args.success_threshold is not None:
        eval_df["below_success_threshold_after"] = eval_df["score_after"] <= args.success_threshold
    else:
        eval_df["below_success_threshold_after"] = np.nan

    if edit_count_col:
        eval_df["edit_count"] = pd.to_numeric(df[edit_count_col], errors="coerce").fillna(0).astype(int)
    elif action_col:
        eval_df["edit_count"] = df[action_col].apply(count_listish).astype(int)
    else:
        eval_df["edit_count"] = np.nan

    if action_col:
        eval_df["action_type"] = df[action_col].apply(infer_action_type)
        eval_df["action_raw"] = df[action_col].astype(str)
    else:
        eval_df["action_type"] = "unknown"
        eval_df["action_raw"] = ""

    if before_viol_col:
        before_viol_numeric = pd.to_numeric(df[before_viol_col], errors="coerce")
        if before_viol_numeric.isna().all():
            before_viol = df[before_viol_col].apply(count_listish).astype(float)
        else:
            before_viol = before_viol_numeric
    else:
        before_viol = pd.Series([np.nan] * len(df), index=df.index)

    if after_viol_col:
        after_viol_numeric = pd.to_numeric(df[after_viol_col], errors="coerce")
        if after_viol_numeric.isna().all():
            after_viol = df[after_viol_col].apply(count_listish).astype(float)
        else:
            after_viol = after_viol_numeric
    else:
        after_viol = pd.Series([np.nan] * len(df), index=df.index)

    eval_df["violations_before"] = before_viol
    eval_df["violations_after"] = after_viol
    eval_df["ontology_improved"] = eval_df["violations_after"] < eval_df["violations_before"]
    eval_df["ontology_fully_resolved"] = (
        (eval_df["violations_before"] > 0) & (eval_df["violations_after"] <= 0)
    )

    # Keep a few useful original columns for manual inspection
    for col in [before_score_col, after_score_col, anomaly_type_col, action_col, before_viol_col, after_viol_col]:
        if col and col not in eval_df.columns:
            eval_df[f"raw_{col}"] = df[col]

    valid_delta = eval_df["delta_scal"].notna()
    valid_viol = eval_df["violations_before"].notna() & eval_df["violations_after"].notna()
    valid_edits = eval_df["edit_count"].notna()

    by_type_rows = []
    for anomaly_type, group in eval_df.groupby("anomaly_type", dropna=False):
        by_type_rows.append(
            {
                "anomaly_type": str(anomaly_type),
                "count": int(len(group)),
                "mean_score_before": float(group["score_before"].mean()),
                "mean_score_after": float(group["score_after"].mean()),
                "mean_delta_scal": float(group["delta_scal"].mean()),
                "median_delta_scal": float(group["delta_scal"].median()),
                "pct_score_reduced": safe_rate(group["score_reduced"]),
                "mean_edit_count": None if group["edit_count"].isna().all() else float(group["edit_count"].mean()),
                "pct_one_edit": None if group["edit_count"].isna().all() else safe_rate(group["edit_count"] == 1),
                "pct_two_edits": None if group["edit_count"].isna().all() else safe_rate(group["edit_count"] == 2),
                "pct_more_than_two_edits": None if group["edit_count"].isna().all() else safe_rate(group["edit_count"] > 2),
                "pct_ontology_improved": None if not valid_viol.loc[group.index].any() else safe_rate(group["ontology_improved"]),
                "pct_ontology_fully_resolved": None if not valid_viol.loc[group.index].any() else safe_rate(group["ontology_fully_resolved"]),
            }
        )

    by_type_df = pd.DataFrame(by_type_rows).sort_values(["count", "anomaly_type"], ascending=[False, True])

    action_counts = Counter(eval_df["action_type"].fillna("unknown").astype(str))
    summary = {
        "day": 37,
        "title": "Counterfactual Evaluation",
        "generated_at": now_utc(),
        "input_path": input_path.as_posix(),
        "out_dir": out_dir.as_posix(),
        "rows": int(len(eval_df)),
        "columns_detected": {
            "record_id": record_id_col,
            "anomaly_type": anomaly_type_col,
            "score_before": before_score_col,
            "score_after": after_score_col,
            "action": action_col,
            "edit_count": edit_count_col,
            "violations_before": before_viol_col,
            "violations_after": after_viol_col,
        },
        "score_reduction": {
            "valid_rows": int(valid_delta.sum()),
            "pct_score_reduced": None if valid_delta.sum() == 0 else safe_rate(eval_df.loc[valid_delta, "score_reduced"]),
            "delta_scal": quantiles(eval_df["delta_scal"]),
            "relative_reduction": quantiles(eval_df["relative_reduction"]),
            "score_before": quantiles(eval_df["score_before"]),
            "score_after": quantiles(eval_df["score_after"]),
        },
        "sparsity": {
            "valid_rows": int(valid_edits.sum()),
            "edit_count": quantiles(eval_df["edit_count"]),
            "pct_zero_edit": None if valid_edits.sum() == 0 else safe_rate(eval_df.loc[valid_edits, "edit_count"] == 0),
            "pct_one_edit": None if valid_edits.sum() == 0 else safe_rate(eval_df.loc[valid_edits, "edit_count"] == 1),
            "pct_two_edits": None if valid_edits.sum() == 0 else safe_rate(eval_df.loc[valid_edits, "edit_count"] == 2),
            "pct_one_or_two_edits": None if valid_edits.sum() == 0 else safe_rate(eval_df.loc[valid_edits, "edit_count"].isin([1, 2])),
            "pct_more_than_two_edits": None if valid_edits.sum() == 0 else safe_rate(eval_df.loc[valid_edits, "edit_count"] > 2),
            "action_type_counts": dict(action_counts),
        },
        "ontology_resolution": {
            "valid_rows": int(valid_viol.sum()),
            "mean_violations_before": None if valid_viol.sum() == 0 else float(eval_df.loc[valid_viol, "violations_before"].mean()),
            "mean_violations_after": None if valid_viol.sum() == 0 else float(eval_df.loc[valid_viol, "violations_after"].mean()),
            "pct_ontology_improved": None if valid_viol.sum() == 0 else safe_rate(eval_df.loc[valid_viol, "ontology_improved"]),
            "pct_ontology_fully_resolved": None if valid_viol.sum() == 0 else safe_rate(eval_df.loc[valid_viol, "ontology_fully_resolved"]),
        },
        "success_threshold": args.success_threshold,
        "threshold_success": {
            "pct_below_threshold_after": None
            if args.success_threshold is None
            else safe_rate(eval_df["below_success_threshold_after"].fillna(False)),
        },
        "by_anomaly_type": by_type_rows,
        "interpretation": (
            "Counterfactual evaluation reports repair-score reduction, edit sparsity, and ontology-resolution metrics. "
            "The result should be interpreted as explanation efficacy, not as a standalone detector-performance claim."
        ),
    }

    eval_df.to_csv(out_dir / "counterfactual_eval_records.csv", index=False)
    by_type_df.to_csv(out_dir / "counterfactual_eval_by_type.csv", index=False)

    (out_dir / "day37_counterfactual_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pct_red = summary["score_reduction"]["pct_score_reduced"]
    pct_one_two = summary["sparsity"]["pct_one_or_two_edits"]
    pct_resolved = summary["ontology_resolution"]["pct_ontology_fully_resolved"]

    readme = []
    readme.append("# Day 37 — Counterfactual Evaluation")
    readme.append("")
    readme.append("## Status")
    readme.append("Complete.")
    readme.append("")
    readme.append("## Purpose")
    readme.append(
        "Day 37 evaluates whether generated counterfactuals reduce ontology/repair scores, "
        "remain sparse, and resolve ontology-related inconsistencies."
    )
    readme.append("")
    readme.append("## Input")
    readme.append(f"- Source file: `{input_path.as_posix()}`")
    readme.append("")
    readme.append("## Main Metrics")
    readme.append("")
    readme.append(f"- Rows evaluated: **{len(eval_df)}**")
    readme.append(f"- Percentage with reduced score: **{pct_red if pct_red is not None else 'n/a'}**")
    readme.append(f"- Mean score reduction: **{summary['score_reduction']['delta_scal']['mean']}**")
    readme.append(f"- Median score reduction: **{summary['score_reduction']['delta_scal']['median']}**")
    readme.append(f"- Mean edit count: **{summary['sparsity']['edit_count']['mean']}**")
    readme.append(f"- One-or-two-edit rate: **{pct_one_two if pct_one_two is not None else 'n/a'}**")
    readme.append(f"- Ontology fully resolved rate: **{pct_resolved if pct_resolved is not None else 'n/a'}**")
    readme.append("")
    readme.append("## Files")
    readme.append("")
    readme.append("- `counterfactual_eval_records.csv`")
    readme.append("- `counterfactual_eval_by_type.csv`")
    readme.append("- `day37_counterfactual_eval_summary.json`")
    readme.append("")
    readme.append("## Scientific Note")
    readme.append(
        "These metrics support the paper's explanation-efficacy analysis. "
        "They should be reported honestly: strong score reduction and sparse edits support the method; "
        "weak or uneven results should be treated as failure-mode evidence rather than hidden."
    )
    readme.append("")

    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
