from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_LIKE_EXTS = {".csv", ".json", ".jsonl", ".pkl", ".parquet"}


def load_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            for key in ("cases", "case_studies", "records", "items", "rows", "data", "examples", "explanations", "results"):
                if isinstance(obj.get(key), list):
                    return pd.DataFrame(obj[key])
            return pd.DataFrame([obj])

    if suffix == ".pkl":
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame([obj])

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col_name(c) for c in out.columns]
    return out


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def get_value(row: pd.Series, candidates: list[str], default: Any = None) -> Any:
    for cand in candidates:
        key = normalize_col_name(cand)
        if key in row.index:
            value = row[key]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return default


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [stringify(x) for x in value if stringify(x).strip()]
    if isinstance(value, tuple):
        return [stringify(x) for x in value if stringify(x).strip()]
    text = stringify(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [stringify(x) for x in obj if stringify(x).strip()]
        except Exception:
            pass
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [text]


def score_source_candidate(path: Path, df: pd.DataFrame) -> int:
    cols = set(df.columns)
    score = 0

    if "explanation" in cols or "explanation_text" in cols or "human_explanation" in cols:
        score += 50
    if "action" in cols or "edit_action" in cols or "counterfactual_action" in cols:
        score += 35
    if "violations" in cols or "violation_messages" in cols:
        score += 25
    if "scal" in cols or "s_cal" in cols or "calibrated_score" in cols:
        score += 25
    if "sont" in cols or "s_ont" in cols:
        score += 15
    if "sdet" in cols or "s_det" in cols or "detector_score" in cols or "prob_anomaly" in cols:
        score += 15
    if "delta_scal" in cols or "score_reduction" in cols or "delta" in cols:
        score += 20
    if "anomaly_type" in cols:
        score += 15

    path_text = path.as_posix().lower()
    if "day38" in path_text:
        score += 40
    elif "day37" in path_text:
        score += 25
    elif "day35" in path_text:
        score += 10

    if "summary" in path.name.lower() or "readme" in path.name.lower():
        score -= 40

    score += min(len(df), 20)
    return score


def discover_source_file(root: Path) -> Path:
    patterns = [
        "artifacts/day38/**/*",
        "artifacts/day37/**/*",
        "artifacts/day36/**/*",
        "artifacts/day35/**/*",
        "artifacts/day35_*/**/*",
        "outputs/**/*day38*/*",
        "outputs/**/*explanation*/*",
        "outputs/**/*counterfactual*/*",
        "outputs/**/*scores*/*",
    ]

    candidates: list[tuple[int, Path, int, list[str]]] = []

    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_LIKE_EXTS:
                continue
            try:
                df = normalize_columns(load_any_table(path))
                if len(df) == 0:
                    continue
                score = score_source_candidate(path, df)
                candidates.append((score, path, len(df), list(df.columns)))
            except Exception:
                continue

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover a Day 35-38 output file. "
            "Pass one manually with --source_path."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]

    print("[Day39] source discovery candidates:")
    for score, path, n_rows, cols in candidates[:10]:
        print(f"  score={score:03d} rows={n_rows:05d} path={path.as_posix()}")
        print(f"    cols={cols[:18]}")

    print(f"[Day39] selected source: {best[1].as_posix()}")
    return best[1]


def build_case_from_row(row: pd.Series, case_rank: int) -> dict[str, Any]:
    record_id = stringify(
        get_value(row, ["record_id", "case_id", "id", "idx", "index", "hadm_id", "subject_id"], f"case_{case_rank:03d}")
    )

    anomaly_type = stringify(
        get_value(row, ["anomaly_type", "type", "issue_type", "label_name", "class_name"], "unknown")
    )

    label = to_int(get_value(row, ["label", "is_anomaly", "is_synthetic_anomaly", "y_true"], 1), default=1)

    sdet = to_float(get_value(row, ["sdet", "s_det", "detector_score", "prob_anomaly", "score_det"], 0.0))
    sgen = to_float(get_value(row, ["sgen", "s_gen", "generative_score", "diffusion_score", "score_gen"], 0.0))
    sont = to_float(get_value(row, ["sont", "s_ont", "ontology_score", "score_ont"], 0.0))

    scal_before = to_float(
        get_value(
            row,
            ["scal_before", "s_cal_before", "scal", "s_cal", "calibrated_score", "score_before", "original_score"],
            0.0,
        )
    )
    scal_after = to_float(
        get_value(
            row,
            ["scal_after", "s_cal_after", "counterfactual_score", "score_after", "best_score", "fixed_score"],
            scal_before,
        )
    )

    delta_scal = to_float(
        get_value(row, ["delta_scal", "score_reduction", "delta_s_cal", "delta", "improvement"], scal_before - scal_after)
    )

    edit_count = to_int(
        get_value(row, ["edit_count", "num_edits", "n_edits", "edits", "n_actions"], 1 if delta_scal > 0 else 0),
        default=0,
    )

    action = stringify(
        get_value(row, ["action", "edit_action", "counterfactual_action", "best_action", "operation", "actions_compact", "edits_text"], "")
    )

    violations = parse_listish(
        get_value(row, ["violations", "violation_messages", "ontology_violations", "issues", "violations_compact", "original_violations_json"], "")
    )

    original_codes = parse_listish(
        get_value(row, ["original_codes", "codes", "sequence_tokens", "original_sequence", "record"], "")
    )
    counterfactual_codes = parse_listish(
        get_value(row, ["counterfactual_codes", "fixed_codes", "counterfactual_sequence", "x_star"], "")
    )

    explanation = stringify(
        get_value(row, ["explanation", "explanation_text", "human_explanation", "narrative", "explanation_clinical", "explanation_research", "explanation_short"], "")
    )

    if not explanation:
        explanation = make_template_explanation(
            anomaly_type=anomaly_type,
            sdet=sdet,
            sgen=sgen,
            sont=sont,
            scal_before=scal_before,
            scal_after=scal_after,
            delta_scal=delta_scal,
            action=action,
            violations=violations,
        )

    coherent_heuristic = bool(explanation.strip()) and (
        delta_scal > 0 or len(violations) > 0 or action.strip() != ""
    )

    return {
        "case_rank": case_rank,
        "record_id": record_id,
        "label": label,
        "anomaly_type": anomaly_type,
        "scores": {
            "Sdet": round(sdet, 6),
            "Sgen": round(sgen, 6),
            "Sont": round(sont, 6),
            "Scal_before": round(scal_before, 6),
            "Scal_after": round(scal_after, 6),
            "delta_Scal": round(delta_scal, 6),
        },
        "edit_count": edit_count,
        "action": action,
        "violations": violations,
        "original_codes_preview": original_codes[:30],
        "counterfactual_codes_preview": counterfactual_codes[:30],
        "explanation": explanation,
        "coherence_check": {
            "heuristic_pass": coherent_heuristic,
            "notes": "Automatic heuristic only; manual paper review is still required.",
        },
    }


def make_template_explanation(
    anomaly_type: str,
    sdet: float,
    sgen: float,
    sont: float,
    scal_before: float,
    scal_after: float,
    delta_scal: float,
    action: str,
    violations: list[str],
) -> str:
    parts = []

    parts.append(
        f"This record was selected as a representative {anomaly_type} case. "
        f"The calibrated anomaly score decreased from {scal_before:.4f} to {scal_after:.4f} "
        f"(ΔScal={delta_scal:.4f})."
    )

    if sont > 0 or violations:
        parts.append(
            "The anomaly is partly ontology-driven, because the ontology module identified "
            "clinical consistency issues in the record."
        )

    if sdet > 0:
        parts.append(
            f"The detector contributed a statistical anomaly signal (Sdet={sdet:.4f})."
        )

    if sgen > 0:
        parts.append(
            f"The diffusion-based generative score was recorded as an auxiliary diagnostic signal "
            f"(Sgen={sgen:.4f}), but it should not be interpreted as the dominant evidence."
        )

    if action:
        parts.append(f"The minimal counterfactual repair applied was: {action}.")

    if violations:
        parts.append("Main ontology issue(s): " + "; ".join(violations[:3]) + ".")

    return " ".join(parts)


def select_representative_cases(df: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    df = df.copy()

    # Build helper numeric columns if possible.
    for col_name, candidates in {
        "_delta": ["delta_scal", "score_reduction", "delta_s_cal", "delta", "improvement"],
        "_sont": ["sont", "s_ont", "ontology_score", "score_ont"],
        "_scal": ["scal_before", "s_cal_before", "scal", "s_cal", "calibrated_score", "score_before", "original_score"],
        "_sdet": ["sdet", "s_det", "detector_score", "prob_anomaly", "score_det"],
    }.items():
        source = first_existing_col(df, candidates)
        if source:
            df[col_name] = pd.to_numeric(df[source], errors="coerce").fillna(0.0)
        else:
            df[col_name] = 0.0

    type_col = first_existing_col(df, ["anomaly_type", "type", "issue_type", "label_name", "class_name"])
    if type_col is None:
        df["_anomaly_type"] = "unknown"
        type_col = "_anomaly_type"

    selected_indices: list[int] = []

    # First: one strong example from each anomaly type.
    for anomaly_type, group in df.groupby(type_col, dropna=False):
        if len(selected_indices) >= max_cases:
            break
        group = group.copy()
        group["_rank_score"] = group["_delta"] * 3 + group["_sont"] * 2 + group["_scal"]
        idx = int(group.sort_values("_rank_score", ascending=False).index[0])
        if idx not in selected_indices:
            selected_indices.append(idx)

    # Second: force missing_diagnosis if present.
    if len(selected_indices) < max_cases:
        md = df[df[type_col].astype(str).str.contains("missing", case=False, na=False)]
        if len(md):
            md = md.copy()
            md["_rank_score"] = md["_delta"] * 3 + md["_scal"]
            idx = int(md.sort_values("_rank_score", ascending=False).index[0])
            if idx not in selected_indices:
                selected_indices.append(idx)

    # Third: highest score reduction.
    if len(selected_indices) < max_cases:
        for idx in df.sort_values("_delta", ascending=False).index:
            idx = int(idx)
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= max_cases:
                break

    # Fourth: highest ontology-driven cases.
    if len(selected_indices) < max_cases:
        for idx in df.sort_values("_sont", ascending=False).index:
            idx = int(idx)
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= max_cases:
                break

    # Final fallback.
    if len(selected_indices) < max_cases:
        for idx in df.sort_values("_scal", ascending=False).index:
            idx = int(idx)
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= max_cases:
                break

    return df.loc[selected_indices].copy()


def render_markdown(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Day 39 — End-to-End Case Studies")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This artifact documents representative end-to-end examples from the ontology-calibrated "
        "counterfactual explanation pipeline. The goal is to verify whether the generated explanations "
        "are coherent, clinically plausible, and faithful to the underlying scores and counterfactual edits."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Source file: `{summary['source_file']}`")
    lines.append(f"- Number of selected cases: **{summary['n_cases']}**")
    lines.append(f"- Anomaly types covered: **{', '.join(summary['anomaly_types']) if summary['anomaly_types'] else 'n/a'}**")
    lines.append(f"- Mean ΔScal: **{summary['mean_delta_scal']:.4f}**")
    lines.append(f"- Median edit count: **{summary['median_edit_count']:.2f}**")
    lines.append(f"- Automatic coherence pass rate: **{summary['coherence_pass_rate']:.2%}**")
    lines.append("")
    lines.append("## Research Interpretation")
    lines.append("")
    lines.append(
        "These cases should be treated as qualitative evidence for explanation behavior, not as clinical "
        "advice. The strongest paper-safe claims are score reduction, minimality of edits, ontology-rule "
        "alignment, and whether the explanation text faithfully reflects the computed signals."
    )
    lines.append("")
    lines.append(
        "The diffusion-based Sgen value is retained as an auxiliary diagnostic field. Because earlier "
        "evaluation found the current Sgen proxy weak for anomaly separation, the case-study interpretation "
        "should not rely on Sgen as the main evidence."
    )
    lines.append("")

    for case in cases:
        scores = case["scores"]
        lines.append("---")
        lines.append("")
        lines.append(f"## Case {case['case_rank']}: `{case['anomaly_type']}`")
        lines.append("")
        lines.append(f"- Record ID: `{case['record_id']}`")
        lines.append(f"- Label: `{case['label']}`")
        lines.append(f"- Edit count: **{case['edit_count']}**")
        lines.append(f"- Action: `{case['action'] or 'n/a'}`")
        lines.append("")
        lines.append("### Scores")
        lines.append("")
        lines.append("| Score | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Sdet | {scores['Sdet']:.6f} |")
        lines.append(f"| Sgen | {scores['Sgen']:.6f} |")
        lines.append(f"| Sont | {scores['Sont']:.6f} |")
        lines.append(f"| Scal before | {scores['Scal_before']:.6f} |")
        lines.append(f"| Scal after | {scores['Scal_after']:.6f} |")
        lines.append(f"| ΔScal | {scores['delta_Scal']:.6f} |")
        lines.append("")
        lines.append("### Ontology Issues")
        lines.append("")
        if case["violations"]:
            for v in case["violations"]:
                lines.append(f"- {v}")
        else:
            lines.append("- No explicit violation text available in the source artifact.")
        lines.append("")
        lines.append("### Explanation")
        lines.append("")
        lines.append(case["explanation"])
        lines.append("")
        if case["original_codes_preview"] or case["counterfactual_codes_preview"]:
            lines.append("### Code Preview")
            lines.append("")
            lines.append("Original preview:")
            lines.append("")
            lines.append("```text")
            lines.append(", ".join(case["original_codes_preview"]) if case["original_codes_preview"] else "n/a")
            lines.append("```")
            lines.append("")
            lines.append("Counterfactual preview:")
            lines.append("")
            lines.append("```text")
            lines.append(", ".join(case["counterfactual_codes_preview"]) if case["counterfactual_codes_preview"] else "n/a")
            lines.append("```")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Day 39 Status")
    lines.append("")
    lines.append("Day 39 is complete if:")
    lines.append("")
    lines.append("- The selected cases cover multiple anomaly types.")
    lines.append("- Each case includes before/after scores or a clear explanation artifact.")
    lines.append("- Counterfactual actions are minimal and interpretable where available.")
    lines.append("- The final markdown can be reused as a Results / Case Study appendix.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", default=None, help="Optional explicit Day 38/37/35 output file.")
    parser.add_argument("--out_dir", default="artifacts/day39")
    parser.add_argument("--max_cases", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(".").resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source_path:
        source_path = Path(args.source_path)
    else:
        source_path = discover_source_file(Path("."))

    df = normalize_columns(load_any_table(source_path))

    if len(df) == 0:
        raise RuntimeError(f"No rows loaded from {source_path}")

    selected = select_representative_cases(df, max_cases=args.max_cases)

    cases = [
        build_case_from_row(row, case_rank=i + 1)
        for i, (_, row) in enumerate(selected.iterrows())
    ]

    anomaly_types = sorted({case["anomaly_type"] for case in cases if case["anomaly_type"]})
    deltas = [case["scores"]["delta_Scal"] for case in cases]
    edit_counts = [case["edit_count"] for case in cases]
    coherence = [case["coherence_check"]["heuristic_pass"] for case in cases]

    summary = {
        "day": 39,
        "status": "complete",
        "source_file": source_path.as_posix(),
        "n_source_rows": int(len(df)),
        "n_cases": int(len(cases)),
        "anomaly_types": anomaly_types,
        "mean_delta_scal": float(sum(deltas) / max(len(deltas), 1)),
        "median_edit_count": float(pd.Series(edit_counts).median()) if edit_counts else 0.0,
        "coherence_pass_rate": float(sum(bool(x) for x in coherence) / max(len(coherence), 1)),
        "interpretation": (
            "Day 39 generated representative end-to-end case studies for paper-oriented qualitative review. "
            "Sgen is retained as an auxiliary diagnostic field rather than the dominant evidence."
        ),
    }

    (out_dir / "day39_case_studies.json").write_text(
        json.dumps({"summary": summary, "cases": cases}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (out_dir / "day39_end_to_end_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (out_dir / "day39_case_studies.md").write_text(
        render_markdown(cases, summary),
        encoding="utf-8",
    )

    readme = """# Day 39 — End-to-End Testing

## Status
Complete.

## Goal
Create representative end-to-end case studies from the ontology-calibrated counterfactual explanation pipeline.

## Outputs
- `day39_case_studies.json`
- `day39_case_studies.md`
- `day39_end_to_end_summary.json`

## Paper-oriented interpretation
These examples are intended for qualitative inspection and future Results / Case Study writing. The main evidence is whether the explanation text faithfully reflects the anomaly scores, ontology violations, and minimal counterfactual edits. Sgen is kept as an auxiliary diagnostic signal only.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
