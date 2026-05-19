from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCORE_CANDIDATES = [
    "artifacts/day43/day43_profiled_records.csv",
    "artifacts/day42/core_system_assessment_scores.csv",
    "artifacts/day41/day41_ablation_scores.csv",
    "artifacts/day41/ablation_scores.csv",
    "artifacts/day40/day40_ablation_score_artifact.csv",
    "artifacts/day39/day39_end_to_end_case_studies.csv",
    "artifacts/day39/end_to_end_case_studies.csv",
    "artifacts/day36_repair_ready/repair_ready_scores.csv",
    "artifacts/day35_7/strict_sont_ablation_scores.csv",
    "artifacts/day35_5/paper_ready_metrics.csv",
    "artifacts/day35/calibrated_scores.csv",
]


ID_COLUMNS = [
    "record_id",
    "case_id",
    "patient_id",
    "hadm_id",
    "source",
    "sample_id",
    "encounter_id",
]


SCORE_ALIASES = {
    "sdet": ["sdet", "Sdet", "detector_score", "prob_anomaly", "det_score", "score_detector"],
    "sont": ["sont", "Sont", "ontology_score", "score_ontology", "rule_score"],
    "sgen": ["sgen", "Sgen", "generative_score", "score_generative", "diffusion_score"],
    "scal": ["scal", "Scal", "calibrated_score", "score_calibrated", "final_score", "score"],
}


TEXT_COLUMNS = [
    "explanation",
    "explanation_text",
    "human_explanation",
    "counterfactual_explanation",
    "rationale",
    "summary",
]


EDIT_COLUMNS = [
    "minimal_edits",
    "edits",
    "edit_summary",
    "counterfactual_edits",
    "repair_actions",
    "actions",
]


PREVIEW_COLUMNS = [
    "original_codes_preview",
    "counterfactual_codes_preview",
    "original_preview",
    "counterfactual_preview",
    "sequence_tokens",
    "tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 44 user-facing CLI for record-level anomaly explanation."
    )
    parser.add_argument(
        "--scores_csv",
        default=None,
        help="Path to a CSV/JSON/JSONL artifact containing scores or case-study rows.",
    )
    parser.add_argument(
        "--record_id",
        default=None,
        help="Optional record/case/source id to explain.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index to explain if --record_id is not provided.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List explainable records instead of generating one explanation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of rows to show with --list.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/day44",
        help="Directory for Day 44 explanation outputs.",
    )
    parser.add_argument(
        "--title",
        default="Day 44 Record-Level Explanation",
        help="Title used in the Markdown output.",
    )
    return parser.parse_args()


def find_default_scores_path() -> Path:
    for candidate in DEFAULT_SCORE_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p

    artifact_csvs = sorted(
        Path("artifacts").glob("**/*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if artifact_csvs:
        return artifact_csvs[0]

    raise FileNotFoundError(
        "No score/case-study CSV was found. Pass --scores_csv explicitly. "
        "Searched common Day 35-Day 43 artifact paths and artifacts/**/*.csv."
    )


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input artifact does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            for key in ["rows", "cases", "records", "items", "data"]:
                if isinstance(raw.get(key), list):
                    return pd.DataFrame(raw[key])
            return pd.DataFrame([raw])
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported input format: {path.suffix}")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_text(value: Any, default: str = "not specified") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return default
    return text


def get_score(row: pd.Series, key: str) -> float | None:
    for col in SCORE_ALIASES[key]:
        if col in row.index:
            value = safe_float(row[col])
            if value is not None:
                return value

    lower_map = {str(c).lower(): c for c in row.index}
    for col in SCORE_ALIASES[key]:
        real_col = lower_map.get(col.lower())
        if real_col is not None:
            value = safe_float(row[real_col])
            if value is not None:
                return value
    return None


def parse_maybe_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, tuple):
            return [str(x) for x in parsed]
        if isinstance(parsed, dict):
            return [f"{k}: {v}" for k, v in parsed.items()]
    except (ValueError, SyntaxError):
        pass

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    if "," in text and len(text) < 500:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def select_row(df: pd.DataFrame, record_id: str | None, row_index: int) -> tuple[int, pd.Series]:
    if len(df) == 0:
        raise ValueError("Input artifact has no rows.")

    if record_id is not None:
        for col in ID_COLUMNS:
            if col in df.columns:
                matches = df[df[col].astype(str) == str(record_id)]
                if len(matches) > 0:
                    idx = int(matches.index[0])
                    return idx, df.loc[idx]

        raise ValueError(
            f"Could not find record_id={record_id!r}. "
            f"Available id columns: {[c for c in ID_COLUMNS if c in df.columns]}"
        )

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} is outside table size {len(df)}.")

    idx = int(df.index[row_index])
    return idx, df.iloc[row_index]


def infer_identifier(row: pd.Series, fallback_idx: int) -> str:
    for col in ID_COLUMNS:
        if col in row.index:
            value = row[col]
            try:
                if not pd.isna(value):
                    return str(value)
            except TypeError:
                return str(value)
    return f"row_{fallback_idx}"


def risk_band(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "moderate-high"
    if score >= 0.40:
        return "moderate"
    return "low"


def score_line(name: str, value: float | None, note: str = "") -> str:
    if value is None:
        return f"- {name}: n/a{note}"
    return f"- {name}: {value:.6f}{note}"


def collect_existing_text(row: pd.Series, columns: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in columns:
        if col in row.index:
            values = parse_maybe_list(row[col])
            if values:
                out[col] = values
    return out


def build_explanation(row: pd.Series, row_idx: int, source_path: Path) -> dict[str, Any]:
    identifier = infer_identifier(row, row_idx)

    sdet = get_score(row, "sdet")
    sont = get_score(row, "sont")
    sgen = get_score(row, "sgen")
    scal = get_score(row, "scal")

    label = row.get("label", row.get("y_true", row.get("target", None)))
    anomaly_type = row.get("anomaly_type", row.get("anomaly_family", row.get("variant", "not specified")))

    text_evidence = collect_existing_text(row, TEXT_COLUMNS)
    edit_evidence = collect_existing_text(row, EDIT_COLUMNS)
    preview_evidence = collect_existing_text(row, PREVIEW_COLUMNS)

    band = risk_band(scal if scal is not None else sdet)

    components = {
        "Sdet": sdet,
        "Sont": sont,
        "Sgen": sgen,
        "Scal": scal,
    }

    if scal is not None:
        primary_score_sentence = (
            f"The calibrated anomaly score is {scal:.4f}, which places this record in the "
            f"{band} anomaly-evidence band for this interface."
        )
    elif sdet is not None:
        primary_score_sentence = (
            f"No calibrated score was available, so the detector score {sdet:.4f} is used "
            f"as the visible primary anomaly signal. This places the record in the {band} band."
        )
    else:
        primary_score_sentence = (
            "No calibrated or detector score was available in the selected artifact. "
            "The interface can still display stored explanation/counterfactual evidence."
        )

    ontology_sentence = (
        "The ontology score is available and should be interpreted as the main semantic consistency signal."
        if sont is not None
        else "No ontology score was available for this row."
    )

    generative_sentence = (
        "The generative score is shown as a diagnostic auxiliary signal only; it should not be over-interpreted as the main evidence source."
        if sgen is not None
        else "No generative score was available for this row."
    )

    if edit_evidence:
        cf_sentence = "Counterfactual or repair evidence is available for this record."
    else:
        cf_sentence = "No explicit counterfactual edit list was found in this artifact row."

    explanation = {
        "record_identifier": identifier,
        "source_artifact": str(source_path),
        "row_index": int(row_idx),
        "label": safe_text(label, default="not specified"),
        "anomaly_type": safe_text(anomaly_type, default="not specified"),
        "risk_band": band,
        "score_components": components,
        "interpretation": {
            "primary_score": primary_score_sentence,
            "ontology": ontology_sentence,
            "generative": generative_sentence,
            "counterfactual": cf_sentence,
        },
        "stored_explanation_text": text_evidence,
        "counterfactual_or_edit_evidence": edit_evidence,
        "record_previews": preview_evidence,
    }
    return explanation


def markdown_from_explanation(expl: dict[str, Any], title: str) -> str:
    scores = expl["score_components"]
    interp = expl["interpretation"]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Record")
    lines.append("")
    lines.append(f"- Record identifier: `{expl['record_identifier']}`")
    lines.append(f"- Source artifact: `{expl['source_artifact']}`")
    lines.append(f"- Row index: `{expl['row_index']}`")
    lines.append(f"- Label: `{expl['label']}`")
    lines.append(f"- Anomaly type / variant: `{expl['anomaly_type']}`")
    lines.append(f"- Interface risk band: **{expl['risk_band']}**")
    lines.append("")
    lines.append("## Score Components")
    lines.append("")
    lines.append(score_line("Sdet / detector score", scores.get("Sdet")))
    lines.append(score_line("Sont / ontology score", scores.get("Sont")))
    lines.append(score_line("Sgen / generative score", scores.get("Sgen"), "  _(diagnostic only)_"))
    lines.append(score_line("Scal / calibrated score", scores.get("Scal")))
    lines.append("")
    lines.append("## Human-Readable Interpretation")
    lines.append("")
    lines.append(f"- {interp['primary_score']}")
    lines.append(f"- {interp['ontology']}")
    lines.append(f"- {interp['generative']}")
    lines.append(f"- {interp['counterfactual']}")
    lines.append("")

    if expl["stored_explanation_text"]:
        lines.append("## Stored Explanation Evidence")
        lines.append("")
        for col, values in expl["stored_explanation_text"].items():
            lines.append(f"### `{col}`")
            for item in values[:12]:
                lines.append(f"- {item}")
            lines.append("")

    if expl["counterfactual_or_edit_evidence"]:
        lines.append("## Counterfactual / Edit Evidence")
        lines.append("")
        for col, values in expl["counterfactual_or_edit_evidence"].items():
            lines.append(f"### `{col}`")
            for item in values[:20]:
                lines.append(f"- {item}")
            lines.append("")

    if expl["record_previews"]:
        lines.append("## Record Preview")
        lines.append("")
        for col, values in expl["record_previews"].items():
            lines.append(f"### `{col}`")
            preview = values[:25]
            for item in preview:
                lines.append(f"- `{item}`")
            lines.append("")

    lines.append("## Paper-Facing Note")
    lines.append("")
    lines.append(
        "This Day 44 interface is designed for reproducible qualitative inspection. "
        "It exposes the score components and available counterfactual evidence without inventing missing values. "
        "In the paper, these outputs can support case-study examples, explanation audits, and demo screenshots."
    )
    lines.append("")

    return "\n".join(lines)


def list_records(df: pd.DataFrame, top_k: int) -> str:
    available_ids = [c for c in ID_COLUMNS if c in df.columns]
    score_col = first_existing_column(
        df,
        SCORE_ALIASES["scal"] + SCORE_ALIASES["sdet"] + SCORE_ALIASES["sont"],
    )

    view_cols = []
    view_cols.extend(available_ids[:2])
    for col in ["label", "anomaly_type", "anomaly_family", "variant"]:
        if col in df.columns and col not in view_cols:
            view_cols.append(col)
    if score_col and score_col not in view_cols:
        view_cols.append(score_col)

    if not view_cols:
        return df.head(top_k).to_string(index=True)

    temp = df.copy()
    if score_col:
        temp["_sort_score"] = pd.to_numeric(temp[score_col], errors="coerce")
        temp = temp.sort_values("_sort_score", ascending=False, na_position="last")
    return temp[view_cols].head(top_k).to_string(index=True)


def write_outputs(expl: dict[str, Any], out_dir: Path, title: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in expl["record_identifier"])
    prefix = f"day44_explanation_{safe_id}"

    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"

    json_path.write_text(json.dumps(expl, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_from_explanation(expl, title), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()

    source_path = Path(args.scores_csv) if args.scores_csv else find_default_scores_path()
    df = load_table(source_path)

    if args.list:
        print(f"[Day 44] Source artifact: {source_path}")
        print(f"[Day 44] Rows: {len(df)}")
        print(list_records(df, args.top_k))
        return

    row_idx, row = select_row(df, args.record_id, args.row)
    expl = build_explanation(row, row_idx, source_path)
    json_path, md_path = write_outputs(expl, Path(args.out_dir), args.title)

    print(markdown_from_explanation(expl, args.title))
    print(f"\n[Day 44] Wrote JSON: {json_path}")
    print(f"[Day 44] Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()


