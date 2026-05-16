from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.explanations.counterfactual import (
    anomaly_type_from_row,
    compact_json,
    generate_counterfactual,
    result_to_dict,
    tokens_from_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_scores", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=1000)
    parser.add_argument("--edit_penalty", type=float, default=0.25)
    parser.add_argument("--max_edits", type=int, default=2)
    parser.add_argument("--only_anomalies", action="store_true")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return None
        return float(text)
    except Exception:
        return None


def infer_label(row: dict[str, Any]) -> int | None:
    for key in ("label", "y_true", "is_anomaly", "is_synthetic_anomaly"):
        if key in row:
            value = safe_float(row.get(key))
            if value is not None:
                return int(value)
    return None


def infer_score(row: dict[str, Any]) -> float | None:
    candidates = (
        "s_cal",
        "S_cal",
        "Scal",
        "calibrated_score",
        "strict_score",
        "score",
        "prob_anomaly",
        "sont_score",
        "Sont",
        "s_ont",
    )
    for key in candidates:
        if key in row:
            value = safe_float(row.get(key))
            if value is not None:
                return value
    return None


def should_process(row: dict[str, Any], only_anomalies: bool) -> bool:
    if not only_anomalies:
        return True

    label = infer_label(row)
    if label == 1:
        return True

    anomaly_type = anomaly_type_from_row(row)
    if anomaly_type and anomaly_type not in {"normal", "none", "nan"}:
        return True

    score = infer_score(row)
    return bool(score is not None and score > 0)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Day 36 — Ontology-Constrained Counterfactual Generator")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- Status: **{summary['status']}**")
    lines.append(f"- Input scores: `{summary['input_scores']}`")
    lines.append(f"- Rows loaded: **{summary['rows_loaded']}**")
    lines.append(f"- Rows processed: **{summary['rows_processed']}**")
    lines.append("")
    lines.append("## Counterfactual results")
    lines.append("")
    lines.append(f"- Improved records: **{summary['improved_records']}**")
    lines.append(f"- No-candidate records: **{summary['no_candidate_records']}**")
    lines.append(f"- Not-improved records: **{summary['not_improved_records']}**")
    lines.append(f"- Mean edit count: **{summary['mean_edit_count']:.4f}**")
    lines.append(
        f"- Mean violation-score reduction: **{summary['mean_delta_violation_score']:.4f}**"
    )
    lines.append(
        f"- One-edit success rate among improved: **{summary['one_edit_success_rate']:.4f}**"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Day 36 implements the first working counterfactual generator. "
        "The generator uses ontology-constrained add/remove/replace edits and selects "
        "minimal edits that reduce the rule-level ontology violation score."
    )
    lines.append("")
    lines.append(
        "This is intentionally conservative: it does not claim that diffusion Sgen is "
        "the main explanation signal. The current generator is designed around Sont-style "
        "ontology consistency and is ready for Day 37 evaluation."
    )
    lines.append("")
    lines.append("## Generated artifacts")
    lines.append("")
    for key, value in summary["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    rows: list[dict[str, Any]] = []

    for idx, row_obj in df.iterrows():
        row = {str(k): v for k, v in row_obj.to_dict().items()}
        if not should_process(row, args.only_anomalies):
            continue

        codes = tokens_from_row(row)
        if not codes:
            continue

        result = generate_counterfactual(
            codes=codes,
            row=row,
            edit_penalty=args.edit_penalty,
            max_edits=args.max_edits,
        )
        payload = result_to_dict(result)

        rows.append(
            {
                "row_index": int(idx),
                "label": infer_label(row),
                "anomaly_type": anomaly_type_from_row(row),
                "input_score": infer_score(row),
                "status": payload["status"],
                "edit_count": payload["edit_count"],
                "edits_text": payload["edits_text"],
                "original_violation_score": payload["original_violation_score"],
                "counterfactual_violation_score": payload[
                    "counterfactual_violation_score"
                ],
                "delta_violation_score": payload["delta_violation_score"],
                "resolved_violation_count": payload["resolved_violation_count"],
                "original_code_count": len(payload["original_codes"]),
                "counterfactual_code_count": len(payload["counterfactual_codes"]),
                "edits_json": compact_json(payload["edits"]),
                "original_violations_json": compact_json(
                    payload["original_violations"]
                ),
                "counterfactual_violations_json": compact_json(
                    payload["counterfactual_violations"]
                ),
                "original_codes_json": compact_json(payload["original_codes"]),
                "counterfactual_codes_json": compact_json(
                    payload["counterfactual_codes"]
                ),
                "explanation": payload["explanation"],
            }
        )

        if len(rows) >= args.max_records:
            break

    out_df = pd.DataFrame(rows)
    counterfactuals_path = out_dir / "counterfactuals.csv"
    out_df.to_csv(counterfactuals_path, index=False)

    if len(out_df) == 0:
        summary = {
            "status": "complete_no_records_processed",
            "input_scores": str(input_path),
            "rows_loaded": int(len(df)),
            "rows_processed": 0,
            "improved_records": 0,
            "no_candidate_records": 0,
            "not_improved_records": 0,
            "mean_edit_count": 0.0,
            "mean_delta_violation_score": 0.0,
            "one_edit_success_rate": 0.0,
            "artifacts": {
                "counterfactuals_csv": str(counterfactuals_path),
            },
        }
    else:
        improved = out_df[out_df["status"] == "improved"]
        summary = {
            "status": "complete",
            "input_scores": str(input_path),
            "rows_loaded": int(len(df)),
            "rows_processed": int(len(out_df)),
            "improved_records": int((out_df["status"] == "improved").sum()),
            "no_candidate_records": int((out_df["status"] == "no_candidate").sum()),
            "not_improved_records": int((out_df["status"] == "not_improved").sum()),
            "mean_edit_count": float(out_df["edit_count"].mean()),
            "mean_delta_violation_score": float(out_df["delta_violation_score"].mean()),
            "one_edit_success_rate": float((improved["edit_count"] == 1).mean())
            if len(improved)
            else 0.0,
            "status_counts": {
                str(k): int(v)
                for k, v in out_df["status"].value_counts().to_dict().items()
            },
            "anomaly_type_status_counts": {
                str(k): {str(kk): int(vv) for kk, vv in value.items()}
                for k, value in out_df.groupby("anomaly_type")["status"]
                .value_counts()
                .unstack(fill_value=0)
                .to_dict(orient="index")
                .items()
            },
            "artifacts": {
                "counterfactuals_csv": str(counterfactuals_path),
                "summary_json": str(out_dir / "summary.json"),
                "readme_md": str(out_dir / "README.md"),
            },
        }

    write_json(out_dir / "summary.json", summary)
    (out_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
