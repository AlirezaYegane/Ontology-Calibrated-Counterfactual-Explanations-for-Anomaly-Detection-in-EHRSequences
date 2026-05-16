from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.explanation.text_generator import build_explanation_batch, summarize_explanations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Day 38 text explanations from Day 37 counterfactual outputs.")
    parser.add_argument("--input_path", default=None, help="Day 37 counterfactual/evaluation artifact: csv/json/jsonl/pkl/parquet.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_cases", type=int, default=50)
    parser.add_argument("--sgen_policy", choices=["diagnostic_only", "full_signal"], default="diagnostic_only")
    parser.add_argument("--demo", action="store_true", help="Run on built-in demo rows for smoke testing.")
    return parser.parse_args()


def _demo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "demo_demographic_001",
                "anomaly_type": "demographic_conflict",
                "s_det": 0.91,
                "s_gen": 0.12,
                "s_ont": 1.00,
                "scal_before": 0.94,
                "scal_after": 0.18,
                "edit_count": 1,
                "violations": ["sex-incompatible code detected"],
                "action": ["remove incompatible pregnancy-related code"],
            },
            {
                "record_id": "demo_medication_001",
                "anomaly_type": "medication_mismatch",
                "s_det": 0.68,
                "s_gen": 0.08,
                "s_ont": 0.70,
                "scal_before": 0.73,
                "scal_after": 0.31,
                "edit_count": 1,
                "violations": ["medication appears without a compatible indication"],
                "action": ["add compatible diagnosis / indication code"],
            },
            {
                "record_id": "demo_missing_dx_001",
                "anomaly_type": "missing_diagnosis",
                "s_det": 0.38,
                "s_gen": 0.10,
                "s_ont": 0.35,
                "scal_before": 0.42,
                "scal_after": 0.34,
                "edit_count": 1,
                "violations": ["expected diagnosis not present"],
                "action": ["add expected diagnosis code"],
            },
        ]
    )


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        obj: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            for key in ("rows", "records", "cases", "results", "explanations", "counterfactuals"):
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

    raise ValueError(f"Unsupported input format: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_case_studies(explanations: list[dict[str, Any]], max_cases: int) -> str:
    lines: list[str] = []
    lines.append("# Day 38 — Explanation Case Studies")
    lines.append("")
    lines.append("These examples are generated from counterfactual evaluation outputs. They are intended for research inspection and paper-oriented qualitative analysis.")
    lines.append("")
    lines.append("> Important: explanations are data-quality / model-explanation statements, not clinical recommendations.")
    lines.append("")

    for idx, item in enumerate(explanations[:max_cases], start=1):
        lines.append(f"## Case {idx}: `{item['record_id']}`")
        lines.append("")
        lines.append(f"- Anomaly type: `{item['anomaly_type']}`")
        lines.append(f"- Primary driver: `{item['primary_driver']}`")
        lines.append(f"- Edit count: `{item['edit_count']}`")
        lines.append(f"- Score change: `{item['scal_before']:.4f}` → `{item['scal_after']:.4f}`")
        lines.append(f"- ΔScal: `{item['delta_scal']:.4f}`")
        lines.append("")
        lines.append("### Short explanation")
        lines.append(item["explanation_short"])
        lines.append("")
        lines.append("### Research explanation")
        lines.append(item["explanation_research"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        df = _demo_frame()
        input_path = "demo"
    else:
        if not args.input_path:
            raise SystemExit("Provide --input_path or use --demo for smoke testing.")
        input_path = str(args.input_path)
        df = read_table(args.input_path)

    if args.max_cases > 0:
        df = df.head(args.max_cases).copy()

    rows = df.to_dict(orient="records")
    explanations = build_explanation_batch(rows, sgen_policy=args.sgen_policy)
    summary = summarize_explanations(explanations)
    summary.update(
        {
            "day": 38,
            "status": "complete",
            "input_path": input_path,
            "sgen_policy": args.sgen_policy,
            "outputs": {
                "csv": str(out_dir / "day38_explanations.csv"),
                "jsonl": str(out_dir / "day38_explanations.jsonl"),
                "case_studies_md": str(out_dir / "day38_case_studies.md"),
                "summary_json": str(out_dir / "day38_explanation_summary.json"),
            },
            "paper_interpretation": (
                "Day 38 converts decomposed anomaly/counterfactual outputs into reproducible, "
                "template-based explanations. Sgen is treated conservatively when configured as diagnostic_only."
            ),
        }
    )

    pd.DataFrame(explanations).to_csv(out_dir / "day38_explanations.csv", index=False)
    write_jsonl(out_dir / "day38_explanations.jsonl", explanations)
    write_json(out_dir / "day38_explanation_summary.json", summary)
    (out_dir / "day38_case_studies.md").write_text(
        render_case_studies(explanations, max_cases=args.max_cases),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
