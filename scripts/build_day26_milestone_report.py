from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def first_existing_json(paths: list[Path]) -> tuple[Path | None, dict[str, Any] | None]:
    for path in paths:
        payload = load_json_if_exists(path)
        if payload is not None:
            return path, payload
    return None, None


def read_breakdown_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def infer_best_worst_from_breakdown(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    filtered = []
    for row in rows:
        anomaly_type = str(row.get("anomaly_type", "")).strip()
        if anomaly_type in {"", "normal"}:
            continue
        mean_score = safe_float(row.get("mean_score"))
        if mean_score is None:
            continue
        filtered.append((anomaly_type, mean_score))

    if not filtered:
        return None, None

    hardest = min(filtered, key=lambda x: x[1])[0]
    easiest = max(filtered, key=lambda x: x[1])[0]
    return hardest, easiest


def flatten_metrics(
    artifact_day20: dict[str, Any] | None,
    eval_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if artifact_day20:
        for key in (
            "best_epoch",
            "roc_auc",
            "average_precision",
            "threshold",
            "precision",
            "recall",
            "f1",
            "predicted_positive_rate",
            "hardest_anomaly_type",
            "easiest_anomaly_type",
            "notes",
        ):
            if key in artifact_day20:
                metrics[key] = artifact_day20[key]

    if eval_summary:
        cm = eval_summary.get("classification_metrics", {})
        rt = eval_summary.get("recommended_threshold", {})
        if "roc_auc" not in metrics and "roc_auc" in cm:
            metrics["roc_auc"] = cm["roc_auc"]
        if "average_precision" not in metrics and "average_precision" in cm:
            metrics["average_precision"] = cm["average_precision"]
        if "threshold" not in metrics and "threshold" in rt:
            metrics["threshold"] = rt["threshold"]
        if "precision" not in metrics and "precision" in rt:
            metrics["precision"] = rt["precision"]
        if "recall" not in metrics and "recall" in rt:
            metrics["recall"] = rt["recall"]
        if "f1" not in metrics and "f1" in rt:
            metrics["f1"] = rt["f1"]
        if "predicted_positive_rate" not in metrics and "predicted_positive_rate" in rt:
            metrics["predicted_positive_rate"] = rt["predicted_positive_rate"]

    return metrics


def profile_from_metrics(metrics: dict[str, Any]) -> str:
    precision = safe_float(metrics.get("precision"))
    recall = safe_float(metrics.get("recall"))
    if precision is None or recall is None:
        return "unknown"
    if precision > recall:
        return "high_precision_moderate_recall"
    if recall > precision:
        return "high_recall_lower_precision"
    return "balanced"


def list_files_if_exists(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_file()])


def build_report() -> dict[str, Any]:
    artifact_day20_path, artifact_day20 = first_existing_json([
        ROOT / "artifacts" / "day20" / "day20_supervised_eval_summary.json",
    ])

    eval_summary_path, eval_summary = first_existing_json([
        ROOT / "outputs" / "detector_eval" / "day20_supervised" / "run_luxury" / "summary.json",
    ])

    breakdown_path = ROOT / "outputs" / "detector_eval" / "day20_supervised" / "run_luxury" / "anomaly_type_breakdown.csv"
    breakdown_rows = read_breakdown_csv(breakdown_path)

    day25_dir = ROOT / "outputs" / "scoring" / "day25" / "run_ref_v2"
    day25_files = list_files_if_exists(day25_dir)

    metrics = flatten_metrics(artifact_day20, eval_summary)

    hardest_from_breakdown, easiest_from_breakdown = infer_best_worst_from_breakdown(breakdown_rows)

    if "hardest_anomaly_type" not in metrics and hardest_from_breakdown:
        metrics["hardest_anomaly_type"] = hardest_from_breakdown
    if "easiest_anomaly_type" not in metrics and easiest_from_breakdown:
        metrics["easiest_anomaly_type"] = easiest_from_breakdown

    strengths: list[str] = []
    limitations: list[str] = []
    implications: list[str] = []

    roc_auc = safe_float(metrics.get("roc_auc"))
    ap = safe_float(metrics.get("average_precision"))
    f1 = safe_float(metrics.get("f1"))
    precision = safe_float(metrics.get("precision"))
    recall = safe_float(metrics.get("recall"))
    threshold = safe_float(metrics.get("threshold"))

    if roc_auc is not None:
        strengths.append(f"Detector achieved ROC-AUC={roc_auc:.4f} on the supervised anomaly evaluation set.")
    if ap is not None:
        strengths.append(f"Average Precision reached {ap:.4f}, indicating meaningful anomaly ranking quality.")
    if f1 is not None:
        strengths.append(f"Operational thresholding produced F1={f1:.4f}.")
    if metrics.get("easiest_anomaly_type"):
        strengths.append(f"Strongest anomaly family at this stage: {metrics['easiest_anomaly_type']}.")
    if profile_from_metrics(metrics) == "high_precision_moderate_recall":
        strengths.append("Current operating point is conservative: precision is stronger than recall.")

    if metrics.get("hardest_anomaly_type"):
        limitations.append(f"Hardest anomaly family remains: {metrics['hardest_anomaly_type']}.")
    if recall is not None and recall < 0.60:
        limitations.append("Recall is still moderate, so some anomalies are likely being missed.")
    limitations.append("This milestone is still pre-diffusion: generative surprise (Sgen) is not yet the main driver of the score.")
    limitations.append("Baseline detector behavior is now understood well enough to serve as a stable pre-generative reference.")

    implications.append("Freeze this detector behavior as the baseline reference before Day 27 data preparation.")
    implications.append("Prioritize the hardest anomaly family during generative design and later counterfactual repair analysis.")
    implications.append("Carry current threshold and anomaly-type breakdown forward as the benchmark for post-diffusion comparison.")
    implications.append("Use Day 27 to prepare tensors/dataloaders cleanly; do not change baseline evaluation definitions during that transition.")

    evidence_paths = {
        "artifact_day20_summary": str(artifact_day20_path) if artifact_day20_path else None,
        "day20_eval_summary": str(eval_summary_path) if eval_summary_path else None,
        "day20_breakdown_csv": str(breakdown_path) if breakdown_path.exists() else None,
        "day25_scoring_dir": str(day25_dir) if day25_dir.exists() else None,
    }

    return {
        "generated_at": now_utc(),
        "milestone": "Day 26 - Baseline Completion",
        "objective": "Summarize final baseline detector performance, strengths, limitations, and implications for the generative stage.",
        "status": "complete",
        "baseline_ready_for_generative_stage": True,
        "metrics": metrics,
        "operating_profile": profile_from_metrics(metrics),
        "strengths": strengths,
        "limitations": limitations,
        "implications_for_generative_stage": implications,
        "supporting_evidence_paths": evidence_paths,
        "day25_scoring_files_detected": day25_files,
        "checkpoint_note": (
            "Day 26 closes Milestone 2. Day 27 should begin from stable, documented baseline behavior rather than ad-hoc memory."
        ),
        "next_step": "Day 27 - prepare diffusion inputs, masks, batching, and storage artifacts."
    }


def render_markdown(report: dict[str, Any]) -> str:
    m = report.get("metrics", {})

    def fmt(key: str) -> str:
        value = m.get(key)
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value) if value is not None else "n/a"

    lines: list[str] = []
    lines.append("# Day 26 - Baseline Completion")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Status: **{report['status']}**")
    lines.append(f"- Baseline ready for generative stage: **{report['baseline_ready_for_generative_stage']}**")
    lines.append(f"- Operating profile: **{report['operating_profile']}**")
    lines.append("")

    lines.append("## Final Metrics")
    lines.append("")
    lines.append(f"- ROC-AUC: **{fmt('roc_auc')}**")
    lines.append(f"- Average Precision: **{fmt('average_precision')}**")
    lines.append(f"- F1: **{fmt('f1')}**")
    lines.append(f"- Precision: **{fmt('precision')}**")
    lines.append(f"- Recall: **{fmt('recall')}**")
    lines.append(f"- Threshold: **{fmt('threshold')}**")
    lines.append(f"- Predicted positive rate: **{fmt('predicted_positive_rate')}**")
    lines.append(f"- Hardest anomaly type: **{fmt('hardest_anomaly_type')}**")
    lines.append(f"- Easiest anomaly type: **{fmt('easiest_anomaly_type')}**")
    lines.append("")

    lines.append("## Strengths")
    lines.append("")
    for item in report["strengths"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Implications for Generative Stage")
    lines.append("")
    for item in report["implications_for_generative_stage"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Supporting Evidence Paths")
    lines.append("")
    for key, value in report["supporting_evidence_paths"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    lines.append("## Day 25 Scoring Files Detected")
    lines.append("")
    if report["day25_scoring_files_detected"]:
        for name in report["day25_scoring_files_detected"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- None detected")
    lines.append("")

    lines.append("## Next Step")
    lines.append("")
    lines.append(f"- {report['next_step']}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = build_report()

    out_json = ROOT / "artifacts" / "day26" / "day26_baseline_milestone.json"
    out_md = ROOT / "artifacts" / "day26" / "day26_baseline_milestone.md"

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "json": str(out_json),
        "markdown": str(out_md),
        "status": report["status"],
        "next_step": report["next_step"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

