from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYWORDS = (
    "summary",
    "scores",
    "breakdown",
    "threshold",
    "review",
    "plausibility",
    "case",
    "assessment",
    "ablation",
    "variant",
    "counterfactual",
    "explanation",
    "audit",
)

SCORE_COLS = (
    "prob_anomaly",
    "calibrated_score",
    "s_cal",
    "S_cal",
    "scal",
    "score",
    "anomaly_score",
    "detector_only",
    "full_model",
    "full_model_conservative",
    "no_generative",
    "no_ontology",
    "Sgen",
    "sgen",
    "Sont",
    "sont",
)

TYPE_COLS = (
    "anomaly_type",
    "type",
    "family",
    "category",
    "violation_type",
)

EDIT_COLS = (
    "edit_count",
    "n_edits",
    "num_edits",
    "edits",
    "num_operations",
)

BEFORE_COLS = (
    "score_before",
    "s_cal_before",
    "S_cal_before",
    "original_score",
    "before_score",
    "scal_before",
)

AFTER_COLS = (
    "score_after",
    "s_cal_after",
    "S_cal_after",
    "counterfactual_score",
    "after_score",
    "scal_after",
)

ISSUE_COUNT_COLS = (
    "violation_count",
    "n_violations",
    "num_violations",
    "issue_count",
    "n_issues",
    "num_issues",
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def first_existing_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_evidence_inventory(artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not artifacts_dir.exists():
        return pd.DataFrame(rows)

    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
            continue

        name_l = path.name.lower()
        rel = path.as_posix()
        relevant = any(k in name_l or k in rel.lower() for k in KEYWORDS)
        if not relevant:
            continue

        row: dict[str, Any] = {
            "path": path.as_posix(),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "evidence_type": "unknown",
            "columns_or_keys": "",
        }

        if path.suffix.lower() == ".json":
            payload = read_json(path)
            if isinstance(payload, dict):
                row["evidence_type"] = "json"
                row["columns_or_keys"] = ", ".join(list(payload.keys())[:30])
        elif path.suffix.lower() == ".csv":
            df = read_csv(path)
            if df is not None:
                row["evidence_type"] = "csv"
                row["rows"] = int(len(df))
                row["columns_or_keys"] = ", ".join(list(df.columns)[:30])
        else:
            row["evidence_type"] = "text"

        rows.append(row)

    return pd.DataFrame(rows)


def add_risk(
    risks: list[dict[str, Any]],
    risk_id: str,
    failure_mode: str,
    evidence_status: str,
    evidence: str,
    severity: str,
    likelihood: str,
    scientific_impact: str,
    mitigation: str,
    paper_wording: str,
) -> None:
    risks.append(
        {
            "id": risk_id,
            "failure_mode": failure_mode,
            "evidence_status": evidence_status,
            "evidence": evidence,
            "severity": severity,
            "likelihood": likelihood,
            "scientific_impact": scientific_impact,
            "mitigation": mitigation,
            "paper_wording": paper_wording,
        }
    )


def analyze_day34_sgen(artifacts_dir: Path, risks: list[dict[str, Any]]) -> None:
    candidates = list(artifacts_dir.rglob("*day34*assessment*.json"))
    candidates += list(artifacts_dir.rglob("day34_final_assessment.json"))

    seen = set()
    candidates = [p for p in candidates if not (p.as_posix() in seen or seen.add(p.as_posix()))]

    for path in candidates:
        payload = read_json(path)
        if not payload:
            continue

        result = payload.get("main_sgen_result", {})
        auc = safe_float(result.get("best_roc_auc"))
        ap = safe_float(result.get("best_average_precision"))
        timestep = result.get("best_timestep")

        if auc is not None:
            ap_text = f"{ap:.4f}" if ap is not None else "n/a"
            evidence = (
                f"Found {path.as_posix()}; best timestep={timestep}, "
                f"Sgen ROC-AUC={auc:.4f}, AP={ap_text}."
            )

            add_risk(
                risks,
                "FM-03",
                "Weak diffusion-based generative surprise signal",
                "measured",
                evidence,
                "high",
                "high",
                "Raw Sgen should not be overclaimed as a standalone anomaly detector.",
                "Keep Sgen as an auxiliary/diagnostic term; report detector and ontology-calibrated scores separately; use conservative weighting for w_gen.",
                "The diffusion-derived surprise proxy was retained as an auxiliary signal because its standalone discrimination was weak in the current benchmark.",
            )
            return

    add_risk(
        risks,
        "FM-03",
        "Weak diffusion-based generative surprise signal",
        "not directly measured in available artifacts",
        "No Day 34 final Sgen assessment JSON was found by the scanner.",
        "medium",
        "unknown",
        "Cannot quantify the reliability of Sgen from the available evidence snapshot.",
        "Run or attach the Day 34 Sgen timestep sweep before making claims about generative surprise.",
        "Generative surprise should be interpreted cautiously unless validated by a dedicated separation analysis.",
    )


def analyze_anomaly_family_scores(artifacts_dir: Path, risks: list[dict[str, Any]]) -> None:
    best_evidence: dict[str, Any] | None = None

    invalid_type_values = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "normal",
        "0",
        "false",
    }

    for path in sorted(artifacts_dir.rglob("*.csv")):
        df = read_csv(path)
        if df is None or df.empty:
            continue

        cols = list(df.columns)
        type_col = first_existing_col(cols, TYPE_COLS)
        score_col = first_existing_col(cols, SCORE_COLS)

        if type_col is None or score_col is None:
            continue

        temp = df[[type_col, score_col]].copy()
        temp[score_col] = pd.to_numeric(temp[score_col], errors="coerce")
        temp[type_col] = temp[type_col].astype("string")

        temp = temp.dropna(subset=[type_col, score_col])
        temp[type_col] = temp[type_col].str.strip()

        temp = temp[
            ~temp[type_col].str.lower().isin(invalid_type_values)
        ].copy()

        if temp.empty:
            continue

        grouped = (
            temp.groupby(type_col, dropna=False)[score_col]
            .agg(["count", "mean", "median"])
            .reset_index()
            .sort_values("mean", ascending=True)
        )

        if grouped.empty:
            continue

        hardest = grouped.iloc[0].to_dict()
        best_evidence = {
            "path": path.as_posix(),
            "type_col": type_col,
            "score_col": score_col,
            "hardest_type": str(hardest[type_col]),
            "hardest_mean": float(hardest["mean"]),
            "hardest_median": float(hardest["median"]),
            "hardest_count": int(hardest["count"]),
            "table": grouped.to_dict(orient="records"),
        }
        break

    if best_evidence:
        add_risk(
            risks,
            "FM-02",
            "Uneven anomaly-family separation",
            "measured",
            (
                f"From {best_evidence['path']}, hardest non-empty anomaly family by mean "
                f"{best_evidence['score_col']} is {best_evidence['hardest_type']} "
                f"(mean={best_evidence['hardest_mean']:.4f}, "
                f"median={best_evidence['hardest_median']:.4f}, "
                f"n={best_evidence['hardest_count']})."
            ),
            "high",
            "high",
            "A single headline AUC may hide weaker performance on clinically important anomaly subtypes.",
            "Report per-family breakdowns; tune thresholds per anomaly family if justified; add targeted rules/examples for weak families such as missing-diagnosis cases.",
            "Performance was not uniform across anomaly categories; family-level breakdowns were therefore used to identify the main residual error modes.",
        )
    else:
        add_risk(
            risks,
            "FM-02",
            "Uneven anomaly-family separation",
            "not directly measured in available artifacts",
            "No CSV with non-empty anomaly family/type and score columns was found.",
            "medium",
            "unknown",
            "Cannot verify whether specific anomaly types are under-detected.",
            "Ensure evaluation exports anomaly_type_breakdown.csv or equivalent family-level score tables.",
            "Future evaluation should report anomaly-family-specific performance rather than only aggregate metrics.",
        )


def analyze_threshold_sensitivity(artifacts_dir: Path, risks: list[dict[str, Any]]) -> None:
    best: dict[str, Any] | None = None

    for path in sorted(artifacts_dir.rglob("*threshold*sweep*.csv")):
        df = read_csv(path)
        if df is None or df.empty:
            continue

        required = {"threshold", "precision", "recall", "f1"}
        if not required.issubset({c.lower() for c in df.columns}):
            continue

        colmap = {c.lower(): c for c in df.columns}
        threshold_col = colmap["threshold"]
        precision_col = colmap["precision"]
        recall_col = colmap["recall"]
        f1_col = colmap["f1"]

        for c in [threshold_col, precision_col, recall_col, f1_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=[threshold_col, precision_col, recall_col, f1_col])
        if df.empty:
            continue

        best_f1 = df.loc[df[f1_col].idxmax()]
        high_precision = df[df[precision_col] >= 0.90]
        if not high_precision.empty:
            high_precision_row = high_precision.loc[high_precision[recall_col].idxmax()]
        else:
            high_precision_row = None

        best = {
            "path": path.as_posix(),
            "best_f1_threshold": float(best_f1[threshold_col]),
            "best_f1_precision": float(best_f1[precision_col]),
            "best_f1_recall": float(best_f1[recall_col]),
            "best_f1": float(best_f1[f1_col]),
            "high_precision_threshold": float(high_precision_row[threshold_col]) if high_precision_row is not None else None,
            "high_precision_precision": float(high_precision_row[precision_col]) if high_precision_row is not None else None,
            "high_precision_recall": float(high_precision_row[recall_col]) if high_precision_row is not None else None,
            "high_precision_f1": float(high_precision_row[f1_col]) if high_precision_row is not None else None,
        }
        break

    if best:
        hp_text = "No threshold with precision >= 0.90 was found."
        if best["high_precision_threshold"] is not None:
            hp_text = (
                f"At precision>=0.90, best recall row: threshold={best['high_precision_threshold']:.4f}, "
                f"precision={best['high_precision_precision']:.4f}, "
                f"recall={best['high_precision_recall']:.4f}, "
                f"F1={best['high_precision_f1']:.4f}."
            )

        add_risk(
            risks,
            "FM-04",
            "Threshold sensitivity and precision-recall trade-off",
            "measured",
            (
                f"From {best['path']}, best-F1 threshold={best['best_f1_threshold']:.4f}, "
                f"precision={best['best_f1_precision']:.4f}, recall={best['best_f1_recall']:.4f}, "
                f"F1={best['best_f1']:.4f}. {hp_text}"
            ),
            "medium",
            "high",
            "A single threshold can make the system look either conservative or sensitive; claims must specify operating point.",
            "Report threshold sweeps; choose a conservative paper threshold for precision and discuss recall trade-off explicitly.",
            "The calibrated score was evaluated across thresholds because the intended use is review prioritization rather than autonomous diagnosis.",
        )
    else:
        add_risk(
            risks,
            "FM-04",
            "Threshold sensitivity and precision-recall trade-off",
            "not directly measured in available artifacts",
            "No threshold_sweep CSV was found.",
            "medium",
            "unknown",
            "Cannot justify the selected operating threshold from available evidence.",
            "Export threshold_sweep.csv for every final evaluation run.",
            "Threshold-dependent metrics should be reported alongside threshold-free metrics such as ROC-AUC and average precision.",
        )


def analyze_mapping_audit(artifacts_dir: Path, risks: list[dict[str, Any]]) -> None:
    candidates = list(artifacts_dir.rglob("mapping_audit.json"))
    candidates += list(artifacts_dir.rglob("*mapping*audit*.json"))

    seen = set()
    candidates = [p for p in candidates if not (p.as_posix() in seen or seen.add(p.as_posix()))]

    for path in candidates:
        payload = read_json(path)
        if not payload:
            continue

        mapping_counts = payload.get("mapping_counts", {})
        edge_case_counts = payload.get("edge_case_counts", {})
        warnings = payload.get("warnings", [])
        critical = payload.get("critical_issues", [])

        evidence_parts = [f"Found mapping audit: {path.as_posix()}."]
        if mapping_counts:
            evidence_parts.append(f"mapping_counts={mapping_counts}")
        if edge_case_counts:
            evidence_parts.append(f"edge_case_counts={edge_case_counts}")
        if warnings:
            evidence_parts.append(f"warnings={len(warnings)}")
        if critical:
            evidence_parts.append(f"critical_issues={len(critical)}")

        add_risk(
            risks,
            "FM-01",
            "Ambiguous or incomplete ontology mappings",
            "measured",
            " ".join(evidence_parts),
            "high",
            "high",
            "Ontology-based scores can be biased if mapping coverage is uneven across diagnoses, procedures, or medications.",
            "Report mapping coverage; compute Sont only for covered rule families; keep unmapped/ambiguous codes explicit rather than silently repairing them.",
            "Ontology-derived conclusions were restricted to code families with available mappings and rules; unmapped or ambiguous concepts were tracked as a coverage limitation.",
        )
        return

    add_risk(
        risks,
        "FM-01",
        "Ambiguous or incomplete ontology mappings",
        "not directly measured in available artifacts",
        "No mapping_audit.json was found in artifacts.",
        "high",
        "medium",
        "Sont validity depends on coverage of ICD/SNOMED/RxNorm mappings.",
        "Run or attach the mapping audit; add coverage table to the paper appendix.",
        "Ontology coverage should be reported explicitly because unmapped codes limit the scope of rule-based anomaly claims.",
    )


def analyze_counterfactuals(artifacts_dir: Path, risks: list[dict[str, Any]]) -> None:
    best: dict[str, Any] | None = None

    for path in sorted(artifacts_dir.rglob("*.csv")):
        name_l = path.name.lower()
        if not any(k in name_l for k in ("counterfactual", "explanation", "case", "review")):
            continue

        df = read_csv(path)
        if df is None or df.empty:
            continue

        cols = list(df.columns)
        edit_col = first_existing_col(cols, EDIT_COLS)
        before_col = first_existing_col(cols, BEFORE_COLS)
        after_col = first_existing_col(cols, AFTER_COLS)

        if edit_col is None and not (before_col and after_col):
            continue

        metrics: dict[str, Any] = {"path": path.as_posix(), "rows": int(len(df))}

        if edit_col is not None:
            edit_values = pd.to_numeric(df[edit_col], errors="coerce").dropna()
            if not edit_values.empty:
                metrics["mean_edits"] = float(edit_values.mean())
                metrics["pct_more_than_two_edits"] = float((edit_values > 2).mean())

        if before_col and after_col:
            before = pd.to_numeric(df[before_col], errors="coerce")
            after = pd.to_numeric(df[after_col], errors="coerce")
            delta = before - after
            delta = delta.dropna()
            if not delta.empty:
                metrics["median_score_drop"] = float(delta.median())
                metrics["mean_score_drop"] = float(delta.mean())
                metrics["pct_no_improvement"] = float((delta <= 0).mean())

        best = metrics
        break

    if best:
        evidence = f"Found counterfactual/explanation evidence in {best['path']}; rows={best['rows']}."
        if "mean_edits" in best:
            evidence += f" mean_edits={best['mean_edits']:.3f}; pct_more_than_two_edits={best['pct_more_than_two_edits']:.3f}."
        if "median_score_drop" in best:
            evidence += f" median_score_drop={best['median_score_drop']:.4f}; pct_no_improvement={best['pct_no_improvement']:.3f}."

        add_risk(
            risks,
            "FM-05",
            "Unstable or non-minimal counterfactuals",
            "measured",
            evidence,
            "medium",
            "medium",
            "Explanations become less interpretable if they require many edits or fail to reduce the calibrated score.",
            "Prefer one- or two-edit candidates; flag no-improvement cases; report edit-count distribution and score reduction.",
            "Counterfactual quality was assessed using both score reduction and sparsity because clinically useful explanations should remain minimal.",
        )
    else:
        add_risk(
            risks,
            "FM-05",
            "Unstable or non-minimal counterfactuals",
            "not directly measured in available artifacts",
            "No CSV with edit-count or before/after score columns was found.",
            "medium",
            "unknown",
            "Cannot quantify whether explanations are sparse and score-reducing.",
            "Ensure counterfactual evaluation exports edit_count, score_before, and score_after columns.",
            "Counterfactual sparsity and score reduction should be reported before making claims about explanation usefulness.",
        )


def analyze_multiple_issues(artifacts_dir: Path, risks: list[dict[str, Any]]) -> None:
    best: dict[str, Any] | None = None

    for path in sorted(artifacts_dir.rglob("*.csv")):
        df = read_csv(path)
        if df is None or df.empty:
            continue

        issue_col = first_existing_col(list(df.columns), ISSUE_COUNT_COLS)
        if issue_col is None:
            continue

        values = pd.to_numeric(df[issue_col], errors="coerce").dropna()
        if values.empty:
            continue

        best = {
            "path": path.as_posix(),
            "issue_col": issue_col,
            "rows": int(len(values)),
            "mean_issues": float(values.mean()),
            "pct_multi_issue": float((values > 1).mean()),
            "max_issues": float(values.max()),
        }
        break

    if best:
        add_risk(
            risks,
            "FM-06",
            "Records with multiple simultaneous issues",
            "measured",
            (
                f"From {best['path']}, column={best['issue_col']}, rows={best['rows']}, "
                f"mean_issues={best['mean_issues']:.3f}, pct_multi_issue={best['pct_multi_issue']:.3f}, "
                f"max_issues={best['max_issues']:.0f}."
            ),
            "medium",
            "medium",
            "Single-edit explanations may under-explain records that contain several independent issues.",
            "Separate single-issue and multi-issue cases in evaluation; use staged counterfactual search for multi-issue records.",
            "Multi-issue records were treated as a distinct failure mode because a minimal edit may only partially reduce the anomaly score.",
        )
    else:
        add_risk(
            risks,
            "FM-06",
            "Records with multiple simultaneous issues",
            "not directly measured in available artifacts",
            "No issue-count or violation-count column was found.",
            "medium",
            "unknown",
            "The current evidence may not distinguish simple anomalies from compound anomalies.",
            "Add violation_count / issue_count to exported explanation outputs.",
            "Future exports should track the number of simultaneous violations so that compound cases can be evaluated separately.",
        )


def analyze_plausibility_review(artifacts_dir: Path, risks: list[dict[str, Any]]) -> None:
    label_cols = (
        "plausibility_label",
        "review_label",
        "judgement",
        "judgment",
        "decision",
        "status",
        "rating",
    )

    for path in sorted(artifacts_dir.rglob("*.csv")):
        name_l = path.as_posix().lower()
        if not any(k in name_l for k in ("day46", "plausibility", "review", "explanation")):
            continue

        df = read_csv(path)
        if df is None or df.empty:
            continue

        label_col = first_existing_col(list(df.columns), label_cols)
        if label_col is None:
            continue

        counts = Counter(df[label_col].astype(str).str.lower().fillna("unknown"))
        unclear_like = sum(v for k, v in counts.items() if any(x in k for x in ("unclear", "bad", "fail", "misleading", "incorrect")))
        total = sum(counts.values())

        add_risk(
            risks,
            "FM-08",
            "Explanation overclaiming or unclear wording",
            "measured",
            f"Found review labels in {path.as_posix()}; label_counts={dict(counts)}; unclear_or_negative_rate={unclear_like / max(total, 1):.3f}.",
            "high",
            "medium",
            "Poor wording can make hypothetical edits look like clinical advice.",
            "Use cautious language; state that edits are hypothetical consistency checks, not treatment recommendations; keep clinical claims limited to ontology consistency.",
            "Explanations were framed as hypothetical consistency edits rather than clinical recommendations.",
        )
        return

    add_risk(
        risks,
        "FM-08",
        "Explanation overclaiming or unclear wording",
        "not directly measured in available artifacts",
        "No Day 46 plausibility review CSV with review labels was found.",
        "high",
        "medium",
        "Generated explanations may be misread as clinical advice if not carefully phrased.",
        "Keep template disclaimers; perform manual review on selected examples; separate documentation/coding suggestions from treatment claims.",
        "The interface and paper should explicitly state that counterfactual edits are explanatory, not prescriptive.",
    )


def add_static_research_risks(risks: list[dict[str, Any]]) -> None:
    existing = {r["id"] for r in risks}

    if "FM-07" not in existing:
        add_risk(
            risks,
            "FM-07",
            "Sensitivity to missing data and rare codes",
            "conceptual risk",
            "This risk follows from EHR sparsity, incomplete medication/diagnosis coverage, and rare but valid clinical combinations.",
            "medium",
            "high",
            "The system may flag rare but valid patient trajectories or miss anomalies when required companion codes are absent.",
            "Separate ontology violations from statistical rarity; report rare-code coverage; avoid treating high Sgen alone as an error.",
            "Rare but ontology-consistent records were treated as review candidates rather than automatic errors.",
        )

    if "FM-09" not in existing:
        add_risk(
            risks,
            "FM-09",
            "Bias or brittleness in demographic rules",
            "conceptual risk",
            "Demographic consistency rules can be useful but may become brittle if sex/age metadata or clinical context is incomplete.",
            "high",
            "medium",
            "Incorrect demographic assumptions can create false positives or inappropriate explanations.",
            "Make demographic rule sets explicit; log demographic attributes used; allow rule disabling and manual review.",
            "Demographic rules were implemented transparently and interpreted as data-consistency checks, not judgments about patient identity.",
        )


def build_markdown(risks: list[dict[str, Any]], inventory: pd.DataFrame) -> str:
    measured = sum(1 for r in risks if r["evidence_status"] == "measured")
    not_measured = sum(1 for r in risks if "not directly measured" in r["evidence_status"])
    conceptual = sum(1 for r in risks if r["evidence_status"] == "conceptual risk")

    lines: list[str] = []
    lines.append("# Day 47 — Risk and Failure Mode Analysis")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Complete.")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Systematically identify failure modes in the ontology-calibrated anomaly explanation pipeline and define mitigation strategies suitable for a research paper.")
    lines.append("")
    lines.append("## Why this matters for the paper")
    lines.append("")
    lines.append("A publishable system should not only report positive metrics. It should also explain when the method may fail, how those failures are detected, and what safeguards or conservative interpretations are used.")
    lines.append("")
    lines.append("## Evidence inventory summary")
    lines.append("")
    lines.append(f"- Evidence files scanned: {len(inventory)}")
    lines.append(f"- Measured failure modes: {measured}")
    lines.append(f"- Not directly measured from available artifacts: {not_measured}")
    lines.append(f"- Conceptual research risks: {conceptual}")
    lines.append("")
    lines.append("## Failure mode matrix")
    lines.append("")
    lines.append("| ID | Failure mode | Evidence status | Severity | Likelihood | Mitigation |")
    lines.append("|---|---|---|---|---|---|")
    for r in risks:
        lines.append(
            f"| {r['id']} | {r['failure_mode']} | {r['evidence_status']} | "
            f"{r['severity']} | {r['likelihood']} | {r['mitigation']} |"
        )
    lines.append("")
    lines.append("## Detailed evidence")
    lines.append("")
    for r in risks:
        lines.append(f"### {r['id']} — {r['failure_mode']}")
        lines.append("")
        lines.append(f"- Evidence status: {r['evidence_status']}")
        lines.append(f"- Evidence: {r['evidence']}")
        lines.append(f"- Scientific impact: {r['scientific_impact']}")
        lines.append(f"- Mitigation: {r['mitigation']}")
        lines.append("")
        lines.append("Paper-ready wording:")
        lines.append("")
        lines.append(f"> {r['paper_wording']}")
        lines.append("")
    lines.append("## Conservative interpretation policy")
    lines.append("")
    lines.append("- Do not present counterfactual edits as treatment recommendations.")
    lines.append("- Do not claim raw diffusion surprise is a strong standalone anomaly signal unless validated by separation metrics.")
    lines.append("- Report ontology coverage and mapping limitations explicitly.")
    lines.append("- Separate statistical rarity from ontology violation wherever possible.")
    lines.append("- Report per-family anomaly behavior rather than relying only on aggregate scores.")
    lines.append("")
    lines.append("## Day 47 conclusion")
    lines.append("")
    lines.append("The main contribution of Day 47 is a paper-ready risk register and mitigation matrix. This strengthens the scientific framing of the project by making limitations explicit, measurable where possible, and connected to concrete engineering safeguards.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--out_dir", default="artifacts/day47")
    parser.add_argument("--docs_dir", default="docs")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    docs_dir = Path(args.docs_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    risks: list[dict[str, Any]] = []
    inventory = collect_evidence_inventory(artifacts_dir)

    analyze_mapping_audit(artifacts_dir, risks)
    analyze_anomaly_family_scores(artifacts_dir, risks)
    analyze_day34_sgen(artifacts_dir, risks)
    analyze_threshold_sensitivity(artifacts_dir, risks)
    analyze_counterfactuals(artifacts_dir, risks)
    analyze_multiple_issues(artifacts_dir, risks)
    analyze_plausibility_review(artifacts_dir, risks)
    add_static_research_risks(risks)

    risks = sorted(risks, key=lambda r: r["id"])

    inventory.to_csv(out_dir / "evidence_inventory.csv", index=False)

    mitigation_df = pd.DataFrame(risks)
    mitigation_df.to_csv(out_dir / "mitigation_matrix.csv", index=False)

    report = {
        "day": 47,
        "title": "Risk and Failure Mode Analysis",
        "status": "complete",
        "artifacts_scanned": int(len(inventory)),
        "risk_count": int(len(risks)),
        "risk_ids": [r["id"] for r in risks],
        "risks": risks,
        "outputs": {
            "json": (out_dir / "failure_mode_analysis.json").as_posix(),
            "markdown": (out_dir / "failure_mode_analysis.md").as_posix(),
            "mitigation_matrix": (out_dir / "mitigation_matrix.csv").as_posix(),
            "evidence_inventory": (out_dir / "evidence_inventory.csv").as_posix(),
            "docs_copy": (docs_dir / "day47_risk_failure_modes.md").as_posix(),
        },
    }

    write_json(out_dir / "failure_mode_analysis.json", report)

    md = build_markdown(risks, inventory)
    (out_dir / "failure_mode_analysis.md").write_text(md, encoding="utf-8")
    (docs_dir / "day47_risk_failure_modes.md").write_text(md, encoding="utf-8")

    readme = """# Day 47 — Risk and Failure Mode Analysis

## Status
Complete.

## Outputs
- `failure_mode_analysis.json`
- `failure_mode_analysis.md`
- `mitigation_matrix.csv`
- `evidence_inventory.csv`
- `../../docs/day47_risk_failure_modes.md`

## Scientific role
This artifact supports the paper's limitations, failure modes, threat-to-validity, and mitigation sections.

## Main principle
The system should be interpreted as a research prototype for anomaly explanation and data-quality review, not as an autonomous clinical decision-support tool.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({
        "status": "complete",
        "risk_count": len(risks),
        "artifacts_scanned": len(inventory),
        "out_dir": out_dir.as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
