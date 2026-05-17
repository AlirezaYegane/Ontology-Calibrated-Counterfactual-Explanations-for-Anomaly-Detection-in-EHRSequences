from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(".")
OUT_DIR = ROOT / "artifacts" / "day42"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_first_existing(candidates: list[str]) -> str | None:
    for item in candidates:
        if exists(item):
            return item
    return None


def collect_json_artifacts(day_dir: str) -> list[str]:
    path = ROOT / day_dir
    if not path.exists():
        return []
    return sorted(str(p).replace("\\", "/") for p in path.rglob("*.json"))


def collect_csv_artifacts(day_dir: str) -> list[str]:
    path = ROOT / day_dir
    if not path.exists():
        return []
    return sorted(str(p).replace("\\", "/") for p in path.rglob("*.csv"))


def status_from_paths(
    required: list[str], optional: list[str] | None = None
) -> dict[str, Any]:
    optional = optional or []
    required_status = {p: exists(p) for p in required}
    optional_status = {p: exists(p) for p in optional}
    complete = all(required_status.values()) if required_status else False
    return {
        "complete": complete,
        "required": required_status,
        "optional": optional_status,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    modules = {
        "data_and_ontology_pipeline": status_from_paths(
            required=[
                "artifacts/day13/mapping_audit.json",
            ],
            optional=[
                "artifacts/day14",
                "src/ontology",
                "src/preprocessing",
            ],
        ),
        "supervised_detector": status_from_paths(
            required=[
                "src/models/detector_supervised.py",
                "artifacts/day20/day20_supervised_eval_summary.json",
            ],
            optional=[
                "src/training/detector_supervised_utils.py",
                "src/training/train_detector_supervised.py",
                "src/evaluation/evaluate_detector_supervised.py",
                "outputs/detector/day20_supervised",
                "outputs/detector_eval/day20_supervised",
            ],
        ),
        "diffusion_generative_model": status_from_paths(
            required=[
                "src/models/diffusion.py",
                "src/training/train_diffusion.py",
            ],
            optional=[
                "artifacts/day28/day28_diffusion_decisions.json",
                "artifacts/day34_final/day34_final_assessment.json",
                "src/models/diffusion_legacy_day33.py",
            ],
        ),
        "calibrated_scoring": status_from_paths(
            required=[
                "artifacts/day35",
            ],
            optional=[
                "artifacts/day35_5",
                "artifacts/day35_6",
                "artifacts/day35_7",
            ],
        ),
        "counterfactual_generator": status_from_paths(
            required=[
                "artifacts/day36",
                "artifacts/day37",
            ],
            optional=[
                "artifacts/day36_repair_ready",
            ],
        ),
        "explanation_and_case_studies": status_from_paths(
            required=[
                "artifacts/day38",
                "artifacts/day39",
            ],
            optional=[
                "scripts/run_day39_end_to_end_case_studies.py",
            ],
        ),
        "ablation_framework_and_results": status_from_paths(
            required=[
                "artifacts/day40",
                "artifacts/day41",
            ],
            optional=[
                "scripts/run_day41_ablation_studies.py",
                "scripts/summarize_day41_ablation.py",
            ],
        ),
    }

    evidence_artifacts = {
        "day20": {
            "json": collect_json_artifacts("artifacts/day20"),
            "csv": collect_csv_artifacts("artifacts/day20"),
        },
        "day28": {
            "json": collect_json_artifacts("artifacts/day28"),
            "csv": collect_csv_artifacts("artifacts/day28"),
        },
        "day34_final": {
            "json": collect_json_artifacts("artifacts/day34_final"),
            "csv": collect_csv_artifacts("artifacts/day34_final"),
        },
        "day35": {
            "json": collect_json_artifacts("artifacts/day35"),
            "csv": collect_csv_artifacts("artifacts/day35"),
        },
        "day36": {
            "json": collect_json_artifacts("artifacts/day36"),
            "csv": collect_csv_artifacts("artifacts/day36"),
        },
        "day37": {
            "json": collect_json_artifacts("artifacts/day37"),
            "csv": collect_csv_artifacts("artifacts/day37"),
        },
        "day38": {
            "json": collect_json_artifacts("artifacts/day38"),
            "csv": collect_csv_artifacts("artifacts/day38"),
        },
        "day39": {
            "json": collect_json_artifacts("artifacts/day39"),
            "csv": collect_csv_artifacts("artifacts/day39"),
        },
        "day40": {
            "json": collect_json_artifacts("artifacts/day40"),
            "csv": collect_csv_artifacts("artifacts/day40"),
        },
        "day41": {
            "json": collect_json_artifacts("artifacts/day41"),
            "csv": collect_csv_artifacts("artifacts/day41"),
        },
    }

    day20_summary_path = (
        ROOT / "artifacts" / "day20" / "day20_supervised_eval_summary.json"
    )
    day34_summary_path = (
        ROOT / "artifacts" / "day34_final" / "day34_final_assessment.json"
    )

    day20_summary = load_json_if_exists(day20_summary_path)
    day34_summary = load_json_if_exists(day34_summary_path)

    module_rows = []
    for name, payload in modules.items():
        module_rows.append(
            {
                "module": name,
                "status": "complete" if payload["complete"] else "needs_review",
                "missing_required": [
                    p for p, ok in payload["required"].items() if not ok
                ],
                "present_required": [p for p, ok in payload["required"].items() if ok],
            }
        )

    completed_modules = sum(1 for row in module_rows if row["status"] == "complete")
    total_modules = len(module_rows)

    gaps = [
        {
            "gap_id": "G1",
            "area": "Generative surprise Sgen",
            "severity": "high_for_claims_low_for_pipeline",
            "finding": "Current diffusion denoising-error Sgen is not discriminative enough to be claimed as the main anomaly signal.",
            "paper_position": "Keep Sgen as auxiliary/diagnostic; emphasize detector + ontology-calibrated scoring as the empirically supported signal.",
            "next_action": "In future work, test stronger likelihood proxies, conditional diffusion scoring, or reconstruction-calibrated generative objectives.",
        },
        {
            "gap_id": "G2",
            "area": "Counterfactual evaluation",
            "severity": "medium",
            "finding": "Counterfactual outputs exist but need broader test-set evaluation and more systematic plausibility review.",
            "paper_position": "Use current case studies as preliminary qualitative evidence; avoid overclaiming clinical validity.",
            "next_action": "Day 45-46 should run comprehensive test evaluation and human/plausibility review.",
        },
        {
            "gap_id": "G3",
            "area": "Runtime profiling",
            "severity": "medium",
            "finding": "Core functionality is implemented, but component-level latency is not yet profiled.",
            "paper_position": "Report runtime only after Day 43 profiling.",
            "next_action": "Measure scoring, ontology checks, counterfactual search, and explanation generation separately.",
        },
        {
            "gap_id": "G4",
            "area": "Clinical validation",
            "severity": "expected_limitation",
            "finding": "No clinician-rated validation has been completed yet.",
            "paper_position": "Frame clinical usefulness as preliminary and future-facing unless expert review is later obtained.",
            "next_action": "Prepare a curated review sheet for representative explanations.",
        },
        {
            "gap_id": "G5",
            "area": "Cross-dataset robustness",
            "severity": "future_work",
            "finding": "Current evidence is strongest on the current MIMIC-derived benchmark and synthetic anomaly setup.",
            "paper_position": "Claim reproducible benchmark evidence, not universal deployment readiness.",
            "next_action": "Later evaluate on MIMIC-IV/eICU subsets if time allows.",
        },
    ]

    assessment = {
        "day": 42,
        "title": "Milestone 3 - Core System Completion",
        "generated_at": now_utc(),
        "status": "complete_with_documented_limitations",
        "purpose": "Review detector, diffusion/generative, ontology scoring, counterfactual, explanation, and ablation modules before moving into formal evaluation and paper-oriented polishing.",
        "module_completion": {
            "completed_modules": completed_modules,
            "total_modules": total_modules,
            "completion_rate": round(completed_modules / total_modules, 4),
            "rows": module_rows,
        },
        "key_scientific_position": {
            "supported_claim": "The current system supports an ontology-calibrated anomaly explanation pipeline with detector and ontology-driven scoring as the strongest empirical signals.",
            "careful_claim": "The generative component is implemented and evaluated, but the current raw Sgen proxy should be treated as auxiliary/diagnostic rather than the primary discriminative signal.",
            "do_not_claim": "Do not claim that diffusion Sgen alone meaningfully separates anomalies unless a stronger future Sgen definition is validated.",
        },
        "known_results": {
            "day20_supervised_detector": day20_summary,
            "day34_generative_assessment": day34_summary,
        },
        "evidence_artifacts": evidence_artifacts,
        "gap_register": gaps,
        "readiness_for_next_phase": {
            "ready_for_day43_profiling": True,
            "ready_for_day44_interface": True,
            "ready_for_day45_comprehensive_evaluation": True,
            "ready_for_paper_writing": True,
            "condition": "Ready as a core research prototype, with limitations explicitly documented and no overclaiming around Sgen.",
        },
    }

    json_path = OUT_DIR / "day42_core_system_assessment.json"
    json_path.write_text(
        json.dumps(assessment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    gap_csv_path = OUT_DIR / "day42_gap_register.csv"
    with gap_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "gap_id",
                "area",
                "severity",
                "finding",
                "paper_position",
                "next_action",
            ],
        )
        writer.writeheader()
        writer.writerows(gaps)

    md_lines = []
    md_lines.append("# Day 42 — Milestone 3: Core System Completion")
    md_lines.append("")
    md_lines.append("## Status")
    md_lines.append("")
    md_lines.append("Complete with documented limitations.")
    md_lines.append("")
    md_lines.append("## Purpose")
    md_lines.append("")
    md_lines.append(
        "Day 42 closes the core-system milestone by reviewing the implemented detector, diffusion/generative, ontology scoring, counterfactual, explanation, and ablation modules before moving into formal evaluation and paper-oriented polishing."
    )
    md_lines.append("")
    md_lines.append("## Module Completion")
    md_lines.append("")
    md_lines.append("| Module | Status | Missing required items |")
    md_lines.append("|---|---|---|")
    for row in module_rows:
        missing = (
            ", ".join(row["missing_required"]) if row["missing_required"] else "None"
        )
        md_lines.append(f"| `{row['module']}` | {row['status']} | {missing} |")
    md_lines.append("")
    md_lines.append("## Main Scientific Position")
    md_lines.append("")
    md_lines.append(
        "The current evidence supports the project as an ontology-calibrated anomaly explanation pipeline. The strongest empirical signal comes from the detector and ontology-calibrated scoring components."
    )
    md_lines.append("")
    md_lines.append(
        "The generative diffusion component has been implemented and evaluated, but the current raw `Sgen` proxy should be treated as an auxiliary or diagnostic signal rather than the main anomaly-discrimination signal."
    )
    md_lines.append("")
    md_lines.append("## What We Can Claim")
    md_lines.append("")
    md_lines.append("- The core pipeline is implemented end-to-end.")
    md_lines.append(
        "- The system can decompose anomaly evidence into detector/statistical and ontology-informed components."
    )
    md_lines.append(
        "- Counterfactual and explanation artifacts exist and can support qualitative case studies."
    )
    md_lines.append(
        "- Ablation evidence is available for comparing full and simplified variants."
    )
    md_lines.append(
        "- The current results should be presented conservatively, especially around the generative `Sgen` component."
    )
    md_lines.append("")
    md_lines.append("## What We Should Not Overclaim")
    md_lines.append("")
    md_lines.append("- Do not claim clinical deployment readiness.")
    md_lines.append(
        "- Do not claim clinician-validated explanation usefulness unless a human review is actually completed."
    )
    md_lines.append(
        "- Do not claim that raw diffusion `Sgen` is the main discriminative anomaly signal."
    )
    md_lines.append(
        "- Do not claim cross-dataset robustness until MIMIC-IV/eICU validation is run."
    )
    md_lines.append("")
    md_lines.append("## Gap Register")
    md_lines.append("")
    md_lines.append("| ID | Area | Severity | Finding | Next Action |")
    md_lines.append("|---|---|---|---|---|")
    for gap in gaps:
        md_lines.append(
            f"| {gap['gap_id']} | {gap['area']} | {gap['severity']} | {gap['finding']} | {gap['next_action']} |"
        )
    md_lines.append("")
    md_lines.append("## Readiness Decision")
    md_lines.append("")
    md_lines.append(
        "The core system is ready to move into Weeks 7–8: profiling, interface wrapping, comprehensive test-set evaluation, plausibility review, and paper-oriented polishing."
    )
    md_lines.append("")
    md_lines.append("## Next Phase")
    md_lines.append("")
    md_lines.append("- Day 43: performance profiling")
    md_lines.append("- Day 44: simple user-facing interface / CLI")
    md_lines.append("- Day 45: comprehensive held-out evaluation")
    md_lines.append("- Day 46: plausibility review and explanation refinement")
    md_lines.append("")

    md_path = OUT_DIR / "day42_core_system_assessment.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    next_lines = []
    next_lines.append("# Day 42 — Next Phase Plan")
    next_lines.append("")
    next_lines.append("## Immediate transition")
    next_lines.append("")
    next_lines.append(
        "The project should now move from core implementation to evaluation hardening."
    )
    next_lines.append("")
    next_lines.append("## Day 43")
    next_lines.append(
        "- Profile runtime for detector scoring, ontology scoring, calibrated scoring, counterfactual search, and explanation generation."
    )
    next_lines.append("- Produce component-level latency table.")
    next_lines.append("")
    next_lines.append("## Day 44")
    next_lines.append(
        "- Build a CLI or notebook interface that runs the full pipeline on selected records."
    )
    next_lines.append("- Output readable explanations and counterfactual edits.")
    next_lines.append("")
    next_lines.append("## Day 45")
    next_lines.append("- Run full evaluation on the held-out benchmark.")
    next_lines.append(
        "- Report ROC-AUC, AP, F1, threshold behavior, and anomaly-type breakdown."
    )
    next_lines.append("")
    next_lines.append("## Day 46")
    next_lines.append("- Curate representative explanations for plausibility review.")
    next_lines.append("- Refine templates and identify misleading cases.")
    next_lines.append("")
    next_lines.append("## Paper strategy")
    next_lines.append(
        "- Lead with ontology-calibrated scoring and counterfactual explanation."
    )
    next_lines.append(
        "- Present diffusion/Sgen honestly as implemented but currently weak as a standalone discriminative signal."
    )
    next_lines.append(
        "- Use ablation results to motivate a conservative final configuration."
    )
    next_lines.append("")

    next_path = OUT_DIR / "day42_next_phase_plan.md"
    next_path.write_text("\n".join(next_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "day42 assessment complete",
                "json": str(json_path),
                "markdown": str(md_path),
                "gap_register": str(gap_csv_path),
                "next_phase_plan": str(next_path),
                "completed_modules": completed_modules,
                "total_modules": total_modules,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
