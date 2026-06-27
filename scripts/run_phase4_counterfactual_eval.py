"""
scripts/run_phase4_counterfactual_eval.py
==========================================
Phase 4 -- Leakage-free counterfactual repair evaluation on benchmark-v2.

Selects anomalous records, runs the leakage-free generator
(:func:`src.explanations.counterfactual.generate_counterfactual`), scores
before/after with the real OntologyAwareScorer (and the smoke detector as a
DIAGNOSTIC-only signal), and writes per-record + aggregate results.

LEAKAGE NOTE: ``label`` / ``anomaly_type`` are used ONLY to (a) select repair
targets and (b) bucket the report. They are never passed to the generator, which
receives a model-visible-only row.

Outputs (artifacts/phase4/):
  counterfactual_results.jsonl, counterfactual_summary.{json,md},
  edit_type_breakdown.csv, failure_cases.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_ONT = PROJECT_ROOT / "ontologies" / "processed"
V2_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_v2"
DETECTOR_DIR = PROJECT_ROOT / "artifacts" / "phase3" / "detector_unsup_v2"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase4"


def _load(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_pickle(path).to_dict(orient="records")


def _model_visible_row(rec: dict[str, Any]) -> dict[str, Any]:
    seq = rec.get("model_visible_sequence", rec.get("codes", []))
    return {
        "model_visible_sequence": [str(t) for t in seq]
        if isinstance(seq, (list, tuple))
        else [],
        "gender": rec.get("gender"),
        "age_group": rec.get("age_group"),
    }


def _maybe_load_detector(use: bool):
    if not use:
        return None, "disabled"
    if not (DETECTOR_DIR / "detector_unsup.pt").exists():
        return None, "unavailable (no checkpoint)"
    try:
        from src.models.detector_unsup import UnsupervisedSequenceDetector

        return UnsupervisedSequenceDetector.load(DETECTOR_DIR), "loaded (smoke-scale)"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"load_failed: {exc}"


def _violation_kinds(viols: list[dict[str, Any]]) -> set[str]:
    return {str(v.get("kind")) for v in viols}


def run(
    split: str,
    max_records: int,
    seed: int,
    max_edits: int,
    beam_size: int,
    use_detector: bool,
    out_dir: Path,
) -> dict[str, Any]:
    from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

    records = _load(V2_DIR / f"{split}.pkl")
    anomalies = [r for r in records if int(r.get("label", 0)) == 1]
    if max_records and len(anomalies) > max_records:
        anomalies = anomalies[:max_records]

    scorer = OntologyAwareScorer.from_processed_dir(
        PROCESSED_ONT, ontology_mode="real", weights=ScoreWeights()
    )
    detector, detector_status = _maybe_load_detector(use_detector)
    return evaluate_records(
        anomalies,
        scorer,
        scorer.engine.index,
        detector=detector,
        detector_status=detector_status,
        split=split,
        seed=seed,
        max_edits=max_edits,
        beam_size=beam_size,
        max_records=max_records,
        out_dir=out_dir,
    )


def evaluate_records(
    anomalies: list[dict[str, Any]],
    scorer: Any,
    index: Any,
    *,
    detector: Any = None,
    detector_status: str = "disabled",
    split: str = "test",
    seed: int = 42,
    max_edits: int = 3,
    beam_size: int = 20,
    max_records: int = 0,
    out_dir: Path = OUT_DIR,
) -> dict[str, Any]:
    """Core eval loop (testable): repair each record, write artifacts, return
    the aggregate summary. ``anomalies`` are raw benchmark rows; only their
    model-visible fields are passed to the generator."""
    from src.explanations.counterfactual import generate_counterfactual

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "counterfactual_results.jsonl"
    failures_path = out_dir / "failure_cases.jsonl"

    per_results: list[dict[str, Any]] = []
    op_counter: Counter = Counter()
    flagged_count = 0  # records the ontology actually flags (S_ont_before > 0)
    flagged_success = 0
    deltas_ont: list[float] = []
    deltas_cal: list[float] = []
    deltas_det: list[float] = []
    edit_counts: list[int] = []
    distances: list[int] = []
    failure_reasons: Counter = Counter()
    success_by_type: dict[str, list[int]] = defaultdict(list)
    success_by_rule: dict[str, list[int]] = defaultdict(list)

    with (
        results_path.open("w", encoding="utf-8") as rf,
        failures_path.open("w", encoding="utf-8") as ff,
    ):
        for i, rec in enumerate(anomalies):
            row = _model_visible_row(rec)  # generator sees ONLY this
            result = generate_counterfactual(
                row,
                scorer,
                index,
                detector=detector,
                max_edits=max_edits,
                beam_size=beam_size,
                seed=seed,
            )
            rd = result.to_dict()
            # analysis-only metadata (label/anomaly_type) -- NOT generator input,
            # never includes hidden_eval/audit answer keys.
            atype = str(rec.get("anomaly_type") or "anomaly")
            rd_out = {"record_index": i, "anomaly_type": atype, **rd}
            rf.write(json.dumps(rd_out, default=str) + "\n")

            per_results.append(rd_out)
            for e in result.edits:
                op_counter[e.operation] += 1
            success = 1 if result.validity else 0
            if result.s_ont_before > 0:
                flagged_count += 1
                flagged_success += success
            success_by_type[atype].append(success)
            for k in _violation_kinds(result.rule_violations_before) or {"none"}:
                success_by_rule[k].append(success)

            if result.validity:
                deltas_ont.append(result.delta_s_ont)
                if result.delta_s_cal is not None:
                    deltas_cal.append(result.delta_s_cal)
                if result.delta_s_det is not None:
                    deltas_det.append(result.delta_s_det)
                edit_counts.append(result.num_edits)
                distances.append(result.total_distance)
            else:
                failure_reasons[result.failure_reason or result.status] += 1
                ff.write(json.dumps(rd_out, default=str) + "\n")

    attempted = len(anomalies)
    succeeded = sum(1 for r in per_results if r["validity"])

    def _mean(xs: list[float]) -> float | None:
        return round(statistics.mean(xs), 6) if xs else None

    summary = {
        "phase": 4,
        "split": split,
        "benchmark": "v2 (non-circular)",
        "generator": "leakage_free_ontology_counterfactual",
        "detector_status": detector_status,
        "config": {
            "max_records": max_records,
            "seed": seed,
            "max_edits": max_edits,
            "beam_size": beam_size,
        },
        "repair_attempted_count": attempted,
        "repair_success_count": succeeded,
        "repair_success_rate": round(succeeded / attempted, 4) if attempted else 0.0,
        "validity_rate": round(succeeded / attempted, 4) if attempted else 0.0,
        "ontology_flagged_count": flagged_count,
        "repair_success_rate_among_flagged": (
            round(flagged_success / flagged_count, 4) if flagged_count else 0.0
        ),
        "unflagged_count": attempted - flagged_count,
        "unflagged_note": (
            "records with S_ont_before==0 are NOT flagged by the ontology (Phase 3b "
            "coverage gap, e.g. uncovered medication anomalies); they are detection "
            "gaps, not Phase 4 repair failures."
        ),
        "mean_delta_s_ont": _mean(deltas_ont),
        "mean_delta_s_cal": _mean(deltas_cal),
        "mean_delta_s_det": _mean(deltas_det),
        "mean_num_edits": _mean([float(x) for x in edit_counts]),
        "median_num_edits": (statistics.median(edit_counts) if edit_counts else None),
        "mean_ontology_distance": _mean([float(x) for x in distances]),
        "edit_operation_counts": dict(op_counter),
        "failure_reason_counts": dict(failure_reasons),
        "success_by_anomaly_type": {
            k: {
                "n": len(v),
                "success": sum(v),
                "rate": round(sum(v) / len(v), 4) if v else 0.0,
            }
            for k, v in sorted(success_by_type.items())
        },
        "success_by_rule_type": {
            k: {
                "n": len(v),
                "success": sum(v),
                "rate": round(sum(v) / len(v), 4) if v else 0.0,
            }
            for k, v in sorted(success_by_rule.items())
        },
        "detector_caveat": (
            "S_det is smoke-scale and diagnostic-only; it does NOT drive repair and "
            "ΔS_det must not be read as repair success."
        ),
        "leakage_note": "Generator received model-visible rows only; label/anomaly_type used for selection+bucketing only.",
    }

    (out_dir / "counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_md(summary, out_dir / "counterfactual_summary.md")
    _write_breakdown_csv(op_counter, summary, out_dir / "edit_type_breakdown.csv")
    return summary


def _write_md(summary: dict[str, Any], path: Path) -> None:
    s = summary
    md = [
        "# Phase 4 -- Counterfactual Repair Evaluation\n",
        f"**Split:** `{s['split']}` ({s['benchmark']}) | **detector:** {s['detector_status']}\n",
        f"- repair attempted: **{s['repair_attempted_count']}** "
        f"(ontology-flagged: {s['ontology_flagged_count']}, "
        f"unflagged/detection-gap: {s['unflagged_count']})",
        f"- repair success (valid): **{s['repair_success_count']}** "
        f"(rate over all **{s['repair_success_rate']}**; "
        f"**over flagged {s['repair_success_rate_among_flagged']}**)",
        f"- mean ΔS_ont: **{s['mean_delta_s_ont']}** | mean ΔS_cal: {s['mean_delta_s_cal']} "
        f"| mean ΔS_det: {s['mean_delta_s_det']} (diagnostic-only)",
        f"- mean edits: **{s['mean_num_edits']}** | median edits: {s['median_num_edits']} "
        f"| mean ontology distance: {s['mean_ontology_distance']}",
        f"- edit ops: {s['edit_operation_counts']}",
        f"- failure reasons: {s['failure_reason_counts']}\n",
        "## Success by anomaly type",
        "| anomaly_type | n | success | rate |",
        "|---|---:|---:|---:|",
    ]
    for k, v in s["success_by_anomaly_type"].items():
        md.append(f"| {k} | {v['n']} | {v['success']} | {v['rate']} |")
    md.append("\n## Success by rule type (violation kind before repair)")
    md.append("| rule_kind | n | success | rate |")
    md.append("|---|---:|---:|---:|")
    for k, v in s["success_by_rule_type"].items():
        md.append(f"| {k} | {v['n']} | {v['success']} | {v['rate']} |")
    md.append(f"\n> {s['detector_caveat']}")
    md.append(f">\n> {s['leakage_note']}")
    path.write_text("\n".join(md), encoding="utf-8")


def _write_breakdown_csv(
    op_counter: Counter, summary: dict[str, Any], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["edit_operation", "count"])
        for op, n in sorted(op_counter.items()):
            w.writerow([op, n])
        w.writerow([])
        w.writerow(["anomaly_type", "n", "success", "rate"])
        for k, v in summary["success_by_anomaly_type"].items():
            w.writerow([k, v["n"], v["success"], v["rate"]])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4 counterfactual eval.")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--max-records", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-edits", type=int, default=3)
    ap.add_argument("--beam-size", type=int, default=20)
    ap.add_argument(
        "--smoke", action="store_true", help="tiny run (caps max-records to 100)"
    )
    ap.add_argument(
        "--use-detector", dest="use_detector", action="store_true", default=True
    )
    ap.add_argument("--no-detector", dest="use_detector", action="store_false")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    max_records = min(args.max_records, 100) if args.smoke else args.max_records
    summary = run(
        split=args.split,
        max_records=max_records,
        seed=args.seed,
        max_edits=args.max_edits,
        beam_size=args.beam_size,
        use_detector=args.use_detector,
        out_dir=Path(args.out_dir),
    )
    print(
        f"[phase4] attempted={summary['repair_attempted_count']} "
        f"success={summary['repair_success_count']} "
        f"rate={summary['repair_success_rate']} "
        f"mean_dS_ont={summary['mean_delta_s_ont']} "
        f"mean_edits={summary['mean_num_edits']} detector={summary['detector_status']}"
    )
    for k, v in summary["success_by_anomaly_type"].items():
        print(f"[phase4]   {k}: {v['success']}/{v['n']} ({v['rate']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
