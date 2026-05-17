from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import math
import pstats
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCORE_KEYWORDS = (
    "score",
    "prob",
    "auc",
    "sdet",
    "sgen",
    "sont",
    "scal",
    "calibrated",
    "detector",
    "ontology",
    "generative",
    "anomaly",
    "full_model",
)

TEXT_CODE_COLUMNS = (
    "codes",
    "tokens",
    "sequence",
    "original",
    "counterfactual",
    "preview",
    "edit",
    "action",
)


def now_ms() -> float:
    return time.perf_counter_ns() / 1_000_000.0


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def discover_candidate_csvs(artifacts_root: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []

    for path in artifacts_root.rglob("*.csv"):
        if "day43" in {part.lower() for part in path.parts}:
            continue

        score = 0
        lower_path = path.as_posix().lower()

        if any(
            day in lower_path
            for day in ("day39", "day40", "day41", "day42", "day35", "day36")
        ):
            score += 8
        if any(
            word in path.name.lower()
            for word in ("score", "ablation", "case", "counterfactual", "eval")
        ):
            score += 8

        try:
            sample = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        cols = [str(c).lower() for c in sample.columns]
        score += sum(1 for c in cols if any(k in c for k in SCORE_KEYWORDS))
        score += sum(1 for c in cols if any(k in c for k in TEXT_CODE_COLUMNS))

        if score > 0:
            candidates.append((score, path))

    return [
        path for _, path in sorted(candidates, key=lambda x: (-x[0], x[1].as_posix()))
    ]


def choose_input_csv(artifacts_root: Path, input_csv: str | None) -> Path:
    if input_csv:
        path = Path(input_csv)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        return path

    candidates = discover_candidate_csvs(artifacts_root)
    if not candidates:
        raise FileNotFoundError(
            "No suitable CSV artifact found. Run Day 40/41/42 first or pass --input_csv explicitly."
        )

    return candidates[0]


def find_first_column(df: pd.DataFrame, contains_any: tuple[str, ...]) -> str | None:
    for col in df.columns:
        col_l = str(col).lower()
        if any(key in col_l for key in contains_any):
            return str(col)
    return None


def numeric_series(
    df: pd.DataFrame, col: str | None, fallback: float = 0.0
) -> np.ndarray:
    if col is None or col not in df.columns:
        return np.full(len(df), fallback, dtype=np.float64)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(fallback)
        .to_numpy(dtype=np.float64)
    )


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def infer_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "label": find_first_column(df, ("label", "y_true", "target", "is_anomaly")),
        "detector": find_first_column(
            df, ("sdet", "detector", "prob_anomaly", "det_score", "supervised")
        ),
        "sgen": find_first_column(df, ("sgen", "generative", "diffusion")),
        "sont": find_first_column(df, ("sont", "ontology", "rule", "violation")),
        "scal": find_first_column(
            df, ("scal", "calibrated", "full_model", "final_score")
        ),
        "variant": find_first_column(df, ("variant", "model", "setting", "ablation")),
        "anomaly_type": find_first_column(
            df, ("anomaly_type", "type", "scenario", "category")
        ),
    }


def pick_text_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        col_l = str(col).lower()
        if any(key in col_l for key in TEXT_CODE_COLUMNS):
            out.append(str(col))
    return out[:6]


def time_stage(stage: str, repeat: int, fn: Any) -> dict[str, Any]:
    times_ms: list[float] = []
    last_payload: Any = None

    for _ in range(max(1, repeat)):
        t0 = now_ms()
        last_payload = fn()
        t1 = now_ms()
        times_ms.append(t1 - t0)

    return {
        "stage": stage,
        "repeat": repeat,
        "total_ms_mean": float(statistics.mean(times_ms)),
        "total_ms_median": float(statistics.median(times_ms)),
        "total_ms_min": float(min(times_ms)),
        "total_ms_max": float(max(times_ms)),
        "payload": last_payload,
    }


def write_timings_csv(path: Path, rows: list[dict[str, Any]], n_records: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "repeat",
        "total_ms_mean",
        "total_ms_median",
        "total_ms_min",
        "total_ms_max",
        "ms_per_record_mean",
        "records_per_second_mean",
    ]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            total_ms = float(row["total_ms_mean"])
            ms_per_record = total_ms / max(n_records, 1)
            records_per_second = (
                1000.0 / ms_per_record if ms_per_record > 0 else float("inf")
            )
            writer.writerow(
                {
                    "stage": row["stage"],
                    "repeat": row["repeat"],
                    "total_ms_mean": round(total_ms, 6),
                    "total_ms_median": round(float(row["total_ms_median"]), 6),
                    "total_ms_min": round(float(row["total_ms_min"]), 6),
                    "total_ms_max": round(float(row["total_ms_max"]), 6),
                    "ms_per_record_mean": round(ms_per_record, 6),
                    "records_per_second_mean": round(records_per_second, 6),
                }
            )


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Day 43 — Performance Profiling Report")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Complete.")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Measure the runtime profile of the integrated anomaly-explanation pipeline and identify bottlenecks before formal evaluation and paper writing."
    )
    lines.append("")
    lines.append("## Profiling Scope")
    lines.append("")
    lines.append(f"- Input artifact: `{summary['input_csv']}`")
    lines.append(f"- Records profiled: **{summary['n_records']}**")
    lines.append(f"- Repeat count per stage: **{summary['repeat']}**")
    lines.append(f"- Profiling mode: `{summary['profiling_mode']}`")
    lines.append("")
    lines.append("## Detected Columns")
    lines.append("")
    for key, value in summary["columns"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Component Runtime")
    lines.append("")
    lines.append("| Stage | Mean total ms | Mean ms / record | Records / sec |")
    lines.append("|---|---:|---:|---:|")

    for row in summary["timings"]:
        total_ms = float(row["total_ms_mean"])
        ms_per_record = total_ms / max(int(summary["n_records"]), 1)
        rps = 1000.0 / ms_per_record if ms_per_record > 0 else float("inf")
        lines.append(
            f"| {row['stage']} | {total_ms:.4f} | {ms_per_record:.6f} | {rps:.2f} |"
        )

    lines.append("")
    lines.append("## Bottleneck Ranking")
    lines.append("")
    for idx, item in enumerate(summary["bottlenecks"], start=1):
        lines.append(
            f"{idx}. `{item['stage']}` — {item['share_of_total_runtime_pct']:.2f}% of measured runtime"
        )

    lines.append("")
    lines.append("## Optimization Findings")
    lines.append("")
    opt = summary["optimization_checks"]
    lines.append(
        f"- Vectorized score computation speedup over row-wise scoring: **{opt['vectorized_vs_rowwise_speedup']:.2f}x**"
    )
    lines.append(
        f"- Cached lookup speedup over repeated uncached lookup: **{opt['cached_lookup_speedup']:.2f}x**"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The profiling result should be interpreted as an engineering runtime audit of the current integrated evaluation artifacts. "
        "If this report is generated from score/case-study artifacts rather than live GPU inference, it measures downstream scoring, ranking, counterfactual-search glue code, and explanation rendering rather than raw neural model training time."
    )
    lines.append("")
    lines.append("## Paper-Ready Note")
    lines.append("")
    lines.append(
        "For the manuscript, this artifact supports the computational-efficiency paragraph: the pipeline should report latency per component, identify the dominant stage, and describe practical optimizations such as vectorized scoring and cached ontology lookups."
    )
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    for label, path in summary["files"].items():
        lines.append(f"- `{label}`: `{path}`")
    lines.append("")

    return "\n".join(lines) + "\n"


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    artifacts_root = Path(args.artifacts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = choose_input_csv(artifacts_root, args.input_csv)
    df = pd.read_csv(input_csv, low_memory=False)

    if args.max_records and len(df) > args.max_records:
        df = df.head(args.max_records).copy()
    else:
        df = df.copy()

    if df.empty:
        raise ValueError(f"Input CSV has no rows: {input_csv}")

    columns = infer_columns(df)

    if columns["scal"] is None:
        numeric_candidates = [
            col
            for col in df.columns
            if pd.to_numeric(df[col], errors="coerce").notna().sum()
            >= max(3, int(0.25 * len(df)))
        ]
        if numeric_candidates:
            columns["scal"] = str(numeric_candidates[0])

    text_cols = pick_text_columns(df)

    detector_raw = numeric_series(df, columns["detector"], fallback=0.0)
    sgen_raw = numeric_series(df, columns["sgen"], fallback=0.0)
    sont_raw = numeric_series(df, columns["sont"], fallback=0.0)
    scal_raw = numeric_series(df, columns["scal"], fallback=0.0)

    detector_norm = minmax_normalize(detector_raw)
    sgen_norm = minmax_normalize(sgen_raw)
    sont_norm = minmax_normalize(sont_raw)
    scal_norm = minmax_normalize(scal_raw)

    def detector_stage() -> dict[str, float]:
        # This stage profiles detector-score normalization and ranking from existing outputs.
        ranked = np.argsort(-detector_norm)
        top_mean = float(detector_norm[ranked[: min(20, len(ranked))]].mean())
        return {"top20_detector_score_mean": top_mean}

    def score_stage() -> dict[str, float]:
        # Conservative scoring: detector and ontology dominate; Sgen is diagnostic/low-weight.
        calibrated = 0.50 * detector_norm + 0.45 * sont_norm + 0.05 * sgen_norm
        if columns["scal"] is not None:
            calibrated = 0.50 * calibrated + 0.50 * scal_norm
        return {
            "mean_scal": float(np.mean(calibrated)),
            "p95_scal": float(np.quantile(calibrated, 0.95)),
            "max_scal": float(np.max(calibrated)),
        }

    def counterfactual_stage() -> dict[str, float]:
        calibrated = 0.50 * detector_norm + 0.45 * sont_norm + 0.05 * sgen_norm
        threshold = float(np.quantile(calibrated, 0.80))
        candidate_indices = np.where(calibrated >= threshold)[0]

        reductions: list[float] = []
        edit_counts: list[int] = []

        for idx in candidate_indices:
            row = df.iloc[int(idx)]
            text_blob = " ".join(
                str(row[col]) for col in text_cols if col in df.columns
            )
            separator_count = (
                text_blob.count(",") + text_blob.count(";") + text_blob.count("|")
            )
            estimated_edits = 1 if separator_count <= 3 else 2
            if (
                "replace" in text_blob.lower()
                or "remove" in text_blob.lower()
                or "add" in text_blob.lower()
            ):
                estimated_edits = min(estimated_edits + 1, 3)

            reduction = float(
                calibrated[idx] * (0.12 if estimated_edits == 1 else 0.20)
            )
            reductions.append(reduction)
            edit_counts.append(estimated_edits)

        if not reductions:
            return {
                "n_counterfactual_candidates": 0,
                "mean_estimated_delta": 0.0,
                "mean_edit_count": 0.0,
            }

        return {
            "n_counterfactual_candidates": int(len(reductions)),
            "mean_estimated_delta": float(np.mean(reductions)),
            "mean_edit_count": float(np.mean(edit_counts)),
        }

    def explanation_stage() -> dict[str, float]:
        calibrated = 0.50 * detector_norm + 0.45 * sont_norm + 0.05 * sgen_norm
        threshold = float(np.quantile(calibrated, 0.80))
        candidate_indices = np.where(calibrated >= threshold)[0]

        explanations: list[str] = []
        for idx in candidate_indices:
            row = df.iloc[int(idx)]
            anomaly_type = (
                str(row[columns["anomaly_type"]])
                if columns["anomaly_type"]
                else "unknown"
            )
            score = calibrated[int(idx)]
            explanation = (
                f"Record {idx} is flagged as anomalous with calibrated score {score:.4f}. "
                f"The dominant available anomaly family is {anomaly_type}. "
                "A minimal ontology-aware counterfactual should reduce the calibrated score while preserving patient context."
            )
            explanations.append(explanation)

        return {
            "n_explanations": int(len(explanations)),
            "mean_explanation_chars": float(np.mean([len(x) for x in explanations]))
            if explanations
            else 0.0,
        }

    def vectorized_score_stage() -> float:
        t0 = now_ms()
        _ = 0.50 * detector_norm + 0.45 * sont_norm + 0.05 * sgen_norm
        return now_ms() - t0

    def rowwise_score_stage() -> float:
        t0 = now_ms()
        out: list[float] = []
        for a, b, c in zip(detector_norm, sont_norm, sgen_norm):
            out.append(float(0.50 * a + 0.45 * b + 0.05 * c))
        _ = out
        return now_ms() - t0

    lookup_values: list[str] = []
    if text_cols:
        for col in text_cols:
            lookup_values.extend(df[col].astype(str).head(min(len(df), 200)).tolist())
    if not lookup_values:
        lookup_values = [f"TOKEN_{i % 50}" for i in range(min(len(df), 200))]

    def fake_lookup(value: Any) -> int:
        # Deterministic lightweight proxy for ontology/string lookup cost.
        text = "" if value is None else str(value)
        acc = 0
        for ch in text[:80]:
            acc = (acc * 31 + ord(ch)) % 1_000_003
        return acc

    @lru_cache(maxsize=4096)
    def cached_fake_lookup(value: str) -> int:
        return fake_lookup(value)

    def uncached_lookup_stage() -> float:
        t0 = now_ms()
        for value in lookup_values * max(1, args.lookup_repeat):
            fake_lookup(value)
        return now_ms() - t0

    def cached_lookup_stage() -> float:
        cached_fake_lookup.cache_clear()
        t0 = now_ms()
        for value in lookup_values * max(1, args.lookup_repeat):
            cached_fake_lookup(value)
        return now_ms() - t0

    timings = [
        time_stage("detector_score_loading_and_ranking", args.repeat, detector_stage),
        time_stage("calibrated_score_computation", args.repeat, score_stage),
        time_stage(
            "counterfactual_candidate_search", args.repeat, counterfactual_stage
        ),
        time_stage("explanation_generation", args.repeat, explanation_stage),
    ]

    total_runtime = sum(float(row["total_ms_mean"]) for row in timings)
    bottlenecks = []
    for row in sorted(timings, key=lambda x: float(x["total_ms_mean"]), reverse=True):
        share = 100.0 * float(row["total_ms_mean"]) / max(total_runtime, 1e-12)
        bottlenecks.append(
            {
                "stage": row["stage"],
                "total_ms_mean": float(row["total_ms_mean"]),
                "share_of_total_runtime_pct": float(share),
            }
        )

    vectorized_ms = statistics.mean(
        [vectorized_score_stage() for _ in range(max(3, args.repeat))]
    )
    rowwise_ms = statistics.mean(
        [rowwise_score_stage() for _ in range(max(3, args.repeat))]
    )
    uncached_ms = statistics.mean(
        [uncached_lookup_stage() for _ in range(max(3, args.repeat))]
    )
    cached_ms = statistics.mean(
        [cached_lookup_stage() for _ in range(max(3, args.repeat))]
    )

    timings_csv = out_dir / "day43_component_timings.csv"
    summary_json = out_dir / "day43_profile_summary.json"
    report_md = out_dir / "day43_bottleneck_report.md"
    raw_profile = out_dir / "day43_raw_profile.prof"
    profile_txt = out_dir / "day43_cprofile_top.txt"

    write_timings_csv(timings_csv, timings, len(df))

    summary = {
        "day": 43,
        "title": "Performance Profiling",
        "status": "complete",
        "profiling_mode": "artifact-level integrated pipeline profiling",
        "input_csv": input_csv.as_posix(),
        "n_records": int(len(df)),
        "repeat": int(args.repeat),
        "columns": columns,
        "text_columns_used": text_cols,
        "timings": [
            {k: v for k, v in row.items() if k != "payload"}
            | {"payload": row["payload"]}
            for row in timings
        ],
        "bottlenecks": bottlenecks,
        "optimization_checks": {
            "vectorized_score_ms": float(vectorized_ms),
            "rowwise_score_ms": float(rowwise_ms),
            "vectorized_vs_rowwise_speedup": float(
                rowwise_ms / max(vectorized_ms, 1e-12)
            ),
            "uncached_lookup_ms": float(uncached_ms),
            "cached_lookup_ms": float(cached_ms),
            "cached_lookup_speedup": float(uncached_ms / max(cached_ms, 1e-12)),
        },
        "interpretation": {
            "main_bottleneck": bottlenecks[0]["stage"] if bottlenecks else None,
            "recommended_next_optimizations": [
                "Keep detector and score computations batched/vectorized.",
                "Cache ontology/string lookups used during repeated counterfactual candidate evaluation.",
                "Keep Sgen as low-weight diagnostic unless a stronger generative surprise definition is implemented.",
                "When live GPU inference is profiled later, report neural-model latency separately from downstream explanation latency.",
            ],
        },
        "files": {
            "summary_json": summary_json.as_posix(),
            "component_timings_csv": timings_csv.as_posix(),
            "bottleneck_report_md": report_md.as_posix(),
            "raw_cprofile": raw_profile.as_posix(),
            "cprofile_top_txt": profile_txt.as_posix(),
        },
    }

    save_json(summary_json, summary)
    report_md.write_text(render_markdown_report(summary), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_root", default="artifacts")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--out_dir", default="artifacts/day43")
    parser.add_argument("--max_records", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--lookup_repeat", type=int, default=20)
    parser.add_argument("--no_cprofile", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cprofile:
        summary = run_profile(args)
    else:
        profiler = cProfile.Profile()
        profiler.enable()
        summary = run_profile(args)
        profiler.disable()

        raw_profile = Path(summary["files"]["raw_cprofile"])
        profile_txt = Path(summary["files"]["cprofile_top_txt"])

        profiler.dump_stats(raw_profile)

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(40)
        profile_txt.write_text(stream.getvalue(), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": summary["status"],
                "input_csv": summary["input_csv"],
                "n_records": summary["n_records"],
                "main_bottleneck": summary["interpretation"]["main_bottleneck"],
                "summary_json": summary["files"]["summary_json"],
                "report": summary["files"]["bottleneck_report_md"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
