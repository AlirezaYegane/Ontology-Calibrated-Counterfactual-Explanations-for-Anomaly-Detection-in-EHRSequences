"""
src/experiments/tracking.py
===========================
Phase 6 -- Lightweight experiment index.

Maintains ``artifacts/phase6/experiment_index.{json,md}`` listing every Phase 6
run with aggregate metadata + metrics ONLY (no PHI / no per-record data). Safe to
commit.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "artifacts" / "phase6"

# Keys that may NEVER appear in an index entry (defensive PHI guard).
_FORBIDDEN_KEYS = frozenset(
    {"per_record", "per_record_scores", "sequences", "codes", "vocab", "subject_id"}
)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_no_phi(entry: dict[str, Any]) -> None:
    for k in entry:
        if str(k).lower() in _FORBIDDEN_KEYS:
            raise ValueError(f"experiment index entry must not contain PHI key {k!r}")


def record_run(entry: dict[str, Any], index_dir: Path = INDEX_DIR) -> Path:
    """Append/update a run entry (by run_id) and rewrite the index JSON + MD."""
    _assert_no_phi(entry)
    index_dir.mkdir(parents=True, exist_ok=True)
    json_path = index_dir / "experiment_index.json"
    runs: list[dict[str, Any]] = []
    if json_path.exists():
        try:
            runs = json.loads(json_path.read_text(encoding="utf-8")).get("runs", [])
        except Exception:
            runs = []
    runs = [r for r in runs if r.get("run_id") != entry.get("run_id")]
    runs.append(entry)
    runs.sort(key=lambda r: str(r.get("timestamp", "")))

    payload = {"phase": 6, "n_runs": len(runs), "runs": runs}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(runs, index_dir / "experiment_index.md")
    return json_path


def _fmt_metric(d: Any, key: str) -> str:
    if isinstance(d, dict) and key in d and d[key] is not None:
        return str(d[key])
    return "-"


def _write_md(runs: list[dict[str, Any]], path: Path) -> None:
    md = [
        "# Phase 6 -- Experiment Index\n",
        f"{len(runs)} run(s). Aggregate metadata + metrics only (no PHI / per-record data).\n",
        "| run_id | timestamp | commit | device | evidence | epochs | best_ep | "
        "det ROC-AUC | comb ROC-AUC | status |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in runs:
        det = r.get("detector_metrics") or {}
        comb = r.get("combined_metrics") or {}
        md.append(
            f"| {r.get('run_id', '-')} | {r.get('timestamp', '-')} | "
            f"{r.get('git_commit', '-')} | {r.get('device', '-')} | "
            f"{r.get('evidence_level', '-')} | {r.get('epochs_completed', '-')} | "
            f"{r.get('best_epoch', '-')} | {_fmt_metric(det, 'roc_auc')} | "
            f"{_fmt_metric(comb, 'combined_real_without_sgen')} | {r.get('status', '-')} |"
        )
    path.write_text("\n".join(md), encoding="utf-8")


__all__ = ["record_run", "git_commit", "now_iso", "INDEX_DIR"]
