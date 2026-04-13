"""
Day 13 edge-case summary report generation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _day13_audit_lib import build_audit_report, build_edge_case_summary, write_json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for edge-case reporting."""
    parser = argparse.ArgumentParser(description="Day 13 edge-case summary report")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed dataset artifacts",
    )
    parser.add_argument(
        "--out",
        default="artifacts/day13/edge_case_report.json",
        help="Path to edge-case JSON output",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 1 if critical issues are found",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=2000,
        help="Maximum number of records to inspect per file for large artifacts",
    )
    return parser.parse_args()


def main() -> int:
    """Generate and write the Day 13 edge-case JSON report."""
    args = parse_args()

    report = build_audit_report(
        Path(args.processed_dir),
        max_records_per_file=args.max_records_per_file,
    )
    edge_summary = build_edge_case_summary(report)
    out_path = Path(args.out)

    write_json(out_path, edge_summary)

    print(f"[day13] wrote edge-case summary to {out_path.as_posix()}")
    for key, value in sorted(edge_summary["edge_case_counts"].items()):
        print(f"[day13] {key}={value}")

    if args.fail_on_critical and edge_summary["critical_issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
