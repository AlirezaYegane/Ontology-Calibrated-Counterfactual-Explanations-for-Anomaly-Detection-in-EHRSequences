from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _day13_audit_lib import build_audit_report, render_markdown, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 13 end-to-end pipeline and mapping audit")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing processed dataset artifacts",
    )
    parser.add_argument(
        "--json-out",
        default="artifacts/day13/mapping_audit.json",
        help="Path to JSON audit output",
    )
    parser.add_argument(
        "--md-out",
        default="artifacts/day13/mapping_audit.md",
        help="Path to Markdown audit output",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 1 if any critical issues are found",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=2000,
        help="Maximum number of records to inspect per file for large artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)

    report = build_audit_report(
        processed_dir,
        max_records_per_file=args.max_records_per_file,
    )
    write_json(json_out, report)
    write_text(md_out, render_markdown(report))

    print(f"[day13] wrote JSON audit to {json_out.as_posix()}")
    print(f"[day13] wrote Markdown audit to {md_out.as_posix()}")
    print(f"[day13] milestone1_ready={report['milestone1_ready']}")
    print(f"[day13] critical_issues={len(report['critical_issues'])}")
    print(f"[day13] warnings={len(report['warnings'])}")

    if args.fail_on_critical and report["critical_issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
