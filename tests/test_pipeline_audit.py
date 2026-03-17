"""
Tests for the Day 13 Mapping Audit Pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _day13_audit_lib import build_audit_report, build_edge_case_summary, render_markdown


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Helper functional to quickly author dummy mock JSONL files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def test_build_audit_report_detects_mapping_and_edge_cases(tmp_path: Path) -> None:
    """Verify that end-to-end extraction tracks mapped vs unmapped namespaces correctly."""
    processed_dir = tmp_path / "data" / "processed"

    rows = [
        {
            "patient_id": 1,
            "gender": "M",
            "diagnoses": ["ICD9_4280", "UNMAPPED_DIAG_X", ""],
            "procedures": ["PROC_3893", "proc_bad"],
            "medications": ["RXNORM_860975", "RAW_DRUG:aspirin"],
            "sequence_tokens": ["ICD9_4280", "ICD9_4280", "bad token"],
        },
        {
            "patient_id": 2,
            "gender": "F",
            "diagnoses": ["SNOMED_123456"],
            "medications": ["RXNORM_111111"],
        },
        {
            "patient_id": 3,
            "age": 64,
            "diagnoses": [],
            "procedures": [],
            "medications": [],
        },
    ]
    write_jsonl(processed_dir / "train_sequences.jsonl", rows)

    report = build_audit_report(processed_dir)

    assert report["total_records"] == 3
    assert report["mapping_summary"]["diagnosis"]["total"] >= 2
    assert report["mapping_summary"]["diagnosis"]["unmapped"] >= 1
    assert report["mapping_summary"]["medication"]["unmapped"] >= 1
    assert report["edge_case_counts"]["empty_tokens"] >= 1
    assert report["edge_case_counts"]["duplicate_tokens_within_record"] >= 1
    assert report["edge_case_counts"]["malformed_tokens"] >= 1
    assert report["edge_case_counts"]["lowercase_namespace_tokens"] >= 1
    assert report["milestone1_ready"] is True

    md = render_markdown(report)
    assert "Mapping Summary" in md
    assert "Edge Case Counts" in md


def test_edge_case_summary_contains_expected_keys(tmp_path: Path) -> None:
    """Verify that build_edge_case_summary filters exactly logic-focused edge case mappings."""
    processed_dir = tmp_path / "data" / "processed"
    write_jsonl(
        processed_dir / "test_sequences.jsonl",
        [
            {
                "diagnoses": ["UNMAPPED_DIAG_Y"],
                "medications": ["RAW_DRUG:ibuprofen"],
                "sequence_tokens": ["weird token"],
            }
        ],
    )

    report = build_audit_report(processed_dir)
    summary = build_edge_case_summary(report)

    assert "edge_case_counts" in summary
    assert "critical_issues" in summary
    assert "warnings" in summary
    assert "top_unmapped" in summary
    assert summary["milestone1_ready"] is True


def test_empty_processed_dir_is_not_milestone_ready(tmp_path: Path) -> None:
    """Validates that a missing or entirely empty processed dir triggers critical issues."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    report = build_audit_report(processed_dir)

    assert report["milestone1_ready"] is False
    assert report["critical_issues"]


def test_stringified_token_lists_are_parsed_and_sequence_tokens_are_not_double_counted(
    tmp_path: Path,
) -> None:
    """Stringified arrays (often seen in dumped Parquet) are cleanly extracted, 
    and redundant seq token lists are ignored if they contain duplicates."""
    processed_dir = tmp_path / "data" / "processed"
    write_jsonl(
        processed_dir / "toy_sequences.jsonl",
        [
            {
                "gender": "F",
                "age_group": "45-64",
                "diagnosis_tokens": '["ICD9_DX:5723", "ICD10_DX:E119"]',
                "procedure_tokens": '["ICD9_PROC:5491"]',
                "medication_tokens": '["MED_NAME:ASPIRIN", "MED_NDC:12345678901"]',
                "sequence_tokens": '["ICD9_DX:5723", "ICD10_DX:E119", "ICD9_PROC:5491", "MED_NAME:ASPIRIN", "MED_NDC:12345678901"]',
                "stay_ids": "[]",
                "admittime": "2180-05-06 22:23:00",
                "dischtime": "2180-05-07 17:15:00",
                "sequence_length": 5,
            }
        ],
    )

    report = build_audit_report(processed_dir)

    assert report["mapping_summary"]["diagnosis"]["total"] == 2
    assert report["mapping_summary"]["procedure"]["total"] == 1
    assert report["mapping_summary"]["medication"]["total"] == 2
    assert report["mapping_summary"]["other"]["total"] == 0
    assert report["edge_case_counts"].get("malformed_tokens", 0) == 0
    assert report["edge_case_counts"].get("unknown_namespace_tokens", 0) == 0
