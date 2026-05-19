from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "explain_day44_record.py"


def load_module():
    spec = importlib.util.spec_from_file_location("explain_day44_record", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_day44_build_explanation_from_minimal_score_row(tmp_path: Path) -> None:
    module = load_module()

    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "record_id": "case_001",
                "label": 1,
                "anomaly_type": "missing_diagnosis",
                "Sdet": 0.82,
                "Sont": 0.77,
                "Sgen": 0.51,
                "Scal": 0.86,
                "minimal_edits": "add diagnosis code for medication indication",
                "original_codes_preview": "med:insulin, lab:hba1c_high",
                "counterfactual_codes_preview": "dx:diabetes, med:insulin, lab:hba1c_high",
            }
        ]
    ).to_csv(csv_path, index=False)

    df = module.load_table(csv_path)
    row_idx, row = module.select_row(df, "case_001", 0)
    explanation = module.build_explanation(row, row_idx, csv_path)
    md = module.markdown_from_explanation(explanation, "Test Explanation")

    assert explanation["record_identifier"] == "case_001"
    assert explanation["risk_band"] == "high"
    assert explanation["score_components"]["Scal"] == 0.86
    assert "diagnostic only" in md
    assert "Counterfactual" in md


def test_day44_list_records_uses_available_columns(tmp_path: Path) -> None:
    module = load_module()

    df = pd.DataFrame(
        [
            {"record_id": "a", "label": 0, "Scal": 0.1},
            {"record_id": "b", "label": 1, "Scal": 0.9},
        ]
    )
    listed = module.list_records(df, top_k=2)

    assert "record_id" in listed
    assert "b" in listed
