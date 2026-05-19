from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_day45_module():
    script_path = Path("scripts/evaluate_day45_test_set.py")
    spec = importlib.util.spec_from_file_location("day45_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_day45_evaluation_writes_metrics(tmp_path: Path) -> None:
    module = _load_day45_module()

    input_path = tmp_path / "scores.csv"
    out_dir = tmp_path / "day45"

    df = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 5, 6],
            "scal": [0.05, 0.10, 0.20, 0.75, 0.90, 0.95],
            "label": [0, 0, 0, 1, 1, 1],
            "anomaly_type": [
                "normal",
                "normal",
                "normal",
                "missing_diagnosis",
                "medication_mismatch",
                "demographic_conflict",
            ],
        }
    )
    df.to_csv(input_path, index=False)

    exit_code = module.main(
        [
            "--project_root",
            ".",
            "--input_scores",
            str(input_path),
            "--out_dir",
            str(out_dir),
            "--score_col",
            "scal",
            "--label_col",
            "label",
            "--type_col",
            "anomaly_type",
            "--max_fpr",
            "0.10",
            "--min_precision",
            "0.80",
        ]
    )

    assert exit_code == 0

    metrics_path = out_dir / "day45_test_set_metrics.json"
    assert metrics_path.exists()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["n_records"] == 6
    assert payload["global_metrics"]["roc_auc"] == 1.0
    assert payload["selected_threshold"]["precision"] >= 0.8

    assert (out_dir / "threshold_sensitivity.csv").exists()
    assert (out_dir / "error_analysis_by_type.csv").exists()
    assert (out_dir / "README.md").exists()
