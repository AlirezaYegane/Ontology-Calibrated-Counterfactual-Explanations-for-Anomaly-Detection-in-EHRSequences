from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_day40_ablation_framework_runs_on_toy_scores(tmp_path: Path) -> None:
    scores = tmp_path / "toy_scores.csv"
    out_dir = tmp_path / "day40_out"

    df = pd.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1],
            "detector_score": [0.10, 0.20, 0.25, 0.70, 0.80, 0.90],
            "ontology_score": [0.00, 0.00, 0.10, 0.70, 0.80, 1.00],
            "generative_score": [0.20, 0.30, 0.20, 0.40, 0.45, 0.50],
            "sequence_length": [10, 11, 10, 20, 22, 21],
            "anomaly_type": [
                "",
                "",
                "",
                "missing_diagnosis",
                "forbidden_pair",
                "demographic",
            ],
        }
    )
    df.to_csv(scores, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_day40_ablation_framework.py",
            "--input_scores",
            str(scores),
            "--out_dir",
            str(out_dir),
            "--allow_vae_proxy",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "full_model_conservative" in result.stdout
    assert (out_dir / "ablation_results.csv").exists()
    assert (out_dir / "variant_scores.csv").exists()
    assert (out_dir / "day40_ablation_summary.json").exists()
    assert (out_dir / "day40_ablation_report.md").exists()

    results = pd.read_csv(out_dir / "ablation_results.csv")
    assert {"variant", "roc_auc", "average_precision", "f1"}.issubset(results.columns)
    assert "no_ontology" in set(results["variant"])
    assert "no_generative" in set(results["variant"])
    assert "vae_replacement_slot" in set(results["variant"])
