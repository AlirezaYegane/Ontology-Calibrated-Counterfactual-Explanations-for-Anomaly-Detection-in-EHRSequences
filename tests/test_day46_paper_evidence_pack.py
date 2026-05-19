from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_day46_paper_evidence_pack_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_day46_paper_evidence_pack.py"

    scores = tmp_path / "scores.csv"
    out_dir = tmp_path / "day46"

    df = pd.DataFrame(
        {
            "y_true": [0, 0, 0, 1, 1, 1],
            "detector_only": [0.05, 0.10, 0.20, 0.70, 0.80, 0.90],
            "generative_only": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            "no_ontology": [0.10, 0.20, 0.30, 0.60, 0.70, 0.80],
        }
    )
    df.to_csv(scores, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--scores_csv",
            str(scores),
            "--out_dir",
            str(out_dir),
            "--n_boot",
            "10",
            "--seed",
            "42",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "best_variant" in result.stdout
    assert (out_dir / "day46_variant_metrics.csv").exists()
    assert (out_dir / "day46_pairwise_deltas.csv").exists()
    assert (out_dir / "day46_paper_results_table.md").exists()
    assert (out_dir / "day46_result_interpretation.md").exists()

def test_average_precision_constant_scores_equals_prevalence() -> None:
    import importlib.util
    import numpy as np
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_day46_paper_evidence_pack.py"

    spec = importlib.util.spec_from_file_location("day46_pack", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.zeros_like(y_true, dtype=float)

    assert module.average_precision_safe(y_true, y_score) == 0.5
