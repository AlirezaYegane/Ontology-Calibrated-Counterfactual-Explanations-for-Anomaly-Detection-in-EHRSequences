from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def test_day30_train_diffusion_smoke(tmp_path: Path) -> None:
    artifact_path = tmp_path / "toy_diffusion.pt"
    out_dir = tmp_path / "out"

    input_ids = torch.randint(low=2, high=32, size=(64, 24), dtype=torch.long)
    input_ids[:, -4:] = 0
    attention_mask = (input_ids != 0).long()

    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vocab_size": 32,
            "max_len": 24,
            "loader_view": "sequence",
        },
        artifact_path,
    )

    cmd = [
        sys.executable,
        "-m",
        "src.training.train_diffusion",
        "--data_path",
        str(artifact_path),
        "--out_dir",
        str(out_dir),
        "--epochs",
        "1",
        "--batch_size",
        "8",
        "--max_records",
        "32",
        "--max_steps_per_epoch",
        "2",
        "--diffusion_steps",
        "8",
        "--d_model",
        "64",
        "--n_heads",
        "4",
        "--n_layers",
        "1",
        "--ff_dim",
        "128",
        "--dropout",
        "0.1",
        "--device",
        "cpu",
        "--num_workers",
        "0",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["global_steps"] > 0
