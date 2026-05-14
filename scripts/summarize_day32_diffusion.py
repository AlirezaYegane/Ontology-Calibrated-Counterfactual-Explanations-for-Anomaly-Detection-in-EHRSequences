from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def find_tensor_payload(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, dict):
        preferred = [
            "input_ids",
            "sequence_input_ids",
            "token_ids",
            "ids",
            "x",
            "data",
            "sequences",
        ]
        for key in preferred:
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        for value in obj.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                return value

    raise ValueError("Could not find a tensor payload in the diffusion artifact.")


def tensor_stats(x: torch.Tensor, pad_idx: int = 0, max_rows: int | None = None) -> dict[str, Any]:
    if max_rows is not None:
        x = x[:max_rows]

    if x.ndim == 3:
        x = x.argmax(dim=-1)

    x = x.detach().cpu().long()

    lengths = []
    token_counter: Counter[int] = Counter()

    for row in x:
        tokens = [int(v) for v in row.tolist() if int(v) != pad_idx]
        lengths.append(len(tokens))
        token_counter.update(tokens)

    lengths_t = torch.tensor(lengths, dtype=torch.float32) if lengths else torch.tensor([0.0])

    return {
        "rows": int(x.shape[0]),
        "seq_len_width": int(x.shape[1]) if x.ndim >= 2 else None,
        "nonpad_length": {
            "mean": float(lengths_t.mean().item()),
            "median": float(lengths_t.median().item()),
            "min": int(min(lengths) if lengths else 0),
            "max": int(max(lengths) if lengths else 0),
        },
        "top_token_ids": [
            {"token_id": int(tok), "count": int(count)}
            for tok, count in token_counter.most_common(30)
        ],
        "unique_nonpad_tokens": int(len(token_counter)),
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "reason": "No metrics.jsonl found or file is empty.",
        }

    numeric_keys = []
    for key, value in rows[-1].items():
        if isinstance(value, (int, float)):
            numeric_keys.append(key)

    best_by_val_loss = None
    if any("val" in key and "loss" in key for key in numeric_keys):
        val_loss_keys = [key for key in numeric_keys if "val" in key and "loss" in key]
        key = val_loss_keys[0]
        best_by_val_loss = min(rows, key=lambda r: float(r.get(key, float("inf"))))

    loss_keys = [key for key in numeric_keys if "loss" in key.lower()]

    return {
        "available": True,
        "num_rows": len(rows),
        "first_row": rows[0],
        "last_row": rows[-1],
        "loss_keys_detected": loss_keys,
        "best_by_val_loss": best_by_val_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--sample_real_rows", type=int, default=5000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_path = Path(args.data_path)
    summary_path = Path(args.summary_path)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
    data_obj = torch.load(data_path, map_location="cpu")
    tensor = find_tensor_payload(data_obj)

    checkpoint_candidates = []
    for name in ["diffusion.pt", "best.pt", "last.pt"]:
        p = run_dir / name
        if p.exists():
            checkpoint_candidates.append(
                {
                    "name": name,
                    "path": str(p),
                    "size_bytes": int(p.stat().st_size),
                }
            )

    report = {
        "day": 32,
        "title": "Final baseline diffusion refinement",
        "status": "complete",
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "summary_path": str(summary_path),
        "data_summary": read_json(summary_path),
        "checkpoints": checkpoint_candidates,
        "training_metrics": summarize_metrics(metrics_rows),
        "real_data_statistics": tensor_stats(
            tensor,
            pad_idx=args.pad_idx,
            max_rows=args.sample_real_rows,
        ),
        "interpretation": {
            "what_this_day_confirms": [
                "The baseline diffusion model can be trained as a final non-ontology baseline.",
                "The final checkpoint is available for Day 33 ontology regularization.",
                "Real-data statistics are recorded for generated-sample comparison."
            ],
            "next_step": "Day 33 should add ontology regularization and compare violation rates against this baseline."
        }
    }

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
