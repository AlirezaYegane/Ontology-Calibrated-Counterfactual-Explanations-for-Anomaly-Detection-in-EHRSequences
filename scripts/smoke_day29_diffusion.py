from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.models.diffusion import DiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact_path",
        default="artifacts/day27/mimiciv_val_diffusion.pt",
    )
    parser.add_argument(
        "--summary_path",
        default="artifacts/day27/mimiciv_val_diffusion_summary.json",
    )
    parser.add_argument(
        "--out_path",
        default="artifacts/day29/day29_smoke_summary.json",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_len", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_tensor(payload: Any, candidate_keys: list[str]) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload

    if isinstance(payload, dict):
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value

        for nested_key in ("tensors", "data", "dataset", "arrays"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                for key in candidate_keys:
                    value = nested.get(key)
                    if isinstance(value, torch.Tensor):
                        return value

    raise KeyError(
        f"Could not find any tensor using keys={candidate_keys}. "
        f"Payload type={type(payload)}"
    )


def infer_vocab_size(input_ids: torch.Tensor, summary: dict[str, Any]) -> int:
    from_summary = summary.get("vocab_size")
    if isinstance(from_summary, int) and from_summary > 2:
        return from_summary

    max_id = int(input_ids.max().item())
    return max(max_id + 1, 3)


def main() -> None:
    args = parse_args()

    artifact_path = Path(args.artifact_path)
    summary_path = Path(args.summary_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing Day 27 artifact: {artifact_path}")

    payload = torch.load(artifact_path, map_location="cpu")
    summary = load_json(summary_path)

    input_ids = find_tensor(
        payload,
        candidate_keys=[
            "input_ids",
            "sequence_input_ids",
            "sequences",
            "ids",
            "x",
        ],
    ).long()

    try:
        attention_mask = find_tensor(
            payload,
            candidate_keys=[
                "attention_mask",
                "mask",
                "sequence_attention_mask",
            ],
        ).long()
    except KeyError:
        attention_mask = (input_ids != 0).long()

    if input_ids.ndim != 2:
        raise ValueError(
            f"Expected input_ids shape [N, L], got {tuple(input_ids.shape)}"
        )

    batch_size = min(args.batch_size, input_ids.shape[0])
    input_ids = input_ids[:batch_size]
    attention_mask = attention_mask[:batch_size]

    max_len = int(input_ids.shape[1])
    vocab_size = infer_vocab_size(input_ids, summary)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    model = DiffusionModel(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        ff_dim=512,
        dropout=0.10,
        num_diffusion_steps=64,
        pad_idx=0,
    ).to(device)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.train()
    loss = model.training_loss(input_ids=input_ids, attention_mask=attention_mask)

    model.eval()
    with torch.no_grad():
        scores = model.surprise_score(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sample_len = min(args.sample_len, max_len)
        sampled_embeddings, sampled_ids = model.sample(
            batch_size=2,
            seq_len=sample_len,
            device=device,
        )

    result = {
        "status": "ok",
        "artifact_path": str(artifact_path),
        "summary_path": str(summary_path),
        "device_used": str(device),
        "input_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "vocab_size": int(vocab_size),
        "max_len": int(max_len),
        "model": {
            "type": "DiffusionModel",
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "ff_dim": 512,
            "num_diffusion_steps": 64,
            "prediction_target": "epsilon",
        },
        "training_loss": float(loss.detach().cpu().item()),
        "surprise_score_shape": list(scores.shape),
        "surprise_score_mean": float(scores.detach().cpu().mean().item()),
        "sampled_embeddings_shape": list(sampled_embeddings.shape),
        "sampled_ids_shape": list(sampled_ids.shape),
    }

    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
