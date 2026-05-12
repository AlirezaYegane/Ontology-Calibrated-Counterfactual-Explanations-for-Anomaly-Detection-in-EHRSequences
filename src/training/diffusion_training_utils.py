from __future__ import annotations

import inspect
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_summary(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_tensor_in_mapping(payload: dict[str, Any], candidate_keys: list[str]) -> torch.Tensor | None:
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    for key in ("tensors", "data", "dataset", "arrays"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            found = _find_tensor_in_mapping(nested, candidate_keys)
            if found is not None:
                return found
    return None


def load_diffusion_artifact(
    path: str | Path,
    *,
    pad_idx: int = 0,
    max_records: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")

    metadata: dict[str, Any] = {"artifact_path": str(path)}

    if isinstance(payload, torch.Tensor):
        input_ids = payload.long()
        attention_mask = (input_ids != pad_idx).long()
        metadata["artifact_type"] = "tensor"
    elif isinstance(payload, dict):
        metadata["artifact_type"] = "dict"
        metadata["artifact_keys"] = sorted(str(k) for k in payload.keys())

        input_ids = _find_tensor_in_mapping(
            payload,
            [
                "input_ids",
                "sequence_input_ids",
                "sequences",
                "sequence_ids",
                "token_ids",
                "ids",
                "x",
            ],
        )
        if input_ids is None:
            tensor_candidates = {
                k: tuple(v.shape)
                for k, v in payload.items()
                if isinstance(v, torch.Tensor)
            }
            raise KeyError(
                "Could not find sequence input ids in diffusion artifact. "
                f"Tensor candidates: {tensor_candidates}"
            )

        input_ids = input_ids.long()

        attention_mask = _find_tensor_in_mapping(
            payload,
            [
                "attention_mask",
                "mask",
                "sequence_mask",
                "valid_mask",
            ],
        )
        if attention_mask is None:
            attention_mask = (input_ids != pad_idx).long()
        else:
            attention_mask = attention_mask.long()

        for key in ("vocab_size", "max_len", "pad_idx", "unk_idx", "loader_view"):
            if key in payload and not isinstance(payload[key], torch.Tensor):
                metadata[key] = payload[key]

        if "vocab" in payload and isinstance(payload["vocab"], dict):
            metadata["vocab_size"] = len(payload["vocab"])
    else:
        raise TypeError(f"Unsupported artifact payload type: {type(payload)}")

    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids shape [N, L], got {tuple(input_ids.shape)}")

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )

    if max_records is not None and max_records > 0:
        input_ids = input_ids[:max_records].contiguous()
        attention_mask = attention_mask[:max_records].contiguous()

    metadata["num_records"] = int(input_ids.shape[0])
    metadata["max_len"] = int(input_ids.shape[1])
    metadata["observed_vocab_size"] = int(input_ids.max().item()) + 1 if input_ids.numel() else 0
    metadata["nonpad_tokens"] = int(attention_mask.sum().item())
    metadata["mean_sequence_length"] = float(attention_mask.sum(dim=1).float().mean().item())

    return input_ids, attention_mask, metadata


class DiffusionSequenceDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids.long()
        self.attention_mask = attention_mask.long()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 1e-5, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


def build_diffusion_schedule(timesteps: int, schedule: str = "cosine") -> dict[str, torch.Tensor]:
    if schedule == "cosine":
        betas = cosine_beta_schedule(timesteps)
    elif schedule == "linear":
        betas = linear_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unsupported beta schedule: {schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def extract_by_timestep(values: torch.Tensor, t: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, t)
    while out.ndim < len(broadcast_shape):
        out = out.unsqueeze(-1)
    return out


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_alpha_bar = extract_by_timestep(
        schedule["sqrt_alphas_cumprod"],
        t,
        x_start.shape,
    )
    sqrt_one_minus_alpha_bar = extract_by_timestep(
        schedule["sqrt_one_minus_alphas_cumprod"],
        t,
        x_start.shape,
    )
    return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}"
        )

    mask = attention_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    sq = (prediction - target) ** 2
    denom = torch.clamp(mask.sum() * prediction.shape[-1], min=1.0)
    return (sq * mask).sum() / denom


def instantiate_diffusion_model(
    *,
    vocab_size: int,
    max_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ff_dim: int,
    dropout: float,
    diffusion_steps: int,
    pad_idx: int,
) -> nn.Module:
    from src.models.diffusion import DiffusionModel

    kwargs = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_heads": n_heads,
        "n_layers": n_layers,
        "num_layers": n_layers,
        "ff_dim": ff_dim,
        "dim_feedforward": ff_dim,
        "dropout": dropout,
        "timesteps": diffusion_steps,
        "diffusion_steps": diffusion_steps,
        "num_steps": diffusion_steps,
        "pad_idx": pad_idx,
    }

    sig = inspect.signature(DiffusionModel)
    accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    if accepts_var_kw:
        try:
            return DiffusionModel(**kwargs)
        except TypeError:
            pass

    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    try:
        return DiffusionModel(**filtered)
    except TypeError as exc:
        minimal_attempts = [
            {"vocab_size": vocab_size, "max_len": max_len},
            {"vocab_size": vocab_size},
            {},
        ]
        for attempt in minimal_attempts:
            try:
                return DiffusionModel(**attempt)
            except TypeError:
                continue
        raise TypeError(
            "Could not instantiate src.models.diffusion.DiffusionModel. "
            f"Signature: {sig}. First error: {exc}"
        ) from exc


def find_embedding_layer(model: nn.Module) -> nn.Embedding:
    candidate_names = [
        "token_embedding",
        "embedding",
        "embed",
        "code_embedding",
        "input_embedding",
        "embeddings",
    ]

    for name in candidate_names:
        layer = getattr(model, name, None)
        if isinstance(layer, nn.Embedding):
            return layer

    for _, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return module

    raise AttributeError(
        "Could not find an nn.Embedding layer inside DiffusionModel. "
        "Day 30 training expects the Day 29 model to contain a token/code embedding layer."
    )


def forward_diffusion_model(
    model: nn.Module,
    x_noisy: torch.Tensor,
    t: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    attempts = [
        lambda: model(x_noisy, t, attention_mask=attention_mask),
        lambda: model(x_noisy, t, mask=attention_mask),
        lambda: model(x_noisy, t, key_padding_mask=(attention_mask == 0)),
        lambda: model(x_noisy, t),
    ]

    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            out = attempt()
            if isinstance(out, dict):
                for key in ("pred_noise", "prediction", "noise", "out", "x"):
                    if key in out:
                        return out[key]
                raise KeyError(f"Model returned dict without known prediction keys: {list(out.keys())}")
            if isinstance(out, tuple):
                return out[0]
            return out
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"All DiffusionModel forward call patterns failed. Last error: {last_exc}") from last_exc
