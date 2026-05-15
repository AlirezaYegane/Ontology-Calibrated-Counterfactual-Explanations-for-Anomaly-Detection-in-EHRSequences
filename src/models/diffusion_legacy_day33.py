from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        if half_dim <= 1:
            return t.float().view(-1, 1)

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = t.float().view(-1, 1) * emb.view(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if emb.shape[1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))

        return emb


class LegacyDay33DiffusionModel(nn.Module):
    """
    Exact compatibility wrapper for the Day 33 diffusion checkpoint.

    Checkpoint naming expected:
    - token_embedding.weight
    - pos_embedding.weight
    - time_embedding.1.*
    - time_embedding.3.*
    - encoder.layers.*
    - norm.*
    - out.*
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 256,
        pad_idx: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype in (torch.long, torch.int64, torch.int32):
            h = self.token_embedding(x.long())
        else:
            h = x.float()

        bsz, seq_len, _ = h.shape

        positions = torch.arange(seq_len, device=h.device).clamp(max=self.max_len - 1)
        h = h + self.pos_embedding(positions).unsqueeze(0)

        t_emb = self.time_embedding(t).unsqueeze(1)
        h = h + t_emb

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        return self.out(h)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        max_len: int | None = None,
        device: torch.device | str | None = None,
        steps: int = 16,
    ) -> torch.Tensor:
        device = torch.device(device) if device is not None else next(self.parameters()).device
        seq_len = max_len or self.max_len

        x = torch.randn(num_samples, seq_len, self.d_model, device=device)

        for step in reversed(range(steps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)
            pred_noise = self.forward(x, t, attention_mask=torch.ones(num_samples, seq_len, device=device, dtype=torch.bool))
            x = x - pred_noise / max(steps, 1)

        # Project embedding-space samples back to nearest vocabulary ids in chunks to avoid OOM.
        weight = self.token_embedding.weight.detach()
        rows = []
        chunk = 32

        for start in range(0, seq_len, chunk):
            part = x[:, start:start + chunk, :]
            logits = torch.matmul(part, weight.t())
            ids = logits.argmax(dim=-1)
            rows.append(ids.detach().cpu())

        return torch.cat(rows, dim=1).long()
