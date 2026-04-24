from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionModelConfig:
    vocab_size: int
    max_len: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.10
    num_diffusion_steps: int = 64
    pad_idx: int = 0


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine DDPM beta schedule, clipped for numerical stability."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


def extract_timestep_values(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """Gather 1D diffusion buffers for batch timesteps and reshape for broadcasting."""
    batch_size = timesteps.shape[0]
    out = values.gather(0, timesteps.to(values.device))
    return out.view(batch_size, *((1,) * (len(target_shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding followed by a small projection MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.d_model // 2

        if half_dim == 0:
            raise ValueError(
                "d_model must be at least 2 for sinusoidal time embeddings."
            )

        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - 1, 1)

        args = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if emb.shape[1] < self.d_model:
            emb = torch.nn.functional.pad(emb, (0, self.d_model - emb.shape[1]))

        return self.proj(emb)


class DiffusionModel(nn.Module):
    """
    Lightweight DDPM-style diffusion model for padded EHR token sequences.

    The model diffuses continuous token embeddings, then uses a time-conditioned
    Transformer encoder to predict the injected noise epsilon.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.10,
        num_diffusion_steps: int = 64,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        if vocab_size <= 2:
            raise ValueError("vocab_size must be greater than 2.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.config = DiffusionModelConfig(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            num_diffusion_steps=num_diffusion_steps,
            pad_idx=pad_idx,
        )

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_diffusion_steps = num_diffusion_steps
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.time_embedding = SinusoidalTimeEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.denoiser = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        betas = cosine_beta_schedule(num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.pad_idx).long()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")
        if input_ids.shape[1] > self.max_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds max_len={self.max_len}."
            )
        return self.token_embedding(input_ids)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = extract_timestep_values(
            self.sqrt_alpha_cumprod,
            timesteps,
            x_start.shape,
        )
        sqrt_one_minus_alpha = extract_timestep_values(
            self.sqrt_one_minus_alpha_cumprod,
            timesteps,
            x_start.shape,
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def forward(
        self,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_noisy.ndim != 3:
            raise ValueError("x_noisy must have shape [batch, seq_len, d_model].")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch].")
        if x_noisy.shape[2] != self.d_model:
            raise ValueError(f"Last dimension must be d_model={self.d_model}.")

        batch_size, seq_len, _ = x_noisy.shape

        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.max_len}.")

        positions = torch.arange(seq_len, device=x_noisy.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        time_emb = self.time_embedding(timesteps).unsqueeze(1)

        hidden = x_noisy + pos_emb + time_emb

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.to(device=x_noisy.device) == 0

        hidden = self.denoiser(hidden, src_key_padding_mask=key_padding_mask)
        pred_noise = self.output_head(hidden)

        if pred_noise.shape != (batch_size, seq_len, self.d_model):
            raise RuntimeError("Unexpected denoiser output shape.")

        return pred_noise

    def training_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_start = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = self.make_attention_mask(input_ids)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.num_diffusion_steps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        pred_noise = self.forward(
            x_noisy=x_noisy,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        token_loss = (pred_noise - noise).pow(2).mean(dim=-1)
        mask = attention_mask.to(device=device, dtype=token_loss.dtype)

        return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def surprise_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Initial Sgen proxy: per-record mean non-pad denoising error.

        This is the operational Day 29 version. Later days can calibrate it
        against normal vs injected anomaly sets.
        """
        was_training = self.training
        self.eval()

        x_start = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = self.make_attention_mask(input_ids)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        if timesteps is None:
            midpoint = max(self.num_diffusion_steps // 2, 1)
            timesteps = torch.full(
                (batch_size,),
                fill_value=midpoint,
                device=device,
                dtype=torch.long,
            )

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        pred_noise = self.forward(
            x_noisy=x_noisy,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        token_error = (pred_noise - noise).pow(2).mean(dim=-1)
        mask = attention_mask.to(device=device, dtype=token_error.dtype)
        score = (token_error * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if was_training:
            self.train()

        return score

    @torch.no_grad()
    def decode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Nearest-token decoding via embedding-table logits.

        This is a simple baseline decoder for smoke tests and early sampling.
        A stronger constrained decoder can be added later.
        """
        logits = embeddings @ self.token_embedding.weight.t()
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate embedding samples and nearest-neighbour token ids.

        Returns:
            sampled_embeddings: [batch, seq_len, d_model]
            sampled_token_ids: [batch, seq_len]
        """
        was_training = self.training
        self.eval()

        if seq_len is None:
            seq_len = self.max_len
        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.max_len}.")

        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)

        x = torch.randn(batch_size, seq_len, self.d_model, device=device)
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.long, device=device
        )

        for step in reversed(range(self.num_diffusion_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            pred_noise = self.forward(
                x_noisy=x,
                timesteps=t,
                attention_mask=attention_mask,
            )

            beta_t = extract_timestep_values(self.betas, t, x.shape)
            sqrt_one_minus_alpha_bar_t = extract_timestep_values(
                self.sqrt_one_minus_alpha_cumprod,
                t,
                x.shape,
            )
            sqrt_recip_alpha_t = extract_timestep_values(
                self.sqrt_recip_alphas, t, x.shape
            )

            model_mean = sqrt_recip_alpha_t * (
                x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t.clamp_min(1e-8)
            )

            if step > 0:
                x = model_mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = model_mean

        token_ids = self.decode_embeddings(x)

        if was_training:
            self.train()

        return x, token_ids


__all__ = [
    "DiffusionModel",
    "DiffusionModelConfig",
    "SinusoidalTimeEmbedding",
    "cosine_beta_schedule",
]
