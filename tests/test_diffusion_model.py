from __future__ import annotations

import torch

from src.models.diffusion import DiffusionModel, cosine_beta_schedule


def test_cosine_beta_schedule_shape_and_range() -> None:
    betas = cosine_beta_schedule(8)

    assert betas.shape == (8,)
    assert torch.all(betas > 0)
    assert torch.all(betas < 1)


def test_diffusion_forward_and_training_loss_shapes() -> None:
    model = DiffusionModel(
        vocab_size=32,
        max_len=12,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        num_diffusion_steps=8,
        pad_idx=0,
    )

    input_ids = torch.tensor(
        [
            [2, 3, 4, 5, 0, 0],
            [6, 7, 8, 9, 10, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()

    x_start = model.embed_tokens(input_ids)
    timesteps = torch.tensor([1, 3], dtype=torch.long)
    noise = torch.randn_like(x_start)
    x_noisy = model.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)

    pred_noise = model(
        x_noisy=x_noisy,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    loss = model.training_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timesteps=timesteps,
        noise=noise,
    )

    assert pred_noise.shape == x_start.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_surprise_score_and_sampling_shapes() -> None:
    model = DiffusionModel(
        vocab_size=32,
        max_len=12,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        num_diffusion_steps=8,
        pad_idx=0,
    )

    input_ids = torch.tensor(
        [
            [2, 3, 4, 0, 0],
            [5, 6, 7, 8, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()

    scores = model.surprise_score(input_ids=input_ids, attention_mask=attention_mask)
    sampled_embeddings, sampled_ids = model.sample(batch_size=2, seq_len=5)

    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))
    assert sampled_embeddings.shape == (2, 5, 16)
    assert sampled_ids.shape == (2, 5)
    assert sampled_ids.dtype == torch.long
