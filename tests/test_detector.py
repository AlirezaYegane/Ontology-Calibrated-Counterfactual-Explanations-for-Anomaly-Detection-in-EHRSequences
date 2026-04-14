"""
tests/test_detector.py
=======================
Unit tests for the GRU anomaly detector (Day 24).
"""

from __future__ import annotations

import torch
import pytest

from src.models.detector import AnomalyDetectorGRU


VOCAB_SIZE = 100
BATCH = 4
SEQ_LEN = 20


@pytest.fixture()
def model() -> AnomalyDetectorGRU:
    return AnomalyDetectorGRU(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        pad_idx=0,
    )


@pytest.fixture()
def sample_input() -> torch.Tensor:
    x = torch.randint(4, VOCAB_SIZE, (BATCH, SEQ_LEN))
    x[:, 0] = 2   # BOS
    x[:, -1] = 3  # EOS
    return x


def test_forward_shape(model: AnomalyDetectorGRU, sample_input: torch.Tensor) -> None:
    logits = model(sample_input)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)


def test_loss_runs(model: AnomalyDetectorGRU, sample_input: torch.Tensor) -> None:
    loss = model.compute_loss(sample_input, pad_idx=0)
    assert loss.dim() == 0
    assert loss.item() > 0
    # Verify backprop works
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0


def test_anomaly_score_shape(model: AnomalyDetectorGRU, sample_input: torch.Tensor) -> None:
    scores = model.anomaly_score(sample_input, pad_idx=0)
    assert scores.shape == (BATCH,)
    assert (scores > 0).all()


def test_short_train_no_crash(model: AnomalyDetectorGRU) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    losses: list[float] = []
    for _ in range(3):
        x = torch.randint(4, VOCAB_SIZE, (8, 15))
        optimizer.zero_grad()
        loss = model.compute_loss(x, pad_idx=0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # All losses should be finite positive numbers
    assert all(l > 0 and l < float("inf") for l in losses)
