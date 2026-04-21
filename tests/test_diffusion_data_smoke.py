from __future__ import annotations

import torch

from src.training.diffusion_data_utils import (
    build_padded_tensors,
    build_vocab,
    make_diffusion_dataloader,
)


def test_diffusion_data_smoke() -> None:
    sequences = [
        ["ICD9_4019", "PROC_3893", "RXNORM_123"],
        ["ICD9_25000", "RXNORM_456"],
        ["ICD9_41401", "PROC_3601", "RXNORM_789", "RXNORM_123"],
    ]

    vocab = build_vocab(sequences)
    input_ids, attention_mask, lengths = build_padded_tensors(
        sequences=sequences,
        vocab=vocab,
        max_len=8,
        truncate_strategy="tail",
    )

    bundle = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "meta": {"vocab_size": len(vocab)},
    }

    loader = make_diffusion_dataloader(
        bundle=bundle,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        view="both",
    )

    batch = next(iter(loader))

    assert "input_ids" in batch
    assert "multi_hot" in batch
    assert "attention_mask" in batch
    assert "lengths" in batch

    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["lengths"].shape == (2,)
    assert batch["multi_hot"].shape[0] == 2
    assert batch["multi_hot"].shape[1] == len(vocab)

    assert batch["input_ids"].dtype == torch.int64
    assert batch["multi_hot"].dtype == torch.float32
