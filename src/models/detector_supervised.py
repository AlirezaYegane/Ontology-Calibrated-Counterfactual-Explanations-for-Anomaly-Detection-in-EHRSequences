from __future__ import annotations

import torch
from torch import nn


class GRUSequenceBinaryClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 160,
        hidden_dim: int = 320,
        num_layers: int = 2,
        dropout: float = 0.30,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        last_hidden = h_n[-1]
        x = self.norm(last_hidden)
        x = self.dropout(x)
        logits = self.head(x).squeeze(-1)
        return logits
