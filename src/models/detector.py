"""
src/models/detector.py
=======================
GRU-based anomaly detector for clinical code sequences.

Uses next-token prediction as the unsupervised objective: the model
learns to predict the next clinical code given the prefix.  At inference
time the mean negative log-likelihood serves as the anomaly score --
sequences that are poorly predicted receive higher scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetectorGRU(nn.Module):
    """GRU autoregressive detector with next-token prediction objective.

    Parameters
    ----------
    vocab_size:
        Total number of tokens (including special tokens).
    embed_dim:
        Embedding dimension.
    hidden_dim:
        GRU hidden state dimension.
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout probability applied after the GRU output.
    pad_idx:
        Index of the padding token (masked in the loss).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input token indices, shape ``(B, L)``.

        Returns
        -------
        Logits of shape ``(B, L, V)`` where ``V = vocab_size``.
        """
        emb = self.embedding(x)          # (B, L, E)
        output, _ = self.gru(emb)        # (B, L, H)
        output = self.dropout(output)    # (B, L, H)
        logits = self.linear(output)     # (B, L, V)
        return logits

    def compute_loss(self, x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Compute next-token prediction loss.

        Predicts ``x[t+1]`` from the hidden state at position ``t``.
        Padding positions in the target are ignored.

        Parameters
        ----------
        x:
            Input token indices, shape ``(B, L)``.
        pad_idx:
            Padding index to ignore in the loss.

        Returns
        -------
        Scalar cross-entropy loss.
        """
        logits = self.forward(x[:, :-1])          # (B, L-1, V)
        targets = x[:, 1:]                        # (B, L-1)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=pad_idx,
        )
        return loss

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Compute per-sequence anomaly score (mean NLL).

        Parameters
        ----------
        x:
            Input token indices, shape ``(B, L)``.
        pad_idx:
            Padding index to exclude from the score.

        Returns
        -------
        Tensor of shape ``(B,)`` with the mean NLL per sequence.
        """
        self.eval()
        logits = self.forward(x[:, :-1])          # (B, L-1, V)
        targets = x[:, 1:]                        # (B, L-1)

        # Per-token NLL
        nll = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=pad_idx,
            reduction="none",
        ).reshape(targets.shape)                  # (B, L-1)

        # Mask padding positions
        mask = (targets != pad_idx).float()       # (B, L-1)
        lengths = mask.sum(dim=1).clamp(min=1)    # (B,)
        scores = (nll * mask).sum(dim=1) / lengths
        return scores
