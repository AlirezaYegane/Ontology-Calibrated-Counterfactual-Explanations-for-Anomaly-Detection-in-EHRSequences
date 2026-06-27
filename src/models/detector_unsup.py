"""
src/models/detector_unsup.py
============================
Phase 3 -- Primary (unsupervised / self-supervised) anomaly detector.

This is the *scientific* detector path: a next-token GRU language model trained
on NORMAL sequences only, scoring records by mean negative log-likelihood. It
never uses synthetic anomaly labels during training, which removes the
circularity of the Day-20 supervised baseline (kept only as
``supervised_synthetic_baseline``).

The neural backbone is the existing :class:`AnomalyDetectorGRU`; this module adds
a thin, testable wrapper with tokenizer/encode, scoring, and save/load.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.models.detector import AnomalyDetectorGRU

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)


@dataclass(frozen=True)
class UnsupDetectorConfig:
    vocab_size: int
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    max_len: int = 256
    pad_idx: int = 0


def build_vocab(sequences: list[list[str]], min_count: int = 1) -> dict[str, int]:
    """Build a token->id vocab (specials first) from training sequences."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for seq in sequences:
        counts.update(seq)
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIALS)}
    for tok, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_count and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


class UnsupervisedSequenceDetector:
    """Wrapper around :class:`AnomalyDetectorGRU` for unsupervised anomaly scoring."""

    def __init__(
        self, config: UnsupDetectorConfig, vocab: dict[str, int], device: str = "cpu"
    ) -> None:
        self.config = config
        self.vocab = vocab
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.model = AnomalyDetectorGRU(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pad_idx=config.pad_idx,
        ).to(self.device)
        self.unk_idx = vocab.get(UNK, 1)
        self.bos_idx = vocab.get(BOS, 2)
        self.eos_idx = vocab.get(EOS, 3)

    def encode(self, tokens: list[str]) -> list[int]:
        toks = list(tokens)[: self.config.max_len - 2]
        ids = (
            [self.bos_idx]
            + [self.vocab.get(t, self.unk_idx) for t in toks]
            + [self.eos_idx]
        )
        return ids

    def _batch(self, sequences: list[list[str]]) -> torch.Tensor:
        encoded = [self.encode(s) for s in sequences]
        max_len = max((len(e) for e in encoded), default=2)
        padded = [e + [self.config.pad_idx] * (max_len - len(e)) for e in encoded]
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def anomaly_scores(
        self, sequences: list[list[str]], batch_size: int = 64
    ) -> list[float]:
        """Mean-NLL anomaly score per record (higher = more anomalous).

        Scored in mini-batches: a single padded batch over the whole set would
        materialise a ``[N, max_len, vocab]`` logits tensor (tens of GB for a
        full split). Per-record scores are invariant to batch composition
        because :meth:`AnomalyDetectorGRU.anomaly_score` masks padding and
        normalises by each sequence's non-pad length.
        """
        if not sequences:
            return []
        self.model.eval()
        out: list[float] = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            x = self._batch(chunk)
            scores = self.model.anomaly_score(x, pad_idx=self.config.pad_idx)
            out.extend(float(s) for s in scores.cpu().tolist())
        return out

    def train_step(
        self, sequences: list[list[str]], optimizer: torch.optim.Optimizer
    ) -> float:
        self.model.train()
        x = self._batch(sequences)
        optimizer.zero_grad()
        loss = self.model.compute_loss(x, pad_idx=self.config.pad_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return float(loss.item())

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "config": asdict(self.config)},
            out_dir / "detector_unsup.pt",
        )
        (out_dir / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(
        cls, out_dir: str | Path, device: str = "cpu"
    ) -> "UnsupervisedSequenceDetector":
        out_dir = Path(out_dir)
        ckpt = torch.load(out_dir / "detector_unsup.pt", map_location="cpu")
        vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
        cfg = UnsupDetectorConfig(**ckpt["config"])
        det = cls(cfg, vocab, device=device)
        det.model.load_state_dict(ckpt["model_state"])
        det.model.to(det.device)
        return det


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


__all__ = [
    "UnsupDetectorConfig",
    "UnsupervisedSequenceDetector",
    "build_vocab",
    "set_seed",
    "SPECIALS",
    "PAD",
    "UNK",
    "BOS",
    "EOS",
]
