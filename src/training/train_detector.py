"""
src/training/train_detector.py
===============================
Training script for the GRU anomaly detector (Days 15-16).

Trains a next-token prediction model on ontology-mapped clinical code
sequences.  Supports early stopping on validation loss, gradient clipping,
and checkpoint saving.

CLI::

    python -m src.training.train_detector --config config/detector.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.detector import AnomalyDetectorGRU
from src.utils.vocab import (
    BOS,
    EOS,
    PAD,
    UNK,
    build_vocab,
    encode_sequence,
    load_vocab,
    save_vocab,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EHRSequenceDataset(Dataset):
    """Dataset that loads sequences from pickle/parquet and encodes them.

    Each item is a list of integer token indices with BOS prepended and
    EOS appended, truncated to ``max_len``.
    """

    def __init__(
        self,
        path: Path,
        vocab: dict[str, int],
        code_col: str = "codes_ont",
        max_len: int = 256,
    ) -> None:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_pickle(path)

        self.sequences: list[list[int]] = []
        for tokens in df[code_col]:
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            if not isinstance(tokens, list) or len(tokens) == 0:
                continue
            # Encode, then prepend BOS and append EOS
            encoded = encode_sequence(tokens, vocab, max_len=max_len - 2)
            seq = [BOS] + encoded + [EOS]
            self.sequences.append(seq)

        log.info(
            "Loaded %d sequences from %s (max_len=%d)",
            len(self.sequences), path.name, max_len,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.sequences[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """Pad sequences to the maximum length in the batch.

    Parameters
    ----------
    batch:
        List of 1-D integer tensors of varying length.

    Returns
    -------
    Tensor of shape ``(B, L_max)`` padded with ``PAD``.
    """
    max_len = max(seq.size(0) for seq in batch)
    padded = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : seq.size(0)] = seq
    return padded


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: AnomalyDetectorGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
) -> float:
    """Train for one epoch, returning mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(batch, pad_idx=PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: AnomalyDetectorGRU,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        loss = model.compute_loss(batch, pad_idx=PAD)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 15-16 -- Train GRU anomaly detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True, type=Path,
        help="Path to YAML config file (e.g. config/detector.yaml)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # --- logging ---
    log_dir = Path(cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "training.log", encoding="utf-8"),
        ],
    )

    # --- config ---
    m_cfg = cfg["model"]
    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    o_cfg = cfg["output"]

    seed = t_cfg["seed"]
    set_seed(seed)
    log.info("Seed: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- vocab ---
    vocab_path = Path(d_cfg["vocab_path"])
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
    else:
        log.info("Vocab file not found -- building from training data")
        vocab = build_vocab(
            Path(d_cfg["train_pkl"]),
            code_col=d_cfg["code_col"],
            min_count=5,
        )
        save_vocab(vocab, vocab_path)

    vocab_size = len(vocab)
    log.info("Vocab size: %d", vocab_size)

    # --- datasets ---
    max_seq_len = t_cfg["max_seq_len"]
    code_col = d_cfg["code_col"]

    train_ds = EHRSequenceDataset(
        Path(d_cfg["train_pkl"]), vocab, code_col=code_col, max_len=max_seq_len,
    )
    val_ds = EHRSequenceDataset(
        Path(d_cfg["val_pkl"]), vocab, code_col=code_col, max_len=max_seq_len,
    )

    batch_size = t_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    log.info("Train: %d sequences, Val: %d sequences", len(train_ds), len(val_ds))

    # --- model ---
    model = AnomalyDetectorGRU(
        vocab_size=vocab_size,
        embed_dim=m_cfg["embed_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        dropout=m_cfg["dropout"],
        pad_idx=PAD,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    # --- training loop ---
    ckpt_dir = Path(o_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = t_cfg["epochs"]
    patience = t_cfg["patience"]
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    log.info("Starting training: %d epochs, patience=%d", epochs, patience)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        log.info(
            "Epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  (%.1fs)",
            epoch, epochs, train_loss, val_loss, elapsed,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "elapsed_s": round(elapsed, 2),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = ckpt_dir / "detector_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "config": cfg,
            }, best_path)
            log.info("  -> Saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # --- save history ---
    history_path = log_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info("Training history saved to %s", history_path)
    log.info("Best val_loss: %.4f", best_val_loss)
    log.info("Done.")


if __name__ == "__main__":
    main()
