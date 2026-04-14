"""
src/utils/vocab.py
===================
Vocabulary builder and sequence encoder for clinical code tokens.

Provides:
- ``build_vocab`` -- count tokens in a training pickle and build a
  token-to-index mapping with special tokens.
- ``save_vocab`` / ``load_vocab`` -- JSON serialisation.
- ``encode_sequence`` -- convert a list of string tokens to integer indices.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Special token indices (fixed)
PAD = 0
UNK = 1
BOS = 2
EOS = 3

_SPECIAL_TOKENS = {"<PAD>": PAD, "<UNK>": UNK, "<BOS>": BOS, "<EOS>": EOS}


# ---------------------------------------------------------------------------
# Vocab building
# ---------------------------------------------------------------------------


def build_vocab(
    train_pkl_path: Path | str,
    code_col: str = "codes_ont",
    min_count: int = 5,
) -> dict[str, int]:
    """Build a token vocabulary from a training dataset.

    Parameters
    ----------
    train_pkl_path:
        Path to a pickle or parquet file containing the training split.
    code_col:
        Column name holding token lists (list[str] or JSON-encoded).
    min_count:
        Minimum occurrence count for a token to be included.

    Returns
    -------
    dict mapping token strings to integer indices.  Special tokens
    ``<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3`` are always present.
    """
    path = Path(train_pkl_path)
    log.info("Building vocab from %s (col=%s, min_count=%d)", path.name, code_col, min_count)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)

    counter: Counter[str] = Counter()
    for tokens in df[code_col]:
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if isinstance(tokens, list):
            counter.update(tokens)

    # Filter by min_count and sort for determinism
    filtered = sorted(tok for tok, cnt in counter.items() if cnt >= min_count)

    vocab: dict[str, int] = dict(_SPECIAL_TOKENS)
    for tok in filtered:
        if tok not in vocab:
            vocab[tok] = len(vocab)

    log.info(
        "Vocab built: %d tokens (from %d unique, %d filtered by min_count=%d)",
        len(vocab), len(counter), len(counter) - len(filtered), min_count,
    )
    return vocab


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_vocab(vocab: dict[str, int], path: Path | str) -> None:
    """Save vocabulary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("Saved vocab (%d tokens) to %s", len(vocab), path.name)


def load_vocab(path: Path | str) -> dict[str, int]:
    """Load vocabulary from a JSON file."""
    path = Path(path)
    vocab: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    log.info("Loaded vocab (%d tokens) from %s", len(vocab), path.name)
    return vocab


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_sequence(
    codes: list[str],
    vocab: dict[str, int],
    max_len: int = 256,
) -> list[int]:
    """Encode a list of token strings to integer indices.

    Unknown tokens are mapped to ``UNK``.  The sequence is truncated to
    *max_len* if necessary.  No BOS/EOS is added here -- that is handled
    by the dataset class.
    """
    unk_idx = vocab.get("<UNK>", UNK)
    encoded = [vocab.get(tok, unk_idx) for tok in codes]
    return encoded[:max_len]
