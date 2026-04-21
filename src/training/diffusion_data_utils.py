from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

SEQ_COL_CANDIDATES = [
    "sequence_tokens",
    "codes",
    "sequence",
    "tokens",
    "event_codes",
    "concepts",
]

BAD_TOKEN_VALUES = {"", "nan", "none", "null", "na", "n/a"}


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_pickle(path: str | Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    raise TypeError(f"Unsupported pickle content type: {type(obj)}")


def infer_sequence_column(df: pd.DataFrame) -> str:
    for col in SEQ_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        sample = df[col].dropna().head(20)
        if len(sample) and any(isinstance(x, (list, tuple)) for x in sample):
            return col
    raise ValueError(f"Could not infer sequence column. Available columns: {list(df.columns)}")


def normalize_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            tok = str(item).strip()
            if tok.lower() not in BAD_TOKEN_VALUES:
                out.append(tok)
        return out

    if isinstance(value, str):
        text = value.strip()
        if text.lower() in BAD_TOKEN_VALUES:
            return []

        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                    if isinstance(parsed, list):
                        return normalize_tokens(parsed)
                except Exception:
                    pass

        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]

        return [part for part in text.split() if part.strip()]

    if value is None:
        return []

    token = str(value).strip()
    return [token] if token and token.lower() not in BAD_TOKEN_VALUES else []


def build_vocab(sequences: list[list[str]], min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for seq in sequences:
        counter.update(seq)

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_vocab(path: str | Path, vocab: dict[str, int]) -> None:
    save_json(path, vocab)


def encode_sequence(
    tokens: list[str],
    vocab: dict[str, int],
    max_len: int = 256,
    truncate_strategy: str = "tail",
) -> list[int]:
    if len(tokens) == 0:
        return [UNK_IDX]

    if len(tokens) > max_len:
        tokens = tokens[:max_len] if truncate_strategy == "head" else tokens[-max_len:]

    return [vocab.get(tok, UNK_IDX) for tok in tokens]


def build_padded_tensors(
    sequences: list[list[str]],
    vocab: dict[str, int],
    max_len: int = 256,
    truncate_strategy: str = "tail",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(sequences)
    input_ids = torch.zeros((n, max_len), dtype=torch.int32)
    attention_mask = torch.zeros((n, max_len), dtype=torch.bool)
    lengths = torch.zeros((n,), dtype=torch.int32)

    for i, seq in enumerate(sequences):
        ids = encode_sequence(
            tokens=seq,
            vocab=vocab,
            max_len=max_len,
            truncate_strategy=truncate_strategy,
        )
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.int32)
        attention_mask[i, :seq_len] = True
        lengths[i] = seq_len

    return input_ids, attention_mask, lengths


class DiffusionTensorDataset(Dataset):
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.input_ids = bundle["input_ids"]
        self.attention_mask = bundle["attention_mask"]
        self.lengths = bundle["lengths"]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "lengths": self.lengths[idx],
        }


def make_diffusion_collate(vocab_size: int, view: str = "both"):
    valid_views = {"token_ids", "multi_hot", "both"}
    if view not in valid_views:
        raise ValueError(f"view must be one of {valid_views}, got {view}")

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch]).long()
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).bool()
        lengths = torch.stack([item["lengths"] for item in batch]).long()

        out: dict[str, torch.Tensor] = {
            "attention_mask": attention_mask,
            "lengths": lengths,
        }

        if view in {"token_ids", "both"}:
            out["input_ids"] = input_ids

        if view in {"multi_hot", "both"}:
            multi_hot = torch.zeros((input_ids.size(0), vocab_size), dtype=torch.float32)
            for i in range(input_ids.size(0)):
                ids = input_ids[i][attention_mask[i]]
                ids = ids[(ids != PAD_IDX) & (ids != UNK_IDX)].unique()
                if ids.numel() > 0:
                    multi_hot[i, ids] = 1.0
            out["multi_hot"] = multi_hot

        return out

    return _collate


def make_diffusion_dataloader(
    bundle: dict[str, Any],
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    view: str = "both",
) -> DataLoader:
    dataset = DiffusionTensorDataset(bundle)
    vocab_size = int(bundle["meta"]["vocab_size"])
    collate_fn = make_diffusion_collate(vocab_size=vocab_size, view=view)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
