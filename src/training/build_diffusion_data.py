from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.training.diffusion_data_utils import (
    build_padded_tensors,
    build_vocab,
    infer_sequence_column,
    load_pickle,
    load_vocab,
    make_diffusion_dataloader,
    normalize_tokens,
    save_json,
    save_vocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_pt", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--truncate_strategy", choices=["head", "tail"], default="tail")
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loader_view", choices=["token_ids", "multi_hot", "both"], default="both")
    parser.add_argument("--vocab_path", default="")
    parser.add_argument("--out_vocab_path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_pickle(args.input_path)
    seq_col = infer_sequence_column(df)

    work_df = df.copy()
    work_df["sequence_tokens"] = work_df[seq_col].apply(normalize_tokens)
    before_rows = len(work_df)
    work_df = work_df[work_df["sequence_tokens"].map(len) > 0].copy()
    dropped_empty = before_rows - len(work_df)

    sequences = work_df["sequence_tokens"].tolist()

    if args.vocab_path:
        vocab = load_vocab(args.vocab_path)
        vocab_source = args.vocab_path
    else:
        vocab = build_vocab(sequences, min_freq=args.min_freq)
        vocab_source = "built_from_input"

    if args.out_vocab_path:
        save_vocab(args.out_vocab_path, vocab)

    input_ids, attention_mask, lengths = build_padded_tensors(
        sequences=sequences,
        vocab=vocab,
        max_len=args.max_len,
        truncate_strategy=args.truncate_strategy,
    )

    bundle = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "meta": {
            "input_path": args.input_path,
            "sequence_column": seq_col,
            "rows_before_filter": int(before_rows),
            "rows_after_filter": int(len(work_df)),
            "dropped_empty_rows": int(dropped_empty),
            "max_len": int(args.max_len),
            "truncate_strategy": args.truncate_strategy,
            "vocab_size": int(len(vocab)),
            "vocab_source": vocab_source,
        },
    }

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_pt)

    loader = make_diffusion_dataloader(
        bundle=bundle,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        view=args.loader_view,
    )
    first_batch = next(iter(loader))

    batch_shapes: dict[str, list[int]] = {}
    for key, value in first_batch.items():
        if torch.is_tensor(value):
            batch_shapes[key] = list(value.shape)

    seq_lengths = work_df["sequence_tokens"].map(len)

    summary = {
        "input_path": args.input_path,
        "out_pt": str(out_pt),
        "sequence_column": seq_col,
        "rows_before_filter": int(before_rows),
        "rows_after_filter": int(len(work_df)),
        "dropped_empty_rows": int(dropped_empty),
        "vocab_size": int(len(vocab)),
        "max_len": int(args.max_len),
        "truncate_strategy": args.truncate_strategy,
        "loader_view": args.loader_view,
        "tensor_shapes": {
            "input_ids": list(bundle["input_ids"].shape),
            "attention_mask": list(bundle["attention_mask"].shape),
            "lengths": list(bundle["lengths"].shape),
        },
        "batch_shapes": batch_shapes,
        "sequence_length_stats": {
            "mean": float(seq_lengths.mean()),
            "median": float(seq_lengths.median()),
            "p95": float(seq_lengths.quantile(0.95)),
            "max": int(seq_lengths.max()),
        },
    }

    save_json(args.summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
