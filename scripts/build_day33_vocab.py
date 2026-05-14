from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

BAD = {"", "nan", "none", "null", "na", "n/a"}


def normalize_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            token = str(x).strip()
            if token.lower() not in BAD:
                out.append(token)
        return out

    if isinstance(value, str):
        text = value.strip()
        if text.lower() in BAD:
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
            return [x.strip() for x in text.split(",") if x.strip()]

        return [x.strip() for x in text.split() if x.strip()]

    return []


def infer_sequence_column(df: pd.DataFrame) -> str:
    candidates = [
        "codes",
        "sequence_tokens",
        "tokens",
        "sequence",
        "event_codes",
        "concepts",
        "diagnosis_tokens",
        "procedure_tokens",
        "medication_tokens",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        sample = df[col].dropna().head(50)
        if len(sample) and any(isinstance(x, (list, tuple)) for x in sample):
            return col

    raise ValueError(f"Could not infer sequence column. Columns: {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--out_vocab", required=True)
    parser.add_argument("--sequence_column", default="")
    parser.add_argument("--max_records", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_pickle(args.input_path)
    col = args.sequence_column or infer_sequence_column(df)

    if args.max_records and len(df) > args.max_records:
        df = df.head(args.max_records).copy()

    counter: Counter[str] = Counter()

    for value in df[col]:
        counter.update(normalize_tokens(value))

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for token, _count in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)

    out_path = Path(args.out_vocab)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "input_path": args.input_path,
        "sequence_column": col,
        "rows_used": int(len(df)),
        "vocab_size": int(len(vocab)),
        "top_tokens": [{"token": t, "count": int(c)} for t, c in counter.most_common(30)],
        "out_vocab": str(out_path),
    }

    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
