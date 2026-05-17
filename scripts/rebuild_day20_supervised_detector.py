from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/mimiciv_train_detector_supervised.pkl")
    parser.add_argument("--val_path", default="data/processed/mimiciv_val_detector_supervised.pkl")
    parser.add_argument("--out_dir", default="outputs/detector/day20_supervised/run_luxury")
    parser.add_argument("--eval_out_dir", default="outputs/detector_eval/day20_supervised/run_luxury")
    parser.add_argument("--vocab_path", default="outputs/detector/day20_supervised/run_luxury/vocab.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=160)
    parser.add_argument("--hidden_dim", type=int, default=320)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--truncate_strategy", choices=["head", "tail"], default="tail")
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path: str | Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(obj)}")
    return obj.copy()


def infer_sequence_column(df: pd.DataFrame) -> str:
    for col in SEQ_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        sample = df[col].dropna().head(20)
        if len(sample) and any(isinstance(x, (list, tuple)) for x in sample):
            return col
    raise ValueError(f"Could not infer sequence column. Columns={list(df.columns)}")


def normalize_tokens(value: Any) -> list[str]:
    bad = {"", "nan", "none", "null", "na", "n/a"}

    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            token = str(item).strip()
            if token.lower() not in bad:
                out.append(token)
        return out

    if isinstance(value, str):
        text = value.strip()
        if text.lower() in bad:
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

        return [part.strip() for part in text.split() if part.strip()]

    if value is None:
        return []

    token = str(value).strip()
    return [token] if token and token.lower() not in bad else []


def ensure_supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    seq_col = infer_sequence_column(df)
    df["sequence_tokens"] = df[seq_col].apply(normalize_tokens)
    df = df[df["sequence_tokens"].map(len) > 0].copy()

    if "label" not in df.columns:
        if "is_synthetic_anomaly" in df.columns:
            df["label"] = df["is_synthetic_anomaly"].astype(int)
        elif "source" in df.columns:
            df["label"] = df["source"].astype(str).str.contains("anomaly", case=False, na=False).astype(int)
        else:
            raise ValueError("Could not infer label column. Expected label/is_synthetic_anomaly/source.")

    df["label"] = df["label"].astype(int)

    if "source" not in df.columns:
        df["source"] = np.where(df["label"] == 1, "synthetic_anomaly", "normal")

    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = ""

    df["anomaly_type"] = df["anomaly_type"].fillna("").astype(str)
    df["sequence_length"] = df["sequence_tokens"].map(len)

    return df[["sequence_tokens", "label", "source", "anomaly_type", "sequence_length"]].reset_index(drop=True)


def build_vocab(sequences: list[list[str]], min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for seq in sequences:
        counter.update(seq)

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_sequence(tokens: list[str], vocab: dict[str, int], max_len: int, truncate_strategy: str) -> list[int]:
    if len(tokens) == 0:
        return [UNK_IDX]

    if len(tokens) > max_len:
        tokens = tokens[:max_len] if truncate_strategy == "head" else tokens[-max_len:]

    return [int(vocab.get(tok, UNK_IDX)) for tok in tokens]


class LabeledSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict[str, int], max_len: int, truncate_strategy: str) -> None:
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.truncate_strategy = truncate_strategy

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        ids = encode_sequence(row["sequence_tokens"], self.vocab, self.max_len, self.truncate_strategy)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": len(ids),
            "label": float(row["label"]),
            "source": str(row["source"]),
            "anomaly_type": str(row["anomaly_type"]),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = [item["input_ids"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    padded = pad_sequence(input_ids, batch_first=True, padding_value=PAD_IDX)
    return {
        "input_ids": padded,
        "lengths": lengths,
        "labels": labels,
        "sources": [item["source"] for item in batch],
        "anomaly_types": [item["anomaly_type"] for item in batch],
    }


class GRUSequenceBinaryClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

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
        x = h_n[-1]
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    if len(thresholds) == 0:
        threshold = 0.5
        pred = (probs >= threshold).astype(int)
        return (
            threshold,
            float(precision_score(y_true, pred, zero_division=0)),
            float(recall_score(y_true, pred, zero_division=0)),
            float(f1_score(y_true, pred, zero_division=0)),
        )

    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1],
        1e-12,
        None,
    )
    idx = int(np.argmax(f1_values))
    threshold = float(thresholds[idx])
    pred = (probs >= threshold).astype(int)

    return (
        threshold,
        float(precision_score(y_true, pred, zero_division=0)),
        float(recall_score(y_true, pred, zero_division=0)),
        float(f1_score(y_true, pred, zero_division=0)),
    )


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    y_true = y_true.astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "threshold": 0.5,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "predicted_positive_rate": float("nan"),
        }

    roc_auc = float(roc_auc_score(y_true, probs))
    avg_precision = float(average_precision_score(y_true, probs))

    if threshold is None:
        threshold, precision, recall, f1 = best_f1_threshold(y_true, probs)
    else:
        pred = (probs >= threshold).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))

    return {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float((probs >= threshold).mean()),
    }


def threshold_sweep(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    rows = []
    for thr in np.linspace(0, 1, 101):
        pred = (probs >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "predicted_positive_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        losses.append(float(loss.item()))
        all_probs.append(probs)
        all_labels.append(labels_np)

        for i in range(len(probs)):
            rows.append(
                {
                    "source": batch["sources"][i],
                    "anomaly_type": batch["anomaly_types"][i],
                    "label": int(labels_np[i]),
                    "prob_anomaly": float(probs[i]),
                }
            )

    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)

    metrics = binary_metrics(labels_np, probs_np)
    metrics["loss"] = float(np.mean(losses))

    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    eval_out_dir = Path(args.eval_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    train_df = ensure_supervised_frame(load_pickle(args.train_path))
    val_df = ensure_supervised_frame(load_pickle(args.val_path))

    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        vocab = {str(k): int(v) for k, v in load_json(vocab_path).items()}
        vocab_source = str(vocab_path)
    else:
        vocab = build_vocab(train_df["sequence_tokens"].tolist(), min_freq=args.min_freq)
        save_json(out_dir / "vocab.json", vocab)
        vocab_source = "rebuilt_from_train"

    train_ds = LabeledSequenceDataset(train_df, vocab, args.max_len, args.truncate_strategy)
    val_ds = LabeledSequenceDataset(val_df, vocab, args.max_len, args.truncate_strategy)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = GRUSequenceBinaryClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pos = float((train_df["label"] == 1).sum())
    neg = float((train_df["label"] == 0).sum())
    pos_weight_value = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ap = -1.0
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_metrics, _ = evaluate(model, val_loader, device)
        train_loss = float(np.mean(train_losses))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_average_precision": val_metrics["average_precision"],
            "val_threshold": val_metrics["threshold"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_predicted_positive_rate": val_metrics["predicted_positive_rate"],
        }

        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if val_metrics["average_precision"] > best_ap:
            best_ap = float(val_metrics["average_precision"])
            best_epoch = epoch
            best_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "vocab_size": len(vocab),
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                },
                out_dir / "best.pt",
            )

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_ap={val_metrics['average_precision']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "vocab_size": len(vocab),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        },
        out_dir / "last.pt",
    )

    ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    final_metrics, scores_df = evaluate(model, val_loader, device)
    sweep_df = threshold_sweep(scores_df["label"].to_numpy(), scores_df["prob_anomaly"].to_numpy())

    scores_df.to_csv(eval_out_dir / "scores.csv", index=False)
    sweep_df.to_csv(eval_out_dir / "threshold_sweep.csv", index=False)

    breakdown_df = (
        scores_df.groupby("anomaly_type", dropna=False)
        .agg(
            count=("label", "size"),
            anomaly_rate=("label", "mean"),
            mean_score=("prob_anomaly", "mean"),
            median_score=("prob_anomaly", "median"),
        )
        .reset_index()
    )
    breakdown_df.to_csv(eval_out_dir / "anomaly_type_breakdown.csv", index=False)

    summary = {
        "status": "recovered_day20_supervised_detector",
        "device_used": str(device),
        "train_path": args.train_path,
        "val_path": args.val_path,
        "vocab_source": vocab_source,
        "vocab_size": int(len(vocab)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "class_balance": {
            "train_normal": int((train_df["label"] == 0).sum()),
            "train_anomaly": int((train_df["label"] == 1).sum()),
            "val_normal": int((val_df["label"] == 0).sum()),
            "val_anomaly": int((val_df["label"] == 1).sum()),
            "pos_weight": float(pos_weight_value),
        },
        "best_epoch": int(best_epoch),
        "best_metrics_during_training": best_metrics,
        "final_eval_metrics_from_best_checkpoint": final_metrics,
        "outputs": {
            "best_checkpoint": str(out_dir / "best.pt"),
            "last_checkpoint": str(out_dir / "last.pt"),
            "scores_csv": str(eval_out_dir / "scores.csv"),
            "threshold_sweep_csv": str(eval_out_dir / "threshold_sweep.csv"),
            "anomaly_type_breakdown_csv": str(eval_out_dir / "anomaly_type_breakdown.csv"),
        },
    }

    save_json(out_dir / "summary.json", summary)
    save_json(eval_out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
