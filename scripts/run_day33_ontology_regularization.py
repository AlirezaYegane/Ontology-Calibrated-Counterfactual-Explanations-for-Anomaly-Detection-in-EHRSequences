from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_torch(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_input_ids(obj: Any, pad_idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    meta: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key in ("input_ids", "sequence_ids", "ids", "tokens"):
            if key in obj and torch.is_tensor(obj[key]):
                input_ids = obj[key].long()
                break
        else:
            raise KeyError(f"Could not find input id tensor in artifact. Keys: {list(obj.keys())}")

        if "attention_mask" in obj and torch.is_tensor(obj["attention_mask"]):
            mask = obj["attention_mask"].bool()
        elif "mask" in obj and torch.is_tensor(obj["mask"]):
            mask = obj["mask"].bool()
        else:
            mask = input_ids.ne(pad_idx)

        for k, v in obj.items():
            if k not in {"input_ids", "sequence_ids", "ids", "tokens", "attention_mask", "mask"}:
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    meta[k] = v

        return input_ids, mask, meta

    if torch.is_tensor(obj):
        input_ids = obj.long()
        return input_ids, input_ids.ne(pad_idx), meta

    raise TypeError(f"Unsupported artifact type: {type(obj)}")


def find_vocab_path(summary_path: Path | None, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    candidates: list[Path] = []

    if summary_path and summary_path.exists():
        try:
            s = load_json(summary_path)
            for key in ("vocab_path", "vocabulary_path", "token_vocab_path"):
                if s.get(key):
                    candidates.append(Path(str(s[key])))
        except Exception:
            pass

    candidates.extend(
        [
            Path("artifacts/day27/vocab.json"),
            Path("artifacts/day27/mimiciv_val_vocab.json"),
            Path("outputs/diffusion/day30_smoke/vocab.json"),
            Path("outputs/diffusion/day31_full/vocab.json"),
            Path("outputs/diffusion/day32_final/vocab.json"),
        ]
    )

    for p in candidates:
        if p.exists():
            return p

    return None


def load_vocab(vocab_path: Path | None) -> dict[int, str]:
    if vocab_path is None:
        return {}

    raw = load_json(vocab_path)

    id_to_token: dict[int, str] = {}

    if isinstance(raw, dict):
        # token -> id
        if raw and all(isinstance(v, int) for v in raw.values()):
            for token, idx in raw.items():
                id_to_token[int(idx)] = str(token)
        # id -> token
        elif raw and all(str(k).isdigit() for k in raw.keys()):
            for idx, token in raw.items():
                id_to_token[int(idx)] = str(token)

    elif isinstance(raw, list):
        for idx, token in enumerate(raw):
            id_to_token[idx] = str(token)

    return id_to_token


def category_from_token(token: str) -> str:
    up = token.upper()

    if token in {"<pad>", "[PAD]"}:
        return "pad"

    if token in {"<unk>", "[UNK]"}:
        return "unknown"

    if up.startswith(("ICD", "DX", "DIAG", "SNOMED")):
        return "diagnosis"

    if up.startswith(("PROC", "CPT")):
        return "procedure"

    if up.startswith(("RXNORM", "NDC", "MED", "DRUG", "RAW_DRUG")):
        return "medication"

    if up.startswith(("UNMAPPED", "UNKNOWN", "UNK", "RAW_", "RAW-", "NO_MAP")):
        return "unknown"

    return "other"


def build_category_masks(id_to_token: dict[int, str], vocab_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    masks = {
        "diagnosis": torch.zeros(vocab_size, device=device),
        "procedure": torch.zeros(vocab_size, device=device),
        "medication": torch.zeros(vocab_size, device=device),
        "unknown": torch.zeros(vocab_size, device=device),
        "other": torch.zeros(vocab_size, device=device),
    }

    for idx in range(vocab_size):
        token = id_to_token.get(idx, str(idx))
        cat = category_from_token(token)
        if cat in masks:
            masks[cat][idx] = 1.0

    return masks


class SequenceTensorDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids.long()
        self.attention_mask = attention_mask.bool()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class OntologyRegularizedDiffusion(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def clean_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x_noisy.shape
        pos = torch.arange(seq_len, device=x_noisy.device).unsqueeze(0).expand(bsz, seq_len)
        h = x_noisy + self.pos_embedding(pos) + self.time_embedding(t).unsqueeze(1)
        key_padding_mask = ~attention_mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.out(self.norm(h))


@dataclass
class DiffusionSchedule:
    steps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def build(cls, steps: int, schedule: str, device: torch.device) -> "DiffusionSchedule":
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, steps, device=device)
        elif schedule == "cosine":
            s = 0.008
            x = torch.linspace(0, steps, steps + 1, device=device)
            alpha_bars = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = betas.clamp(1e-5, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(steps=steps, betas=betas, alphas=alphas, alpha_bars=alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def masked_mse(pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    valid = attention_mask.unsqueeze(-1).float()
    denom = valid.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target) ** 2 * valid).sum() / denom


def ontology_loss_from_x0(
    model: OntologyRegularizedDiffusion,
    x0_pred: torch.Tensor,
    attention_mask: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    scale = math.sqrt(model.d_model)
    logits = torch.matmul(x0_pred, model.token_embedding.weight.t()) / scale
    probs = torch.softmax(logits, dim=-1)

    valid = attention_mask.float()

    diag_mass = torch.matmul(probs, masks["diagnosis"])
    med_mass = torch.matmul(probs, masks["medication"])
    proc_mass = torch.matmul(probs, masks["procedure"])
    unk_mass = torch.matmul(probs, masks["unknown"] + masks["other"]).clamp_max(1.0)

    diag_any = (diag_mass * valid).amax(dim=1)
    med_any = (med_mass * valid).amax(dim=1)
    proc_any = (proc_mass * valid).amax(dim=1)

    med_without_diag = torch.relu(med_any - diag_any)
    proc_without_diag = torch.relu(proc_any - diag_any) * 0.5
    unknown_penalty = (unk_mass * valid).sum() / valid.sum().clamp_min(1.0)

    loss = med_without_diag.mean() + proc_without_diag.mean() + unknown_penalty

    metrics = {
        "ont_med_without_diag": float(med_without_diag.detach().mean().cpu()),
        "ont_proc_without_diag": float(proc_without_diag.detach().mean().cpu()),
        "ont_unknown_mass": float(unknown_penalty.detach().cpu()),
        "ont_total": float(loss.detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def decode_nearest(model: OntologyRegularizedDiffusion, x: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(x, model.token_embedding.weight.t()) / math.sqrt(model.d_model)
    return logits.argmax(dim=-1)


@torch.no_grad()
def violation_summary_from_ids(ids: torch.Tensor, masks_cpu: dict[str, torch.Tensor], pad_idx: int) -> dict[str, float]:
    diag_set = set(torch.where(masks_cpu["diagnosis"] > 0)[0].tolist())
    med_set = set(torch.where(masks_cpu["medication"] > 0)[0].tolist())
    proc_set = set(torch.where(masks_cpu["procedure"] > 0)[0].tolist())
    unknown_set = set(torch.where((masks_cpu["unknown"] + masks_cpu["other"]) > 0)[0].tolist())

    n = int(ids.shape[0])
    if n == 0:
        return {
            "records": 0,
            "violation_rate": 0.0,
            "med_without_diag_rate": 0.0,
            "proc_without_diag_rate": 0.0,
            "unknown_or_other_rate": 0.0,
        }

    med_without_diag = 0
    proc_without_diag = 0
    unknown_or_other = 0
    any_violation = 0

    for row in ids.cpu().tolist():
        toks = [int(x) for x in row if int(x) != pad_idx]
        has_diag = any(x in diag_set for x in toks)
        has_med = any(x in med_set for x in toks)
        has_proc = any(x in proc_set for x in toks)
        has_unknown = any(x in unknown_set for x in toks)

        v1 = has_med and not has_diag
        v2 = has_proc and not has_diag
        v3 = has_unknown

        med_without_diag += int(v1)
        proc_without_diag += int(v2)
        unknown_or_other += int(v3)
        any_violation += int(v1 or v2 or v3)

    return {
        "records": n,
        "violation_rate": any_violation / n,
        "med_without_diag_rate": med_without_diag / n,
        "proc_without_diag_rate": proc_without_diag / n,
        "unknown_or_other_rate": unknown_or_other / n,
    }


@torch.no_grad()
def evaluate(
    model: OntologyRegularizedDiffusion,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    masks: dict[str, torch.Tensor],
    lambda_total: float,
    device: torch.device,
    pad_idx: int,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    losses = []
    diff_losses = []
    ont_losses = []
    decoded_rows = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        x0 = model.clean_embeddings(input_ids)
        noise = torch.randn_like(x0)
        t = torch.randint(0, schedule.steps, (input_ids.shape[0],), device=device)
        xt = schedule.q_sample(x0, t, noise)

        pred = model(xt, t, attention_mask)
        diff_loss = masked_mse(pred, noise, attention_mask)

        ab = schedule.alpha_bars[t].view(-1, 1, 1)
        x0_pred = (xt - torch.sqrt(1.0 - ab) * pred) / torch.sqrt(ab).clamp_min(1e-6)
        ont_loss, _ = ontology_loss_from_x0(model, x0_pred, attention_mask, masks)

        loss = diff_loss + lambda_total * ont_loss

        losses.append(float(loss.detach().cpu()))
        diff_losses.append(float(diff_loss.detach().cpu()))
        ont_losses.append(float(ont_loss.detach().cpu()))

        decoded_rows.append(decode_nearest(model, x0_pred).cpu())

    decoded = torch.cat(decoded_rows, dim=0) if decoded_rows else torch.empty(0, 1, dtype=torch.long)
    masks_cpu = {k: v.detach().cpu() for k, v in masks.items()}
    violations = violation_summary_from_ids(decoded, masks_cpu, pad_idx=pad_idx)

    return {
        "val_loss": sum(losses) / max(len(losses), 1),
        "val_diffusion_loss": sum(diff_losses) / max(len(diff_losses), 1),
        "val_ontology_loss": sum(ont_losses) / max(len(ont_losses), 1),
        **{f"val_{k}": v for k, v in violations.items()},
    }


@torch.no_grad()
def sample_sequences(
    model: OntologyRegularizedDiffusion,
    schedule: DiffusionSchedule,
    n: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(n, seq_len, model.d_model, device=device)
    attention_mask = torch.ones(n, seq_len, dtype=torch.bool, device=device)

    for step in reversed(range(schedule.steps)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        pred_noise = model(x, t, attention_mask)

        beta = schedule.betas[step]
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]

        mean = (x - beta / torch.sqrt(1.0 - alpha_bar).clamp_min(1e-6) * pred_noise) / torch.sqrt(alpha).clamp_min(1e-6)

        if step > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean

    return decode_nearest(model, x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--summary_path", default="")
    p.add_argument("--vocab_path", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--artifact_dir", default="artifacts/day33")
    p.add_argument("--max_records", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps_per_epoch", type=int, default=200)
    p.add_argument("--diffusion_steps", type=int, default=64)
    p.add_argument("--beta_schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--lambda_total", type=float, default=0.10)
    p.add_argument("--pad_idx", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_sample_records", type=int, default=64)
    p.add_argument("--baseline_report", default="artifacts/day32/day32_final_diffusion_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    artifact_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_path) if args.summary_path else None
    vocab_path = find_vocab_path(summary_path, args.vocab_path or None)

    obj = load_torch(args.data_path)
    input_ids, attention_mask, meta = infer_input_ids(obj, pad_idx=args.pad_idx)

    if args.max_records and input_ids.shape[0] > args.max_records:
        input_ids = input_ids[: args.max_records]
        attention_mask = attention_mask[: args.max_records]

    vocab_size = int(input_ids.max().item()) + 1
    id_to_token = load_vocab(vocab_path)
    if id_to_token:
        vocab_size = max(vocab_size, max(id_to_token.keys()) + 1)

    n_records, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    indices = torch.randperm(n_records)
    val_n = max(1, int(0.1 * n_records))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    train_ds = SequenceTensorDataset(input_ids[train_idx], attention_mask[train_idx])
    val_ds = SequenceTensorDataset(input_ids[val_idx], attention_mask[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = OntologyRegularizedDiffusion(
        vocab_size=vocab_size,
        max_len=seq_len,
        pad_idx=args.pad_idx,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    schedule = DiffusionSchedule.build(args.diffusion_steps, args.beta_schedule, device)
    masks = build_category_masks(id_to_token, vocab_size, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    metrics_path = out_dir / "metrics.jsonl"

    config = vars(args).copy()
    config["vocab_path_resolved"] = str(vocab_path) if vocab_path else None
    config["vocab_size"] = vocab_size
    config["records_used"] = n_records
    config["seq_len"] = seq_len
    config["device_used"] = str(device)
    config["ontology_masks"] = {
        k: int(v.sum().detach().cpu().item())
        for k, v in masks.items()
    }
    save_json(out_dir / "config.json", config)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_diff = []
        epoch_ont = []

        for step, batch in enumerate(train_loader, start=1):
            if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
                break

            input_ids_b = batch["input_ids"].to(device)
            attention_mask_b = batch["attention_mask"].to(device)

            x0 = model.clean_embeddings(input_ids_b)
            noise = torch.randn_like(x0)
            t = torch.randint(0, schedule.steps, (input_ids_b.shape[0],), device=device)
            xt = schedule.q_sample(x0, t, noise)

            pred = model(xt, t, attention_mask_b)
            diff_loss = masked_mse(pred, noise, attention_mask_b)

            ab = schedule.alpha_bars[t].view(-1, 1, 1)
            x0_pred = (xt - torch.sqrt(1.0 - ab) * pred) / torch.sqrt(ab).clamp_min(1e-6)

            ont_loss, _ = ontology_loss_from_x0(model, x0_pred, attention_mask_b, masks)
            loss = diff_loss + args.lambda_total * ont_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_diff.append(float(diff_loss.detach().cpu()))
            epoch_ont.append(float(ont_loss.detach().cpu()))

        val = evaluate(
            model=model,
            loader=val_loader,
            schedule=schedule,
            masks=masks,
            lambda_total=args.lambda_total,
            device=device,
            pad_idx=args.pad_idx,
            max_batches=20,
        )

        generated_ids = sample_sequences(
            model=model,
            schedule=schedule,
            n=args.eval_sample_records,
            seq_len=seq_len,
            device=device,
        )
        gen_viol = violation_summary_from_ids(
            generated_ids,
            {k: v.detach().cpu() for k, v in masks.items()},
            pad_idx=args.pad_idx,
        )

        row = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "train_diffusion_loss": sum(epoch_diff) / max(len(epoch_diff), 1),
            "train_ontology_loss": sum(epoch_ont) / max(len(epoch_ont), 1),
            **val,
            **{f"generated_{k}": v for k, v in gen_viol.items()},
        }

        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[epoch {epoch:02d}] "
            f"train={row['train_loss']:.4f} "
            f"diff={row['train_diffusion_loss']:.4f} "
            f"ont={row['train_ontology_loss']:.4f} "
            f"val={row['val_loss']:.4f} "
            f"gen_violation={row['generated_violation_rate']:.4f}"
        )

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_row": row,
                },
                out_dir / "best.pt",
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
        },
        out_dir / "last.pt",
    )

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    best_row = min(rows, key=lambda r: r["val_loss"]) if rows else {}

    baseline_payload: dict[str, Any] | None = None
    baseline_path = Path(args.baseline_report)
    if baseline_path.exists():
        try:
            baseline_payload = load_json(baseline_path)
        except Exception:
            baseline_payload = None

    final_report = {
        "day": 33,
        "title": "Ontology-Regularized Diffusion Fine-Tuning",
        "status": "complete",
        "goal": "Add lightweight ontology-consistency regularization to the diffusion baseline and monitor generated violation proxies.",
        "data_path": args.data_path,
        "summary_path": args.summary_path,
        "out_dir": str(out_dir),
        "vocab_path": str(vocab_path) if vocab_path else None,
        "records_used": n_records,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "ontology_regularization": {
            "enabled": True,
            "lambda_total": args.lambda_total,
            "implemented_proxy_rules": [
                "medication_without_diagnosis",
                "procedure_without_diagnosis",
                "unknown_or_other_token_mass",
            ],
            "deferred": [
                "full SNOMED graph shortest-path regularization",
                "demographic-specific constraints requiring reliable metadata",
                "full counterfactual repair sampling loop",
            ],
        },
        "best_epoch": best_epoch,
        "best_metrics": best_row,
        "baseline_day32_reference": baseline_payload,
        "interpretation": (
            "Day 33 introduces a conservative ontology-aware training signal. "
            "The reported generated_violation_rate is a proxy metric based on decoded generated sequences, "
            "not a final clinical validation metric."
        ),
    }

    save_json(artifact_dir / "day33_ontology_regularization_report.json", final_report)
    save_json(out_dir / "summary.json", final_report)

    readme = f"""Day 33 — Ontology-Regularized Diffusion

Status: complete

What was added
- Lightweight ontology-consistency loss L_ont
- Proxy rules for medication/procedure context and unknown-token mass
- Generated/reconstructed violation monitoring
- Best checkpoint selection by validation loss

Main outputs
- {out_dir.as_posix()}/best.pt
- {out_dir.as_posix()}/last.pt
- {out_dir.as_posix()}/metrics.jsonl
- {out_dir.as_posix()}/summary.json
- {artifact_dir.as_posix()}/day33_ontology_regularization_report.json

Important limitation
This is a conservative ontology-regularization pass. Full SNOMED graph-distance loss,
demographic metadata constraints, and counterfactual repair search remain deferred.
"""
    (artifact_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
