from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.training.diffusion_training_utils import (
    DiffusionSequenceDataset,
    build_diffusion_schedule,
    find_embedding_layer,
    forward_diffusion_model,
    instantiate_diffusion_model,
    load_diffusion_artifact,
    load_summary,
    masked_mse_loss,
    q_sample,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 30 diffusion training loop")

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_records", type=int, default=2048)
    parser.add_argument("--max_steps_per_epoch", type=int, default=50)

    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--beta_schedule", choices=["cosine", "linear"], default="cosine")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = load_summary(args.summary_path)

    input_ids, attention_mask, artifact_metadata = load_diffusion_artifact(
        args.data_path,
        pad_idx=args.pad_idx,
        max_records=args.max_records,
    )

    vocab_size = int(
        raw_summary.get("vocab_size")
        or artifact_metadata.get("vocab_size")
        or artifact_metadata["observed_vocab_size"]
    )
    vocab_size = max(vocab_size, int(input_ids.max().item()) + 1)

    max_len = int(input_ids.shape[1])

    dataset = DiffusionSequenceDataset(input_ids, attention_mask)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    model = instantiate_diffusion_model(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        pad_idx=args.pad_idx,
    ).to(device)

    embedding_layer = find_embedding_layer(model)

    schedule = build_diffusion_schedule(args.diffusion_steps, args.beta_schedule)
    schedule = {k: v.to(device) for k, v in schedule.items()}

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config_payload = {
        "args": vars(args),
        "device_used": str(device),
        "use_amp": use_amp,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "artifact_metadata": artifact_metadata,
        "raw_summary_subset": {
            k: raw_summary.get(k)
            for k in [
                "input_path",
                "rows_after_filter",
                "vocab_size",
                "max_len",
                "truncate_strategy",
                "loader_view",
                "tensor_shapes",
            ]
        },
        "model_class": model.__class__.__name__,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    save_json(out_dir / "config_resolved.json", config_payload)

    metrics_path = out_dir / "metrics.jsonl"
    best_loss = float("inf")
    best_epoch = -1

    global_step = 0
    epoch_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        token_counts: list[int] = []

        for step, batch in enumerate(loader, start=1):
            if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
                break

            batch_input_ids = batch["input_ids"].to(device, non_blocking=True)
            batch_attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            t = torch.randint(
                low=0,
                high=args.diffusion_steps,
                size=(batch_input_ids.shape[0],),
                device=device,
                dtype=torch.long,
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                x_start = embedding_layer(batch_input_ids)
                noise = torch.randn_like(x_start)
                x_noisy = q_sample(x_start, t, noise, schedule)

                pred_noise = forward_diffusion_model(
                    model=model,
                    x_noisy=x_noisy,
                    t=t,
                    attention_mask=batch_attention_mask,
                )

                loss = masked_mse_loss(
                    prediction=pred_noise,
                    target=noise,
                    attention_mask=batch_attention_mask,
                )

            scaler.scale(loss).backward()

            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            token_counts.append(int(batch_attention_mask.sum().item()))

            row = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "train_loss": loss_value,
                "batch_size": int(batch_input_ids.shape[0]),
                "nonpad_tokens": int(batch_attention_mask.sum().item()),
                "mean_timestep": float(t.float().mean().detach().cpu().item()),
            }

            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            if step == 1 or step % 10 == 0:
                print(
                    f"[epoch {epoch:02d} step {step:03d}] "
                    f"loss={loss_value:.6f} "
                    f"nonpad={row['nonpad_tokens']} "
                    f"mean_t={row['mean_timestep']:.2f}"
                )

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_losses.append(epoch_loss)

        epoch_summary = {
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "steps": len(losses),
            "mean_nonpad_tokens_per_batch": float(np.mean(token_counts)) if token_counts else 0.0,
        }

        save_json(out_dir / f"epoch_{epoch:03d}_summary.json", epoch_summary)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "config": config_payload,
                },
                out_dir / "best.pt",
            )

        print(f"[epoch {epoch:02d} done] epoch_loss={epoch_loss:.6f} best_loss={best_loss:.6f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": args.epochs,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "config": config_payload,
        },
        out_dir / "last.pt",
    )

    first_epoch_loss = epoch_losses[0] if epoch_losses else float("nan")
    last_epoch_loss = epoch_losses[-1] if epoch_losses else float("nan")

    final_summary = {
        "status": "complete",
        "day": 30,
        "title": "Diffusion training loop",
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "global_steps": global_step,
        "first_epoch_loss": first_epoch_loss,
        "last_epoch_loss": last_epoch_loss,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "loss_decreased": bool(last_epoch_loss <= first_epoch_loss),
        "artifacts": {
            "metrics_jsonl": str(metrics_path),
            "best_checkpoint": str(out_dir / "best.pt"),
            "last_checkpoint": str(out_dir / "last.pt"),
            "config_resolved": str(out_dir / "config_resolved.json"),
        },
    }

    save_json(out_dir / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
