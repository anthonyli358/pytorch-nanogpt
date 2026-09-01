"""Supervised fine-tuning (SFT) on TinyStories-Instruct.

Fine-tunes a pretrained checkpoint to follow the instruction format. Epoch-based
over a DataLoader (finite instruct set), with the prompt masked so only the story
contributes to the loss.
"""

import time
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import DataLoader

from src.config import (
    DATA_DIR,
    CKPT_DIR,
    SFT_CKPT_DIR,
    SEED,
)
from src.models.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    new_run_dir,
    resolve_checkpoint,
)
from src.models.tokenizer import Tokenizer
from src.data.sft_data import download_instruct, SFTDataset, MAX_SFT_EXAMPLES
from src.data.common import pad_batch
from src.eval.metrics import log_metrics, plot_series
from src.training.common import setup_amp, cosine_lr, optimizer_step, configure_optimizers


@dataclass
class SFTConfig:
    """
    SFT trainer hyperparameters for fine-tuning, init from a pretrained run.
    """

    init_run: str | None = None   # pretrained run to fine-tune from (None = latest)
    max_len: int | None = None    # cap example length in tokens; None -> model block_size
    patience: int = 2             # early-stop after this many epochs without val improvement
    batch_size: int = 32
    epochs: int = 1
    lr: float = 3e-4              # lower than pretraining peak (fine-tuning)
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9   # Adam betas
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_interval: int = 50


@torch.no_grad()
def evaluate_sft(model, loader: DataLoader, ctx, device: str) -> float:
    """Token-weighted mean loss over the masked (response-only) targets."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        n = int((y != -1).sum().item())  # unmasked (response) target tokens
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def train_sft(cfg: SFTConfig = SFTConfig()) -> None:
    """Run epoch-based SFT with per-epoch eval, best/last checkpoints, early stop."""
    torch.manual_seed(SEED)
    device, ctx, scaler = setup_amp()

    # Init from the pretrained checkpoint.
    init_ckpt = resolve_checkpoint(cfg.init_run, "best.pt", CKPT_DIR)
    model, _ = load_checkpoint(init_ckpt, device)
    gpt_cfg = model.cfg
    optimizer = configure_optimizers(
        model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
    )
    print(
        f"SFT init from {init_ckpt} ({model.num_params():,} non-embedding params) on {device}"
    )

    # Data: examples are filtered to fit the context (never truncated).
    tok = Tokenizer()
    max_len = min(cfg.max_len or gpt_cfg.block_size, gpt_cfg.block_size)
    paths = download_instruct(DATA_DIR)
    train_ds = SFTDataset(paths["train"], tok, max_len, MAX_SFT_EXAMPLES)
    val_ds = SFTDataset(paths["valid"], tok, max_len)
    collate = partial(pad_batch, pad_id=tok.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
    )

    total_steps = len(train_loader) * cfg.epochs
    run_dir = new_run_dir(SFT_CKPT_DIR)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    print(
        f"run dir: {run_dir} | {len(train_ds):,} train / {len(val_ds):,} val examples | "
        f"{total_steps:,} steps"
    )

    best_val = float("inf")
    no_improve = 0
    global_step = 0
    model.train()
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ep_loss_sum, ep_steps = 0.0, 0
        for x, y in train_loader:
            lr = cosine_lr(global_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
            ep_loss_sum += loss.item()
            ep_steps += 1
            scaler.scale(loss).backward()
            optimizer_step(scaler, [(optimizer, model.parameters())], cfg.grad_clip)

            if global_step % cfg.log_interval == 0:
                print(
                    f"epoch {epoch} step {global_step:>6}: loss {loss.item():.4f} | lr {lr:.2e}"
                )
                log_metrics(run_dir, {
                    "step": global_step,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": "",  # filled only at epoch boundaries
                    "lr": lr,
                })
            global_step += 1

        val_loss = evaluate_sft(model, val_loader, ctx, device)
        train_loss = ep_loss_sum / max(1, ep_steps)
        print(
            f"epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | {(time.time() - start) / 60:.1f} min"
        )
        log_metrics(  # one row per epoch carries the val point (same schema as the per-step rows)
            run_dir,
            {
                "step": global_step,
                "train_loss": "",
                "val_loss": round(val_loss, 4),
                "lr": lr,
            },
        )

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(best_path, model, optimizer, global_step, best_val, gpt_cfg)
            print(f"  new best val {best_val:.4f} - saved {best_path}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"early stopping after {no_improve} epochs without improvement")
                break
        save_checkpoint(last_path, model, optimizer, global_step, best_val, gpt_cfg)

    png = plot_series(run_dir, "step", [("SFT loss (masked response)", ["train_loss", "val_loss"])])
    print(
        f"done. best val loss {best_val:.4f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train_sft()
