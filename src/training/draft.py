"""Train a small DRAFT model for speculative decoding (Part I, step 8).

Same corpus and next-token objective as base pretraining, but a much smaller GPT
so each forward is cheap. Its only job is to propose tokens the target usually
accepts. Reuses the pretraining machinery from ``training.common``; writes to
``SPEC_DRAFT_DIR``.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import (
    PACKED_DIR,
    META_FILE,
    CONTEXT_LEN,
    SEED,
    SPEC_DRAFT_DIR,
    GPTConfig,
)
from src.models.gpt import GPT
from src.models.checkpoints import save_checkpoint, new_run_dir
from src.eval.metrics import log_metrics, plot_losses
from src.training.common import (
    setup_amp,
    cosine_lr,
    optimizer_step,
    get_batch,
    configure_optimizers,
    estimate_loss,
)


@dataclass
class DraftConfig:
    """
    Draft-model architecture + training schedule.

    A small GPT on the same corpus/objective as pretraining, which
    proposes tokens the target accepts. A better draft raises the acceptance rate (speedup).
    """

    n_layer: int = 2
    n_head: int = 2  # d_model 128 / 2 = 64 head dim
    d_model: int = 128
    max_steps: int = 3000  # ~0.75 epoch, enough for a usable draft
    batch_size: int = 64
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9   # Adam betas
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 8       # effective batch = batch_size * grad_accum_steps
    eval_interval: int = 500        # steps between val evals + checkpoints
    log_interval: int = 20          # steps between train-loss logs


def train_draft(cfg: DraftConfig = DraftConfig()) -> None:
    """Pretrain a small draft GPT on the packed corpus; save to SPEC_DRAFT_DIR."""
    torch.manual_seed(SEED)
    device, ctx, scaler = setup_amp()

    meta = json.loads((Path(PACKED_DIR) / META_FILE).read_text())
    block_size = CONTEXT_LEN
    gpt_cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
    )
    model = GPT(gpt_cfg).to(device)
    optimizer = configure_optimizers(
        model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
    )

    run_dir = new_run_dir(SPEC_DRAFT_DIR)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    print(
        f"draft: {model.num_params():,} non-embedding params "
        f"({cfg.n_layer}L/{cfg.n_head}H/{cfg.d_model}d) on {device}"
    )
    print(f"run dir: {run_dir} | {cfg.max_steps:,} steps")

    best_val = float("inf")
    model.train()
    x, y = get_batch("train", block_size, cfg.batch_size, device)
    t0 = time.time()

    for step in range(cfg.max_steps + 1):
        lr = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, ctx, block_size, device, cfg.batch_size)
            print(
                f"step {step:>6}/{cfg.max_steps}: train {losses['train']:.4f} | "
                f"val {losses['valid']:.4f} | lr {lr:.2e}"
            )
            log_metrics(
                run_dir,
                {
                    "step": step,
                    "train_loss": round(losses["train"], 4),
                    "val_loss": round(losses["valid"], 4),
                    "lr": lr,
                },
            )
            if losses["valid"] < best_val:
                best_val = losses["valid"]
                save_checkpoint(best_path, model, optimizer, step, best_val, gpt_cfg)
            save_checkpoint(last_path, model, optimizer, step, best_val, gpt_cfg)

        if step == cfg.max_steps:
            break

        for _ in range(cfg.grad_accum_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            x, y = get_batch("train", block_size, cfg.batch_size, device)
            scaler.scale(loss).backward()

        optimizer_step(scaler, [(optimizer, model.parameters())], cfg.grad_clip)

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"step {step:>6}/{cfg.max_steps} ({cfg.max_steps - step:,} left): "
                f"loss {loss.item() * cfg.grad_accum_steps:.4f} | lr {lr:.2e} | "
                f"{dt / max(1, cfg.log_interval) * 1000:.0f} ms/step"
            )

    png = plot_losses(run_dir, x="step")
    print(
        f"done. best val {best_val:.4f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train_draft()
