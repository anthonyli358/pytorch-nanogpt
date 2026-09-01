"""
Base pretraining: a GPT trained on the packed TinyStories corpus.

Samples random fixed-length windows straight off the uint16 memmaps (stateless,
resumable), with periodic val eval and checkpointing.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import (
    PACKED_DIR,
    META_FILE,
    SEED,
    CKPT_DIR,
    RESUME,
    RESUME_FROM,
    CONTEXT_LEN,
    GPTConfig,
)
from src.models.gpt import GPT
from src.models.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    new_run_dir,
    latest_run_dir,
)
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
class PretrainConfig:
    """
    Base-pretraining schedule + optimizer knobs.

    Batch size, betas, and eval-iters are shared with common.py/draft and stay in
    src/config.py; the resume toggles live there too.
    """

    max_steps: int = 10_000  # ~2.5 epochs
    batch_size: int = 64
    warmup_steps: int = 200
    lr: float = 6e-4  # peak LR
    min_lr: float = 6e-5  # cosine floor (~ lr / 10)
    weight_decay: float = 0.1
    beta1: float = 0.9  # Adam betas
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 8  # effective batch = batch_size * grad_accum_steps
    eval_interval: int = 500  # steps between val evals + checkpoints
    log_interval: int = 20  # steps between train-loss logs
    compile: bool = False


def train(cfg: PretrainConfig = PretrainConfig()) -> None:
    """Run the pretraining loop, evaluating and checkpointing periodically."""
    torch.manual_seed(SEED)
    device, ctx, scaler = setup_amp()

    meta = json.loads((Path(PACKED_DIR) / META_FILE).read_text())
    block_size = CONTEXT_LEN

    best_val = float("inf")
    start_step = 0

    resume_dir = None
    if RESUME:
        p = Path(RESUME_FROM) if RESUME_FROM else latest_run_dir(CKPT_DIR)
        if p is not None:
            resume_dir = p.parent if p.suffix == ".pt" else p

    if resume_dir is not None and (resume_dir / "last.pt").exists():
        run_dir = resume_dir
        model, ckpt = load_checkpoint(run_dir / "last.pt", device)
        gpt_cfg = model.cfg
        optimizer = configure_optimizers(
            model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt["best_val_loss"]
        print(
            f"resumed {run_dir}/last.pt at step {start_step} (best_val {best_val:.4f})"
        )
    else:
        run_dir = new_run_dir(CKPT_DIR)
        gpt_cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=block_size)
        model = GPT(gpt_cfg).to(device)
        optimizer = configure_optimizers(
            model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
        )
        print(f"fresh model: {model.num_params():,} non-embedding params on {device}")
        print(f"run dir: {run_dir}")

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    if cfg.compile:
        model = torch.compile(model)

    model.train()
    x, y = get_batch(
        "train", block_size, cfg.batch_size, device
    )  # prefetch first batch
    t0 = time.time()

    for step in range(start_step, cfg.max_steps + 1):
        lr = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, ctx, block_size, device, cfg.batch_size)
            print(
                f"step {step:>6}: train {losses['train']:.4f} | val {losses['valid']:.4f} | lr {lr:.2e}"
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
            x, y = get_batch(
                "train", block_size, cfg.batch_size, device
            )  # prefetch during backward
            scaler.scale(loss).backward()

        optimizer_step(scaler, [(optimizer, model.parameters())], cfg.grad_clip)

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"step {step:>6}: loss {loss.item() * cfg.grad_accum_steps:.4f} | "
                f"lr {lr:.2e} | {dt / max(1, cfg.log_interval) * 1000:.0f} ms/step"
            )

    png = plot_losses(run_dir, x="step")
    print(
        f"done. best val loss {best_val:.4f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train()
