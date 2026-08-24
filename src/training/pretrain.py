"""Base pretraining: a GPT on the packed TinyStories corpus (Part I, step 5).

Samples random fixed-length windows straight off the uint16 memmaps (stateless,
resumable), with periodic val eval and checkpointing. The batching / optimizer /
AMP / LR machinery lives in ``training.common``; this file is just the loop.
"""

import json
import time
from pathlib import Path

import torch

from src.config import (
    PACKED_DIR,
    META_FILE,
    SEED,
    COMPILE,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    MAX_STEPS,
    WARMUP_STEPS,
    LR,
    MIN_LR,
    WEIGHT_DECAY,
    BETA1,
    BETA2,
    GRAD_CLIP,
    EVAL_INTERVAL,
    LOG_INTERVAL,
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


def train() -> None:
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
        cfg = model.cfg
        optimizer = configure_optimizers(model, WEIGHT_DECAY, LR, (BETA1, BETA2), device)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt["best_val_loss"]
        print(f"resumed {run_dir}/last.pt at step {start_step} (best_val {best_val:.4f})")
    else:
        run_dir = new_run_dir(CKPT_DIR)
        cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=block_size)
        model = GPT(cfg).to(device)
        optimizer = configure_optimizers(model, WEIGHT_DECAY, LR, (BETA1, BETA2), device)
        print(f"fresh model: {model.num_params():,} non-embedding params on {device}")
        print(f"run dir: {run_dir}")

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    if COMPILE:
        model = torch.compile(model)

    model.train()
    x, y = get_batch("train", block_size, BATCH_SIZE, device)  # prefetch first batch
    t0 = time.time()

    for step in range(start_step, MAX_STEPS + 1):
        lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, LR, MIN_LR)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, ctx, block_size, device)
            print(f"step {step:>6}: train {losses['train']:.4f} | val {losses['valid']:.4f} | lr {lr:.2e}")
            log_metrics(run_dir, {"step": step, "train_loss": round(losses["train"], 4),
                                  "val_loss": round(losses["valid"], 4), "lr": lr})
            if losses["valid"] < best_val:
                best_val = losses["valid"]
                save_checkpoint(best_path, model, optimizer, step, best_val, cfg)
            save_checkpoint(last_path, model, optimizer, step, best_val, cfg)

        if step == MAX_STEPS:
            break

        for _ in range(GRAD_ACCUM_STEPS):
            with ctx:
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM_STEPS
            x, y = get_batch("train", block_size, BATCH_SIZE, device)  # prefetch during backward
            scaler.scale(loss).backward()

        optimizer_step(scaler, [(optimizer, model.parameters())], GRAD_CLIP)

        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {step:>6}: loss {loss.item() * GRAD_ACCUM_STEPS:.4f} | "
                  f"lr {lr:.2e} | {dt / max(1, LOG_INTERVAL) * 1000:.0f} ms/step")

    png = plot_losses(run_dir, x="step")
    print(f"done. best val loss {best_val:.4f}. checkpoints in {run_dir}/" + (f" (curve: {png})" if png else ""))


if __name__ == "__main__":
    train()
