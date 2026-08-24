"""Train a small DRAFT model for speculative decoding (Part I, step 8).

Same corpus and next-token objective as base pretraining, but a much smaller GPT
so each forward is cheap. Its only job is to propose tokens the target usually
accepts. Reuses the pretraining machinery from ``training.common``; writes to
``SPEC_DRAFT_DIR``.
"""

import json
import time
from pathlib import Path

import torch

from src.config import (
    PACKED_DIR,
    META_FILE,
    CONTEXT_LEN,
    SEED,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    WEIGHT_DECAY,
    BETA1,
    BETA2,
    GRAD_CLIP,
    EVAL_INTERVAL,
    LOG_INTERVAL,
    SPEC_DRAFT_DIR,
    DRAFT_N_LAYER,
    DRAFT_N_HEAD,
    DRAFT_D_MODEL,
    DRAFT_MAX_STEPS,
    DRAFT_LR,
    DRAFT_MIN_LR,
    DRAFT_WARMUP_STEPS,
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


def train_draft() -> None:
    """Pretrain a small draft GPT on the packed corpus; save to SPEC_DRAFT_DIR."""
    torch.manual_seed(SEED)
    device, ctx, scaler = setup_amp()

    meta = json.loads((Path(PACKED_DIR) / META_FILE).read_text())
    block_size = CONTEXT_LEN
    cfg = GPTConfig(
        vocab_size=meta["vocab_size"], block_size=block_size,
        n_layer=DRAFT_N_LAYER, n_head=DRAFT_N_HEAD, d_model=DRAFT_D_MODEL,
    )
    model = GPT(cfg).to(device)
    optimizer = configure_optimizers(model, WEIGHT_DECAY, DRAFT_LR, (BETA1, BETA2), device)

    run_dir = new_run_dir(SPEC_DRAFT_DIR)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    print(
        f"draft: {model.num_params():,} non-embedding params "
        f"({DRAFT_N_LAYER}L/{DRAFT_N_HEAD}H/{DRAFT_D_MODEL}d) on {device}"
    )
    print(f"run dir: {run_dir} | {DRAFT_MAX_STEPS:,} steps")

    best_val = float("inf")
    model.train()
    x, y = get_batch("train", block_size, BATCH_SIZE, device)
    t0 = time.time()

    for step in range(DRAFT_MAX_STEPS + 1):
        lr = cosine_lr(step, DRAFT_WARMUP_STEPS, DRAFT_MAX_STEPS, DRAFT_LR, DRAFT_MIN_LR)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, ctx, block_size, device)
            print(f"step {step:>6}/{DRAFT_MAX_STEPS}: train {losses['train']:.4f} | "
                  f"val {losses['valid']:.4f} | lr {lr:.2e}")
            log_metrics(run_dir, {"step": step, "train_loss": round(losses["train"], 4),
                                  "val_loss": round(losses["valid"], 4), "lr": lr})
            if losses["valid"] < best_val:
                best_val = losses["valid"]
                save_checkpoint(best_path, model, optimizer, step, best_val, cfg)
            save_checkpoint(last_path, model, optimizer, step, best_val, cfg)

        if step == DRAFT_MAX_STEPS:
            break

        for _ in range(GRAD_ACCUM_STEPS):
            with ctx:
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM_STEPS
            x, y = get_batch("train", block_size, BATCH_SIZE, device)
            scaler.scale(loss).backward()

        optimizer_step(scaler, [(optimizer, model.parameters())], GRAD_CLIP)

        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {step:>6}/{DRAFT_MAX_STEPS} ({DRAFT_MAX_STEPS - step:,} left): "
                  f"loss {loss.item() * GRAD_ACCUM_STEPS:.4f} | lr {lr:.2e} | "
                  f"{dt / max(1, LOG_INTERVAL) * 1000:.0f} ms/step")

    png = plot_losses(run_dir, x="step")
    print(f"done. best val {best_val:.4f}. checkpoints in {run_dir}/" + (f" (curve: {png})" if png else ""))


if __name__ == "__main__":
    train_draft()
