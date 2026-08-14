import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.config import (
    PACKED_DIR,
    PACKED_FILES,
    META_FILE,
    SEED,
    DEVICE,
    DTYPE,
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
    EVAL_ITERS,
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

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_device_dtype() -> tuple[str, torch.dtype]:
    """Pick the device and a supported autocast dtype, downgrading if needed."""
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _DTYPES[DTYPE]
    if dtype is torch.bfloat16 and not (
        device.startswith("cuda") and torch.cuda.is_bf16_supported()
    ):
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if dtype is torch.float16 and not device.startswith("cuda"):
        dtype = torch.float32
    return device, dtype


def get_batch(
    split: str, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of random windows from a packed split.

    The memmap is re-opened each call: this avoids a memory leak where the
    mapping accumulates references across a long training run.

    Args:
        split: "train" or "valid".
        block_size: Window (context) length.
        batch_size: Number of windows.
        device: Target device.

    Returns:
        (x, y) of shape (batch_size, block_size), where y is x
        shifted by one (next-token targets).
    """
    path = Path(PACKED_DIR) / PACKED_FILES[split]
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    return x.to(device), y.to(device)


def configure_optimizers(
    model: GPT, weight_decay: float, lr: float, betas: tuple[float, float], device: str
) -> torch.optim.AdamW:
    """Build AdamW with weight decay on 2D+ params only (not biases / norms)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device.startswith("cuda")  # fused AdamW is a CUDA-only fast path
    return torch.optim.AdamW(
        groups, lr=lr, betas=betas, **({"fused": True} if fused else {})
    )


def get_lr(step: int) -> float:
    """
    Cosine schedule with linear warmup; a pure function of the step.

    Ensures resume from checkpoints happens correctly.
    """

    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


@torch.no_grad()
def estimate_loss(model: GPT, ctx, block_size: int, device: str) -> dict[str, float]:
    """Average loss over EVAL_ITERS batches for each split."""
    out = {}
    model.eval()
    for split in ("train", "valid"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, block_size, BATCH_SIZE, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train() -> None:
    """
    Run the training loop, evaluating and checkpointing periodically.

    Samples random fixed-length windows straight off the uint16 memmaps.

    """
    torch.manual_seed(SEED)
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(pt_dtype is torch.float16))

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
        optimizer = configure_optimizers(
            model, WEIGHT_DECAY, LR, (BETA1, BETA2), device
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt["best_val_loss"]
        print(
            f"resumed {run_dir}/last.pt at step {start_step} (best_val {best_val:.4f})"
        )
    else:
        run_dir = new_run_dir(CKPT_DIR)
        cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=block_size)
        model = GPT(cfg).to(device)
        optimizer = configure_optimizers(
            model, WEIGHT_DECAY, LR, (BETA1, BETA2), device
        )
        print(
            f"fresh model: {model.num_params():,} non-embedding params on {device} ({pt_dtype})"
        )
        print(f"run dir: {run_dir}")

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    if COMPILE:
        model = torch.compile(model)

    model.train()
    x, y = get_batch("train", block_size, BATCH_SIZE, device)  # prefetch first batch
    t0 = time.time()

    for step in range(start_step, MAX_STEPS + 1):
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, ctx, block_size, device)
            print(
                f"step {step:>6}: train {losses['train']:.4f} | val {losses['valid']:.4f} | lr {lr:.2e}"
            )
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
            x, y = get_batch(
                "train", block_size, BATCH_SIZE, device
            )  # prefetch during backward
            scaler.scale(loss).backward()

        if GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"step {step:>6}: loss {loss.item() * GRAD_ACCUM_STEPS:.4f} | "
                f"lr {lr:.2e} | {dt / max(1, LOG_INTERVAL) * 1000:.0f} ms/step "
                f"| {step/MAX_STEPS:.1f}% of {MAX_STEPS}"
            )

    print(f"done. best val loss {best_val:.4f}. checkpoints in {CKPT_DIR}/")


if __name__ == "__main__":
    train()
