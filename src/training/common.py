"""
Shared training machinery for every stage (pretrain, draft, SFT, DPO, GRPO, PPO).
"""

import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.config import PACKED_DIR, PACKED_FILES

EVAL_ITERS = 100  # batches averaged per eval-loss estimate (monitoring precision)


def resolve_device_dtype() -> tuple[str, torch.dtype]:
    """Auto-detect the device and a supported autocast dtype (bf16, downgrading if needed)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    if not (device.startswith("cuda") and torch.cuda.is_bf16_supported()):
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return device, dtype


def setup_amp() -> tuple[str, object, "torch.amp.GradScaler"]:
    """Resolve the device and return `(device, autocast_ctx, grad_scaler)`.

    `ctx` is only enabled for fp16 (used for controllable ranges), 
    whilst bf16 needs no loss scaling and is better for stability. 
    """
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(pt_dtype is torch.float16))
    return device, ctx, scaler


def cosine_lr(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    """
    Linear warmup then cosine decay from `peak` to `floor` over `total` steps.

    Passing `floor == peak` gives warmup, then constant lr (what GRPO/PPO use).
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return floor
    ratio = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (peak - floor)


def optimizer_step(scaler, updates, grad_clip: float) -> None:
    """
    Unscale, clip, step, update, and zero after the caller's `backward()`.

    `updates` is a list of `(optimizer, params)` pairs - one for most stages
    and two for PPO (actor + critic, sharing one scaler).
    """
    if grad_clip and grad_clip > 0:
        for opt, _ in updates:
            scaler.unscale_(opt)
        for _, params in updates:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
    for opt, _ in updates:
        scaler.step(opt)
    scaler.update()
    for opt, _ in updates:
        opt.zero_grad(set_to_none=True)


def get_batch(
    split: str, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of random windows from a packed split.

    The memmap is re-opened each call: this avoids a memory leak where the mapping
    accumulates references across a long training run.

    Returns:
        `(x, y)` of shape `(batch_size, block_size)`, `y` is shifted by one.
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
    model, weight_decay: float, lr: float, betas: tuple[float, float], device: str
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


@torch.no_grad()
def estimate_loss(model, ctx, block_size: int, device: str, batch_size: int) -> dict[str, float]:
    """Average loss over `EVAL_ITERS` batches for each split."""
    out = {}
    model.eval()
    for split in ("train", "valid"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
