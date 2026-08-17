"""Evaluate a trained checkpoint: validation loss and perplexity.

Perplexity is ``exp(mean cross-entropy)`` -- the average per-token branching
factor. Lower is better; a uniform model over the vocab would score ``vocab_size``.
The sweep is deterministic: it walks non-overlapping windows across the whole
split (not random batches), so the number is stable run to run.
"""

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.config import (
    PACKED_DIR,
    PACKED_FILES,
    META_FILE,
    CONTEXT_LEN,
    CKPT_DIR,
    CKPT_RUN,
    DEVICE,
    DTYPE,
    EVAL_BATCH_SIZE,
    EVAL_MAX_BATCHES,
)
from src.models.checkpoints import load_checkpoint, resolve_checkpoint

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_device_dtype() -> tuple[str, torch.dtype]:
    """Pick device and a supported autocast dtype (mirrors train.py)."""
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _DTYPES[DTYPE]
    if dtype is torch.bfloat16 and not (
        device.startswith("cuda") and torch.cuda.is_bf16_supported()
    ):
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if dtype is torch.float16 and not device.startswith("cuda"):
        dtype = torch.float32
    return device, dtype


@torch.no_grad()
def evaluate_split(
    model,
    split: str,
    block_size: int,
    batch_size: int,
    device: str,
    ctx,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Token-weighted mean loss and perplexity over a split.

    With ``max_batches`` set, evaluates that many batches of windows spread
    evenly across the whole split (a representative, deterministic subsample) --
    far cheaper than a full sweep on the ~100x-larger train split. With None,
    sweeps every non-overlapping window.

    Args:
        model: A model in eval mode.
        split: ``"train"`` or ``"valid"``.
        block_size: Window length.
        batch_size: Windows per forward pass.
        device: Target device.
        ctx: Autocast context manager.
        max_batches: Cap on batches, or None for the full sweep.

    Returns:
        ``{"loss": ..., "perplexity": ...}``.
    """
    data = np.memmap(Path(PACKED_DIR) / PACKED_FILES[split], dtype=np.uint16, mode="r")
    n_windows = (len(data) - 1) // block_size

    n_select = n_windows
    if max_batches is not None and max_batches * batch_size < n_windows:
        n_select = max_batches * batch_size
    # evenly spaced window indices across the whole split (deterministic, representative)
    starts = np.linspace(0, n_windows - 1, n_select).astype(np.int64)

    total_loss = 0.0
    total_tokens = 0
    for b0 in range(0, len(starts), batch_size):
        sel = starts[b0 : b0 + batch_size]
        xb = torch.stack(
            [
                torch.from_numpy(
                    data[s * block_size : s * block_size + block_size].astype(np.int64)
                )
                for s in sel
            ]
        )
        yb = torch.stack(
            [
                torch.from_numpy(
                    data[s * block_size + 1 : s * block_size + block_size + 1].astype(
                        np.int64
                    )
                )
                for s in sel
            ]
        )
        xb, yb = xb.to(device), yb.to(device)
        with ctx:
            _, loss = model(xb, yb)
        total_loss += loss.item() * yb.numel()  # weight by token count
        total_tokens += yb.numel()

    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}


def main() -> None:
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )

    ckpt_path = resolve_checkpoint(CKPT_RUN, "best.pt", CKPT_DIR)
    model, ckpt = load_checkpoint(ckpt_path, device)
    model.eval()
    print(f"loaded {ckpt_path} (trained {ckpt['step']} steps)")

    for split in ("valid", "train"):
        r = evaluate_split(
            model, split, CONTEXT_LEN, EVAL_BATCH_SIZE, device, ctx, EVAL_MAX_BATCHES
        )
        print(f"{split}: loss {r['loss']:.4f} | perplexity {r['perplexity']:.2f}")


if __name__ == "__main__":
    main()
