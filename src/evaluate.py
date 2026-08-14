import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.config import (
    PACKED_DIR,
    PACKED_FILES,
    CONTEXT_LEN,
    CKPT_DIR,
    DEVICE,
    DTYPE,
    EVAL_BATCH_SIZE,
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
    model, split: str, block_size: int, batch_size: int, device: str, ctx
) -> dict[str, float]:
    """Token-weighted mean loss and perplexity over a whole split.

    Perplexity is exp(mean cross-entropy), lower is better.
    It's the average per-token branching factor, so a uniform model would score = vocab_size.

    Args:
        model: A model in eval mode.
        split: "train" or`"valid".
        block_size: Window length.
        batch_size: Windows per forward pass.
        device: Target device.
        ctx: Autocast context manager.

    Returns:
        {"loss": ..., "perplexity": ...}.
    """
    data = np.memmap(Path(PACKED_DIR) / PACKED_FILES[split], dtype=np.uint16, mode="r")
    n_windows = (len(data) - 1) // block_size
    total_loss = 0.0
    total_tokens = 0

    for b0 in range(0, n_windows, batch_size):
        idxs = range(b0, min(b0 + batch_size, n_windows))
        xb = torch.stack(
            [
                torch.from_numpy(
                    data[i * block_size : i * block_size + block_size].astype(np.int64)
                )
                for i in idxs
            ]
        )
        yb = torch.stack(
            [
                torch.from_numpy(
                    data[i * block_size + 1 : i * block_size + block_size + 1].astype(
                        np.int64
                    )
                )
                for i in idxs
            ]
        )
        xb, yb = xb.to(device), yb.to(device)
        with ctx:
            _, loss = model(xb, yb)
        total_loss += loss.item() * yb.numel()  # weight by token count
        total_tokens += yb.numel()

    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}


def evaluate() -> None:
    """Evaluate a trained checkpoint: validation loss and perplexity."""
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
        r = evaluate_split(model, split, CONTEXT_LEN, EVAL_BATCH_SIZE, device, ctx)
        print(f"{split}: loss {r['loss']:.4f} | perplexity {r['perplexity']:.2f}")


if __name__ == "__main__":
    evaluate()
