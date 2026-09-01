import math
from pathlib import Path

import numpy as np
import torch

from src.config import PACKED_DIR, PACKED_FILES


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
    """
    Evaluate a trained checkpoint: validation loss and perplexity.

    - Perplexity is exp(mean cross-entropy) - the average per-token branching
        factor. Lower is better.

    With max_batches set, evaluates that many batches of windows spread
    evenly across the whole split. Cheaper than a full sweep on the ~100x-larger train split.
    With None, sweeps every non-overlapping window.

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
