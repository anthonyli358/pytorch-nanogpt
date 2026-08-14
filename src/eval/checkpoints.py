from dataclasses import asdict
from pathlib import Path

import torch

from src.config import GPTConfig
from src.models.gpt import GPT


def save_checkpoint(path: Path, model: GPT, optimizer, step: int,
                    best_val: float, cfg: GPTConfig) -> None:
    """Write a resumable checkpoint (unwrapping torch.compile if present).

    Bundles the model weights, optimizer state, step, best val loss, and the GPTConfig.

    Args:
        path: Destination .pt file.
        model: The model (compiled or raw).
        optimizer: The optimizer whose state to save.
        step: Current optimizer step.
        best_val: Best validation loss seen so far.
        cfg: The model config, stored via asdict for a clean rebuild.
    """
    raw = getattr(model, "_orig_mod", model)   # unwrap compiled model for clean keys
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val,
        "cfg": asdict(cfg),
    }, path)


def load_checkpoint(path: Path, device: str) -> tuple[GPT, dict]:
    """
    Rebuild the model from a checkpoint and load its weights.
    No dependence on current config values.

    Args:
        path: Checkpoint .pt file.
        device: Device to map tensors onto.

    Returns:
        (model, ckpt) where model is on device with weights loaded,
        and ckpt is the raw dict (for optimizer state, step, etc.).
    """
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt