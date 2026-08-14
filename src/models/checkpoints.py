from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from src.config import GPTConfig
from src.models.gpt import GPT


def new_run_dir(base: str = "checkpoints") -> Path:
    """Create and return a fresh timestamped run directory: ``base/YYYYmmdd_HHMMSS``.

    If a directory for the current second already exists, a numeric suffix is
    appended so concurrent/rapid runs never collide.
    """
    base_path = Path(base)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = base_path / stamp
    i = 1
    while run.exists():
        run = base_path / f"{stamp}_{i}"
        i += 1
    run.mkdir(parents=True, exist_ok=True)
    return run


def latest_run_dir(base: str = "checkpoints") -> Path | None:
    """Return the most recent run directory under ``base``, or None if there are none.

    Timestamp names are zero-padded, so lexical sort is chronological.
    """
    base_path = Path(base)
    if not base_path.exists():
        return None
    runs = sorted(d for d in base_path.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def resolve_checkpoint(spec, name: str = "best.pt", base: str = "checkpoints") -> Path:
    """Resolve a checkpoint path from a flexible spec.

    Args:
        spec: ``None`` -> ``<latest run>/<name>``; a directory -> ``<dir>/<name>``;
            a file path -> that file as-is.
        name: Checkpoint filename to use when ``spec`` is None or a directory.
        base: Root checkpoints directory.

    Returns:
        The resolved checkpoint path.
    """
    if spec is None:
        run = latest_run_dir(base)
        if run is None:
            raise FileNotFoundError(f"no run directories under {base!r}")
        return run / name
    p = Path(spec)
    return p / name if p.is_dir() else p


def save_checkpoint(
    path: Path, model: GPT, optimizer, step: int, best_val: float, cfg: GPTConfig
) -> None:
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
    raw = getattr(model, "_orig_mod", model)  # unwrap compiled model for clean keys
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val,
            "cfg": asdict(cfg),
        },
        path,
    )


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
