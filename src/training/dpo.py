"""
Direct Preference Optimization (DPO) on the SFT checkpoint.

Offline preference learning with no reward model, no sampling loop, and no critic. The
policy is initialized from SFT and trained against a *frozen* reference clone of
the same SFT model, on ``(chosen, rejected)`` pairs from ``make_preferences.py``.
"""

import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.config import (
    DATA_DIR,
    SFT_CKPT_DIR,
    DPO_CKPT_DIR,
    DPO_DATA_DIR,
    PAIRS_FILE,
    SEED,
)
from src.data.dpo_data import DPODataset, dpo_collate
from src.eval.metrics import log_metrics, plot_series
from src.models.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    new_run_dir,
    resolve_checkpoint,
)
from src.models.tokenizer import Tokenizer
from src.training.common import setup_amp, cosine_lr, optimizer_step, configure_optimizers
from src.training.rl_common import sequence_logprob


@dataclass
class DPOConfig:
    """
    DPO trainer hyperparameters (init from an SFT run).
    """

    init_run: str | None = None   # SFT run to init policy + frozen reference from (None = latest)
    val_fraction: float = 0.05    # fraction of pairs held out for val loss / reward accuracy
    beta: float = 0.3             # KL strength in the DPO objective (higher = stay closer to ref)
    batch_size: int = 16          # pairs per step (2x sequences forwarded: chosen + rejected)
    epochs: int = 1               # 2+ epochs over-optimized (+42% val ppl, reward hacking); 1 is safe
    lr: float = 1e-5              # DPO is sensitive; well below the SFT peak
    min_lr: float = 1e-6
    warmup_steps: int = 50
    weight_decay: float = 0.0
    beta1: float = 0.9   # Adam betas
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_interval: int = 20
    patience: int = 2             # early-stop after this many epochs without val improvement


def dpo_loss(policy, reference, x, y, beta, ctx):
    """
    DPO loss + diagnostics for one `(2B, T)` batch.
    The batch arranges with all chosen pairs first, then rejected for easy splitting.

    Returns `(loss, accuracy, margin)` where accuracy is the fraction of pairs
    the policy already prefers correctly and margin is the mean reward gap.
    """
    b = x.size(0) // 2
    with ctx:
        pol = sequence_logprob(policy, x, y)
        with torch.no_grad():
            ref = sequence_logprob(reference, x, y)
    pol_c, pol_r = pol[:b], pol[b:]
    ref_c, ref_r = ref[:b], ref[b:]
    logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))  # (B,)
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean()
    margin = logits.mean() / beta
    return loss, acc, margin


@torch.no_grad()
def evaluate_dpo(policy, reference, loader, beta, ctx, device):
    """Mean DPO loss, reward accuracy, and margin over a loader."""
    policy.eval()
    tot_loss = tot_acc = tot_margin = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss, acc, margin = dpo_loss(policy, reference, x, y, beta, ctx)
        bsz = x.size(0) // 2
        tot_loss += loss.item() * bsz
        tot_acc += acc.item() * bsz
        tot_margin += margin.item() * bsz
        n += bsz
    policy.train()
    n = max(1, n)
    return tot_loss / n, tot_acc / n, tot_margin / n


def train_dpo(cfg: DPOConfig = DPOConfig()) -> None:
    """Run epoch-based DPO with per-epoch eval and best/last checkpoints with early stopping."""
    torch.manual_seed(SEED)
    device, ctx, scaler = setup_amp()

    # Policy (trained) and reference (frozen) both start from the SFT checkpoint.
    init_ckpt = resolve_checkpoint(cfg.init_run, "best.pt", SFT_CKPT_DIR)
    policy, _ = load_checkpoint(init_ckpt, device)
    reference, _ = load_checkpoint(init_ckpt, device)
    reference.eval()
    reference.requires_grad_(False)
    gpt_cfg = policy.cfg
    optimizer = configure_optimizers(
        policy, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
    )
    print(
        f"DPO init from {init_ckpt} ({policy.num_params():,} non-embedding params) "
        f"| beta={cfg.beta} on {device}"
    )

    # Data: preference pairs -> masked (chosen, rejected) examples, split for val.
    tok = Tokenizer()
    max_len = gpt_cfg.block_size
    pairs_path = Path(DPO_DATA_DIR) / PAIRS_FILE
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"{pairs_path} not found -- run `python -m src.make_preferences` first"
        )
    full_ds = DPODataset(pairs_path, tok, max_len)
    n_val = max(1, int(len(full_ds) * cfg.val_fraction)) if len(full_ds) > 1 else 0
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    collate = partial(dpo_collate, pad_id=tok.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        if n_val
        else None
    )

    total_steps = len(train_loader) * cfg.epochs
    run_dir = new_run_dir(DPO_CKPT_DIR)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    print(
        f"run dir: {run_dir} | {n_train:,} train / {n_val:,} val pairs | {total_steps:,} steps"
    )

    best_val = float("inf")
    no_improve = 0
    global_step = 0
    policy.train()
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ep_loss_sum = ep_acc_sum = 0.0
        ep_steps = 0
        for x, y in train_loader:
            lr = cosine_lr(global_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = x.to(device), y.to(device)
            loss, acc, margin = dpo_loss(policy, reference, x, y, cfg.beta, ctx)
            ep_loss_sum += loss.item()
            ep_acc_sum += acc.item()
            ep_steps += 1

            scaler.scale(loss).backward()
            optimizer_step(scaler, [(optimizer, policy.parameters())], cfg.grad_clip)

            if global_step % cfg.log_interval == 0:
                left = total_steps - global_step
                print(
                    f"epoch {epoch} step {global_step:>6}/{total_steps} ({left:,} left): "
                    f"loss {loss.item():.4f} | acc {acc.item():.3f} | "
                    f"margin {margin.item():+.3f} | lr {lr:.2e}"
                )
                log_metrics(run_dir, {
                    "step": global_step,
                    "loss": round(loss.item(), 4),
                    "acc": round(acc.item(), 4),
                    "margin": round(margin.item(), 4),
                    "lr": lr,
                })
            global_step += 1

        train_loss = ep_loss_sum / max(1, ep_steps)
        train_acc = ep_acc_sum / max(1, ep_steps)
        if val_loader is not None:
            val_loss, val_acc, val_margin = evaluate_dpo(
                policy, reference, val_loader, cfg.beta, ctx, device
            )
        else:  # too few pairs to hold any out -- fall back to train metrics
            val_loss, val_acc, val_margin = train_loss, train_acc, 0.0
        print(
            f"epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} margin {val_margin:+.3f} | "
            f"{(time.time() - start) / 60:.1f} min"
        )
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(best_path, policy, optimizer, global_step, best_val, gpt_cfg)
            print(f"  new best val {best_val:.4f} - saved {best_path}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"early stopping after {no_improve} epochs without improvement")
                save_checkpoint(last_path, policy, optimizer, global_step, best_val, gpt_cfg)
                break
        save_checkpoint(last_path, policy, optimizer, global_step, best_val, gpt_cfg)

    png = plot_series(run_dir, "step", [
        ("DPO loss", ["loss"]),
        ("reward margin", ["margin"]),
        ("pref accuracy", ["acc"]),
    ])
    print(
        f"done. best val loss {best_val:.4f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train_dpo()
