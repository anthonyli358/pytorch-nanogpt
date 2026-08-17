"""Direct Preference Optimization (DPO) on the SFT checkpoint (step 10).

Offline preference learning: no reward model, no sampling loop, no critic. The
policy is initialized from SFT and trained against a *frozen* reference clone of
the same SFT model, on ``(chosen, rejected)`` pairs from ``make_preferences.py``.

The objective is
    L = -log sigmoid( beta * [ (logp_pi(y_w) - logp_ref(y_w))
                             - (logp_pi(y_l) - logp_ref(y_l)) ] )
where ``logp(y)`` is the summed log-prob of the response tokens (``sequence_logprob``),
using the same prompt masking as SFT. Raising a chosen completion's advantage
over the reference while lowering the rejected one's; the reference term bakes a
KL leash into the loss so the policy can't wander far from SFT.

Reuses the optimizer / cosine-schedule / autocast / checkpoint machinery from the
pretraining and SFT loops. Deliverable: ``dpo_checkpoints/<run>/best.pt`` (``dpo.pt``).
"""

import math
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.config import (
    DATA_DIR,
    SFT_CKPT_DIR,
    DPO_CKPT_DIR,
    DPO_INIT_RUN,
    DPO_DATA_DIR,
    PAIRS_FILE,
    DPO_MAX_LEN,
    DPO_VAL_FRACTION,
    DPO_BETA,
    DPO_BATCH_SIZE,
    DPO_EPOCHS,
    DPO_LR,
    DPO_MIN_LR,
    DPO_WARMUP_STEPS,
    DPO_WEIGHT_DECAY,
    DPO_GRAD_CLIP,
    DPO_LOG_INTERVAL,
    DPO_PATIENCE,
    BETA1,
    BETA2,
    SEED,
)
from src.data.dpo_data import DPODataset, dpo_collate
from src.eval.metrics import log_metrics, plot_losses
from src.models.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    new_run_dir,
    resolve_checkpoint,
)
from src.models.tokenizer import Tokenizer
from src.train import configure_optimizers, resolve_device_dtype

from pathlib import Path


def dpo_get_lr(step: int, total_steps: int) -> float:
    """Cosine schedule with linear warmup over the full DPO run."""
    if step < DPO_WARMUP_STEPS:
        return DPO_LR * (step + 1) / DPO_WARMUP_STEPS
    if step >= total_steps:
        return DPO_MIN_LR
    ratio = (step - DPO_WARMUP_STEPS) / max(1, total_steps - DPO_WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return DPO_MIN_LR + coeff * (DPO_LR - DPO_MIN_LR)


def sequence_logprob(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Summed log-prob of the response tokens for each sequence in the batch.

    ``y`` is the shifted next-token target with ``-1`` over the prompt and padding
    (the SFT masking), so the sum covers exactly the response span (+EOS). Passing
    ``y`` as targets makes ``forward`` return the full ``(B, T, vocab)`` logits.

    Returns:
        Tensor of shape ``(B,)`` -- one summed response log-prob per sequence.
    """
    logits, _ = model(x, y)  # (B, T, vocab); the returned loss is unused here
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = y != -1
    gather_idx = y.clamp(min=0).unsqueeze(-1)  # -1 -> 0 so gather is in range; masked out below
    token_logp = torch.gather(logp, -1, gather_idx).squeeze(-1)  # (B, T)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(policy, reference, x, y, beta, ctx):
    """DPO loss + diagnostics for one ``(2B, T)`` batch (chosen first, then rejected).

    Returns ``(loss, accuracy, margin)`` where accuracy is the fraction of pairs
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


def train_dpo() -> None:
    """Run epoch-based DPO with per-epoch eval, best/last checkpoints, early stop."""
    torch.manual_seed(SEED)
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(pt_dtype is torch.float16))

    # Policy (trained) and reference (frozen) both start from the SFT checkpoint.
    init_ckpt = resolve_checkpoint(DPO_INIT_RUN, "best.pt", SFT_CKPT_DIR)
    policy, _ = load_checkpoint(init_ckpt, device)
    reference, _ = load_checkpoint(init_ckpt, device)
    reference.eval()
    reference.requires_grad_(False)
    cfg = policy.cfg
    optimizer = configure_optimizers(
        policy, DPO_WEIGHT_DECAY, DPO_LR, (BETA1, BETA2), device
    )
    print(
        f"DPO init from {init_ckpt} ({policy.num_params():,} non-embedding params) "
        f"| beta={DPO_BETA} on {device}"
    )

    # Data: preference pairs -> masked (chosen, rejected) examples, split for val.
    tok = Tokenizer()
    max_len = min(DPO_MAX_LEN or cfg.block_size, cfg.block_size)
    pairs_path = Path(DPO_DATA_DIR) / PAIRS_FILE
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"{pairs_path} not found -- run `python -m src.make_preferences` first"
        )
    full_ds = DPODataset(pairs_path, tok, max_len)
    n_val = max(1, int(len(full_ds) * DPO_VAL_FRACTION)) if len(full_ds) > 1 else 0
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    collate = partial(dpo_collate, pad_id=tok.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=DPO_BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    val_loader = (
        DataLoader(val_ds, batch_size=DPO_BATCH_SIZE, shuffle=False, collate_fn=collate)
        if n_val
        else None
    )

    total_steps = len(train_loader) * DPO_EPOCHS
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

    for epoch in range(1, DPO_EPOCHS + 1):
        ep_loss_sum = ep_acc_sum = 0.0
        ep_steps = 0
        for x, y in train_loader:
            lr = dpo_get_lr(global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = x.to(device), y.to(device)
            loss, acc, margin = dpo_loss(policy, reference, x, y, DPO_BETA, ctx)
            ep_loss_sum += loss.item()
            ep_acc_sum += acc.item()
            ep_steps += 1

            scaler.scale(loss).backward()
            if DPO_GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), DPO_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if global_step % DPO_LOG_INTERVAL == 0:
                left = total_steps - global_step
                print(
                    f"epoch {epoch} step {global_step:>6}/{total_steps} ({left:,} left): "
                    f"loss {loss.item():.4f} | acc {acc.item():.3f} | "
                    f"margin {margin.item():+.3f} | lr {lr:.2e}"
                )
            global_step += 1

        train_loss = ep_loss_sum / max(1, ep_steps)
        train_acc = ep_acc_sum / max(1, ep_steps)
        if val_loader is not None:
            val_loss, val_acc, val_margin = evaluate_dpo(
                policy, reference, val_loader, DPO_BETA, ctx, device
            )
        else:  # too few pairs to hold any out -- fall back to train metrics
            val_loss, val_acc, val_margin = train_loss, train_acc, 0.0
        print(
            f"epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} margin {val_margin:+.3f} | "
            f"{(time.time() - start) / 60:.1f} min"
        )
        log_metrics(
            run_dir,
            {
                "epoch": epoch,
                "step": global_step,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_margin": round(val_margin, 4),
                "lr": lr,
            },
        )

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(best_path, policy, optimizer, global_step, best_val, cfg)
            print(f"  new best val {best_val:.4f} - saved {best_path}")
        else:
            no_improve += 1
            if no_improve >= DPO_PATIENCE:
                print(f"early stopping after {no_improve} epochs without improvement")
                save_checkpoint(last_path, policy, optimizer, global_step, best_val, cfg)
                break
        save_checkpoint(last_path, policy, optimizer, global_step, best_val, cfg)

    png = plot_losses(run_dir, x="epoch")
    print(
        f"done. best val loss {best_val:.4f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train_dpo()
