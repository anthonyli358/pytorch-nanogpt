"""GRPO: online RL with verifiable + shaped rewards (step 12).

Group Relative Policy Optimization -- PPO minus the critic. Each step:

1. **Rollout.** Sample a batch of instruct prompts; for each, sample a *group* of
   ``G`` completions from the current policy.
2. **Reward.** Score every completion with the shaped reward (verifiable word
   inclusion minus a repetition penalty -- ``src.reward.shaped_reward``).
3. **Advantage.** Normalize each reward against its group: ``A = (r - mean) / std``.
   No value network -- the group mean is the baseline. A group whose completions
   all score the same has zero advantage and contributes no gradient.
4. **Update.** A clipped PPO surrogate on the response tokens, with a per-token
   KL leash to the frozen reference (the SFT model), so the policy improves reward
   without drifting into the reward-hacking degeneracy step 13 catches.

Reuses the sampling, reward, masking, and checkpoint machinery. Policy and the
frozen reference both init from ``GRPOConfig.init_dir`` (SFT by default; point it at
``checkpoints/dpo`` to continue from DPO). Deliverable: ``checkpoints/grpo/<run>``.
"""

import time
from dataclasses import dataclass

import torch

from src.config import (
    GRPO_CKPT_DIR,
    SFT_CKPT_DIR,
    REP_NGRAM,
    REP_WEIGHT,
    SEED,
)
from src.data.common import pad_batch
from src.eval.metrics import log_metrics, plot_series
from src.models.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    new_run_dir,
    resolve_checkpoint,
)
from src.models.tokenizer import Tokenizer
from src.reward import shaped_reward
from src.training.common import setup_amp, cosine_lr, optimizer_step, configure_optimizers
from src.training.rl_common import token_logprobs, load_prompt_pool, build_rollout_example


@dataclass
class GRPOConfig:
    """GRPO trainer hyperparameters (online RL; init from an SFT run).

    Shared betas and the reward-shaping n-gram (REP_NGRAM) stay in src/config.py.
    """

    init_dir: str = SFT_CKPT_DIR   # dir to resolve the init from (SFT for canonical GRPO; DPO to continue)
    init_run: str | None = None    # policy + frozen reference init (None = latest run under init_dir)
    prompt_pool: int = 20_000      # instruct-train prompts (with Words:) pre-loaded to sample rollouts from
    group_size: int = 8            # completions per prompt (the group the advantage normalizes within)
    prompts_per_step: int = 16     # prompts per rollout; rollout batch = this * group_size sequences
    steps: int = 400               # optimizer steps (rollouts)
    inner_epochs: int = 1          # PPO update passes per rollout (1 = pure on-policy, ratio ~ 1)
    lr: float = 1e-6               # RL is delicate; well below SFT/DPO
    warmup_steps: int = 20
    weight_decay: float = 0.0
    beta1: float = 0.9   # Adam betas
    beta2: float = 0.95
    grad_clip: float = 1.0
    beta: float = 0.04             # KL(policy || reference) penalty weight
    clip_eps: float = 0.2          # PPO ratio clip range (1 +/- eps)
    temperature: float = 1.0       # rollout sampling temperature (> 0 for group diversity)
    top_k: int = 200
    top_p: float = 0.95
    max_new_tokens: int = 200      # cap on rollout completion length (also bounded by remaining context)
    rep_weight: float = REP_WEIGHT  # repetition-penalty weight in the rollout reward (0 = verifiable only)
    log_interval: int = 5
    ckpt_interval: int = 50        # save last.pt every N steps; best.pt on best smoothed reward
    seed: int = SEED


@torch.no_grad()
def rollout(
    policy, tok, gpt_cfg, pool, ctx, device, cfg: GRPOConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Sample groups, score them, and build the padded update batch.

    Returns ``(x, y, advantages, stats)`` where ``x, y`` are ``(N, T)`` with
    ``N = prompts_per_step * group_size``, ``advantages`` is ``(N,)``, and stats
    holds mean reward / active-group fraction / response length for logging.
    """
    g = cfg.group_size
    idx = torch.randint(len(pool), (cfg.prompts_per_step,)).tolist()
    examples, advs = [], []
    reward_sum = active = resp_tokens = 0.0

    for pi in idx:
        prompt_text, prompt_ids = pool[pi]
        n_new = min(cfg.max_new_tokens, gpt_cfg.block_size - len(prompt_ids))
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).expand(g, -1)
        with ctx:
            out = policy.generate(
                x, n_new, temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p
            )
        rewards = []
        group_examples = []
        for row in out.tolist():
            gen = row[len(prompt_ids):]
            inp, lab, story = build_rollout_example(prompt_ids, gen, tok.eos_id, tok)
            r = shaped_reward(prompt_text, story, rep_weight=cfg.rep_weight, n=REP_NGRAM)
            rewards.append(r if r is not None else 0.0)
            group_examples.append((inp, lab))

        r = torch.tensor(rewards, dtype=torch.float32)
        reward_sum += r.mean().item()
        std = r.std()
        if std > 1e-6:
            active += 1
        adv = (r - r.mean()) / (std + 1e-6)  # group-relative advantage
        for (inp, lab), a in zip(group_examples, adv.tolist()):
            examples.append((torch.tensor(inp), torch.tensor(lab)))
            advs.append(a)
            resp_tokens += sum(1 for t in lab if t != -1)

    x, y = pad_batch(examples, pad_id=tok.pad_id)
    advantages = torch.tensor(advs, dtype=torch.float32)
    stats = {
        "reward": reward_sum / cfg.prompts_per_step,
        "active_frac": active / cfg.prompts_per_step,
        "resp_len": resp_tokens / len(examples),
    }
    return x.to(device), y.to(device), advantages.to(device), stats


def grpo_step(
    policy, reference, x, y, advantages, old_logp, ref_logp, ctx, cfg: GRPOConfig
) -> tuple[torch.Tensor, float]:
    """One clipped-surrogate + KL update pass; returns ``(loss, mean_kl)``."""
    with ctx:
        logp, mask = token_logprobs(policy, x, y)
    ratio = torch.exp(logp - old_logp)                 # (N, T); ~1 on the first inner pass
    adv = advantages.unsqueeze(1)                       # (N, 1) broadcast over tokens
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)
    delta = ref_logp - logp                            # KL(policy || ref), k3 estimator
    kl = torch.exp(delta) - delta - 1.0
    per_tok = surr - cfg.beta * kl
    n = mask.sum().clamp(min=1)
    loss = -(per_tok * mask).sum() / n
    mean_kl = (kl.detach() * mask).sum() / n
    return loss, mean_kl.item()


def train_grpo(cfg: GRPOConfig = GRPOConfig()) -> None:
    """Run the GRPO loop: rollout, group-normalized advantage, clipped PPO step."""
    torch.manual_seed(cfg.seed)
    device, ctx, scaler = setup_amp()

    init_ckpt = resolve_checkpoint(cfg.init_run, "best.pt", cfg.init_dir)
    policy, _ = load_checkpoint(init_ckpt, device)
    reference, _ = load_checkpoint(init_ckpt, device)
    reference.eval()
    reference.requires_grad_(False)
    gpt_cfg = policy.cfg
    optimizer = configure_optimizers(policy, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device)
    tok = Tokenizer()
    max_prompt = gpt_cfg.block_size - cfg.max_new_tokens
    pool = load_prompt_pool(tok, max_prompt, cfg.prompt_pool)
    print(
        f"GRPO init from {init_ckpt} ({policy.num_params():,} non-embedding params) | "
        f"G={cfg.group_size} prompts/step={cfg.prompts_per_step} beta={cfg.beta} on {device}"
    )

    run_dir = new_run_dir(GRPO_CKPT_DIR)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    print(f"run dir: {run_dir} | {cfg.steps:,} steps")

    ema_reward = None
    best_reward = -float("inf")
    start = time.time()
    policy.train()

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.warmup_steps, cfg.steps, cfg.lr, cfg.lr)  # warmup then constant
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y, advantages, stats = rollout(policy, tok, gpt_cfg, pool, ctx, device, cfg)
        with torch.no_grad():
            old_logp, _ = token_logprobs(policy, x, y)
            ref_logp, _ = token_logprobs(reference, x, y)

        loss_val = kl_val = 0.0
        for _ in range(cfg.inner_epochs):
            loss, mean_kl = grpo_step(policy, reference, x, y, advantages, old_logp, ref_logp, ctx, cfg)
            scaler.scale(loss).backward()
            optimizer_step(scaler, [(optimizer, policy.parameters())], cfg.grad_clip)
            loss_val, kl_val = loss.item(), mean_kl

        ema_reward = stats["reward"] if ema_reward is None else 0.9 * ema_reward + 0.1 * stats["reward"]

        if step % cfg.log_interval == 0:
            left = cfg.steps - step
            elapsed = (time.time() - start) / 60
            eta = left / ((step + 1) / max(1e-9, elapsed)) if step else 0.0
            print(
                f"step {step:>4}/{cfg.steps} ({left:,} left): reward {stats['reward']:.3f} "
                f"(ema {ema_reward:.3f}) | KL {kl_val:.3f} | active {stats['active_frac']:.2f} | "
                f"resp {stats['resp_len']:.0f}t | loss {loss_val:+.4f} | lr {lr:.2e} | "
                f"{elapsed:.1f} min, ~{eta:.1f} left"
            )
            log_metrics(run_dir, {
                "step": step,
                "reward": round(stats["reward"], 4),
                "ema_reward": round(ema_reward, 4),
                "kl": round(kl_val, 4),
                "active_frac": round(stats["active_frac"], 3),
                "resp_len": round(stats["resp_len"], 1),
                "loss": round(loss_val, 4),
                "lr": lr,
            })

        if ema_reward > best_reward:
            best_reward = ema_reward
            save_checkpoint(best_path, policy, optimizer, step, best_reward, gpt_cfg)
        if step % cfg.ckpt_interval == 0 or step == cfg.steps - 1:
            save_checkpoint(last_path, policy, optimizer, step, best_reward, gpt_cfg)

    png = plot_series(run_dir, "step", [
        ("reward", ["reward", "ema_reward"]),
        ("KL(policy||ref)", ["kl"]),
        ("active groups", ["active_frac"]),
    ])
    print(
        f"done. best ema reward {best_reward:.3f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train_grpo()
