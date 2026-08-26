"""PPO: the full classic actor-critic stack (step 12, the education path).

Everything GRPO drops, added back. Each step:

1. **Rollout.** Sample prompts, one completion each from the policy; record the
   old per-token log-probs, the reference log-probs, and the critic's per-token
   values.
2. **Reward.** A per-token KL penalty ``-beta*(logp_policy - logp_ref)`` at every
   response token, plus the scalar shaped reward added at the final (EOS) token --
   the standard RLHF token-reward shaping.
3. **GAE.** Generalized Advantage Estimation over the response tokens using the
   critic's value baseline (this is what GRPO replaces with a group mean).
4. **Update.** ``INNER_EPOCHS`` passes of a clipped actor surrogate + a clipped
   value loss, actor and critic on separate optimizers/LRs.

The critic is a second GPT backbone (init from SFT) with a scalar value head
(``ValueModel``). Reward model (step 11) is skipped: PPO optimizes the verifiable
reward directly. Shares the rollout/log-prob/reward machinery with GRPO.
Deliverable: ``checkpoints/ppo/<run>/best.pt`` (the policy; the critic is training
scaffolding and isn't persisted).
"""

import time

import torch
import torch.nn as nn

from src.config import (
    PPO_CKPT_DIR,
    PPO_INIT_DIR,
    PPO_INIT_RUN,
    PPO_PROMPT_POOL,
    PPO_PROMPTS_PER_STEP,
    PPO_STEPS,
    PPO_INNER_EPOCHS,
    PPO_LR,
    PPO_VALUE_LR,
    PPO_WARMUP_STEPS,
    PPO_WEIGHT_DECAY,
    PPO_GRAD_CLIP,
    PPO_CLIP_EPS,
    PPO_VF_COEF,
    PPO_KL_BETA,
    PPO_GAMMA,
    PPO_LAM,
    PPO_TEMPERATURE,
    PPO_TOP_K,
    PPO_TOP_P,
    PPO_MAX_NEW_TOKENS,
    PPO_REP_WEIGHT,
    PPO_LOG_INTERVAL,
    PPO_CKPT_INTERVAL,
    PPO_SEED,
    REP_NGRAM,
    BETA1,
    BETA2,
)
from src.data.common import pad_batch
from src.eval.metrics import log_metrics, plot_series
from src.training.rl_common import token_logprobs, load_prompt_pool, build_rollout_example
from src.models.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    new_run_dir,
    resolve_checkpoint,
)
from src.models.gpt import GPT
from src.models.tokenizer import Tokenizer
from src.reward import shaped_reward
from src.training.common import setup_amp, cosine_lr, optimizer_step, configure_optimizers


class ValueModel(nn.Module):
    """Critic: a GPT backbone with a scalar value head over each position."""

    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(gpt.cfg.d_model, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.gpt.hidden(idx)).squeeze(-1)  # (B, T)


@torch.no_grad()
def rollout(
    policy, value_model, reference, tok, cfg, pool, ctx, device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Sample one completion per prompt; return the padded batch + rollout tensors."""
    idx = torch.randint(len(pool), (PPO_PROMPTS_PER_STEP,)).tolist()
    examples, rewards = [], []
    for pi in idx:
        prompt_text, prompt_ids = pool[pi]
        n_new = min(PPO_MAX_NEW_TOKENS, cfg.block_size - len(prompt_ids))
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
        with ctx:
            out = policy.generate(
                x, n_new, temperature=PPO_TEMPERATURE, top_k=PPO_TOP_K, top_p=PPO_TOP_P
            )
        gen = out[0].tolist()[len(prompt_ids):]
        inp, lab, story = build_rollout_example(prompt_ids, gen, tok.eos_id, tok)
        r = shaped_reward(prompt_text, story, rep_weight=PPO_REP_WEIGHT, n=REP_NGRAM)
        examples.append((torch.tensor(inp), torch.tensor(lab)))
        rewards.append(r if r is not None else 0.0)

    x, y = pad_batch(examples, pad_id=tok.pad_id)
    x, y = x.to(device), y.to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    with ctx:
        old_logp, mask = token_logprobs(policy, x, y)
        ref_logp, _ = token_logprobs(reference, x, y)
        values = value_model(x).float()
    resp_len = mask.sum(1).float().mean().item()
    return x, y, mask, rewards, old_logp, ref_logp, values, resp_len


def compute_gae(
    rewards, values, mask, old_logp, ref_logp
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token reward shaping + GAE; returns normalized advantages and returns.

    Reward per response token is the KL penalty; the scalar reward lands on each
    sequence's last response token. Advantages come from GAE over the value
    baseline, then are normalized across the batch's response tokens.
    """
    N, T = mask.shape
    m = mask.float()
    r = (-PPO_KL_BETA * (old_logp - ref_logp)) * m  # per-token KL penalty
    last_idx = (T - 1) - m.flip(1).argmax(1)  # last response position per sequence
    r[torch.arange(N, device=r.device), last_idx] += rewards  # terminal scalar reward

    adv = torch.zeros_like(values)
    lastgae = torch.zeros(N, device=values.device)
    for t in reversed(range(T)):
        next_val = values[:, t + 1] if t + 1 < T else torch.zeros(N, device=values.device)
        nonterm = m[:, t + 1] if t + 1 < T else torch.zeros(N, device=values.device)
        delta = r[:, t] + PPO_GAMMA * next_val * nonterm - values[:, t]
        lastgae = delta + PPO_GAMMA * PPO_LAM * nonterm * lastgae
        adv[:, t] = lastgae
    adv = adv * m
    returns = adv + values  # critic target (only response positions are used in the loss)

    sel = mask.bool()
    adv[sel] = (adv[sel] - adv[sel].mean()) / (adv[sel].std() + 1e-8)  # normalize advantages
    return adv, returns


def ppo_losses(
    policy, value_model, x, y, mask, adv, returns, old_logp, old_values, ctx
) -> tuple[torch.Tensor, float, float]:
    """Clipped actor surrogate + clipped value loss for one update pass."""
    with ctx:
        new_logp, _ = token_logprobs(policy, x, y)
        new_values = value_model(x).float()
    m = mask.float()
    n = m.sum().clamp(min=1)

    ratio = torch.exp(new_logp - old_logp)
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv)
    actor_loss = -(surr * m).sum() / n

    v_clipped = old_values + (new_values - old_values).clamp(-PPO_CLIP_EPS, PPO_CLIP_EPS)
    v_loss = torch.max((new_values - returns) ** 2, (v_clipped - returns) ** 2)
    value_loss = 0.5 * (v_loss * m).sum() / n

    loss = actor_loss + PPO_VF_COEF * value_loss
    return loss, actor_loss.item(), value_loss.item()


def train_ppo() -> None:
    """Run the PPO loop: rollout, GAE, clipped actor + critic updates."""
    torch.manual_seed(PPO_SEED)
    device, ctx, scaler = setup_amp()

    init_ckpt = resolve_checkpoint(PPO_INIT_RUN, "best.pt", PPO_INIT_DIR)
    policy, _ = load_checkpoint(init_ckpt, device)
    reference, _ = load_checkpoint(init_ckpt, device)
    reference.eval()
    reference.requires_grad_(False)
    critic_backbone, _ = load_checkpoint(init_ckpt, device)
    value_model = ValueModel(critic_backbone).to(device)
    cfg = policy.cfg

    opt_policy = configure_optimizers(policy, PPO_WEIGHT_DECAY, PPO_LR, (BETA1, BETA2), device)
    opt_value = configure_optimizers(value_model, PPO_WEIGHT_DECAY, PPO_VALUE_LR, (BETA1, BETA2), device)

    tok = Tokenizer()
    max_prompt = cfg.block_size - PPO_MAX_NEW_TOKENS
    pool = load_prompt_pool(tok, max_prompt, PPO_PROMPT_POOL)
    print(
        f"PPO init from {init_ckpt} ({policy.num_params():,} non-embedding params) | "
        f"prompts/step={PPO_PROMPTS_PER_STEP} inner={PPO_INNER_EPOCHS} "
        f"clip={PPO_CLIP_EPS} beta={PPO_KL_BETA} on {device}"
    )

    run_dir = new_run_dir(PPO_CKPT_DIR)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    print(f"run dir: {run_dir} | {PPO_STEPS:,} steps")

    ema_reward = None
    best_reward = -float("inf")
    start = time.time()
    policy.train()
    value_model.train()

    for step in range(PPO_STEPS):
        lr_p = cosine_lr(step, PPO_WARMUP_STEPS, PPO_STEPS, PPO_LR, PPO_LR)  # warmup then constant
        lr_v = cosine_lr(step, PPO_WARMUP_STEPS, PPO_STEPS, PPO_VALUE_LR, PPO_VALUE_LR)
        for grp in opt_policy.param_groups:
            grp["lr"] = lr_p
        for grp in opt_value.param_groups:
            grp["lr"] = lr_v

        x, y, mask, rewards, old_logp, ref_logp, old_values, resp_len = rollout(
            policy, value_model, reference, tok, cfg, pool, ctx, device
        )
        adv, returns = compute_gae(rewards, old_values, mask, old_logp, ref_logp)

        a_loss = v_loss = 0.0
        for _ in range(PPO_INNER_EPOCHS):
            loss, a_loss, v_loss = ppo_losses(
                policy, value_model, x, y, mask, adv, returns, old_logp, old_values, ctx
            )
            scaler.scale(loss).backward()
            optimizer_step(
                scaler,
                [(opt_policy, policy.parameters()), (opt_value, value_model.parameters())],
                PPO_GRAD_CLIP,
            )

        mean_reward = rewards.mean().item()
        ema_reward = mean_reward if ema_reward is None else 0.9 * ema_reward + 0.1 * mean_reward

        if step % PPO_LOG_INTERVAL == 0:
            left = PPO_STEPS - step
            elapsed = (time.time() - start) / 60
            eta = left / ((step + 1) / max(1e-9, elapsed)) if step else 0.0
            print(
                f"step {step:>4}/{PPO_STEPS} ({left:,} left): reward {mean_reward:.3f} "
                f"(ema {ema_reward:.3f}) | actor {a_loss:+.4f} | value {v_loss:.4f} | "
                f"resp {resp_len:.0f}t | lr {lr_p:.1e}/{lr_v:.1e} | "
                f"{elapsed:.1f} min, ~{eta:.1f} left"
            )
            log_metrics(run_dir, {
                "step": step,
                "reward": round(mean_reward, 4),
                "ema_reward": round(ema_reward, 4),
                "actor_loss": round(a_loss, 4),
                "value_loss": round(v_loss, 4),
                "resp_len": round(resp_len, 1),
                "lr": lr_p,
            })

        if ema_reward > best_reward:
            best_reward = ema_reward
            save_checkpoint(best_path, policy, opt_policy, step, best_reward, cfg)
        if step % PPO_CKPT_INTERVAL == 0 or step == PPO_STEPS - 1:
            save_checkpoint(last_path, policy, opt_policy, step, best_reward, cfg)

    png = plot_series(run_dir, "step", [
        ("reward", ["reward", "ema_reward"]),
        ("value loss", ["value_loss"]),
        ("actor loss", ["actor_loss"]),
    ])
    print(
        f"done. best ema reward {best_reward:.3f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train_ppo()
