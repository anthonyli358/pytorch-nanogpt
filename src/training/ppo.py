"""
PPO: the full classic actor-critic stack..


The critic is a second GPT backbone with a scalar value head `ValueModel`.
PPO optimizes the verifiable reward directly.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.config import (
    PPO_CKPT_DIR,
    SFT_CKPT_DIR,
    REP_NGRAM,
    REP_WEIGHT,
    SEED,
)
from src.data.common import pad_batch
from src.eval.metrics import log_metrics, plot_series
from src.training.rl_common import (
    token_logprobs,
    load_prompt_pool,
    build_rollout_example,
)
from src.models.checkpoints import (
    load_checkpoint,
    save_checkpoint,
    new_run_dir,
    resolve_checkpoint,
)
from src.models.gpt import GPT
from src.models.tokenizer import Tokenizer
from src.eval.reward import shaped_reward
from src.training.common import (
    setup_amp,
    cosine_lr,
    optimizer_step,
    configure_optimizers,
)


@dataclass
class PPOConfig:
    """
    PPO trainer hyperparameters (actor-critic; init from an SFT run).
    """

    init_dir: str = SFT_CKPT_DIR  # policy + critic backbone + frozen reference init
    init_run: str | None = None  # None = latest run under init_dir
    prompt_pool: int = 20_000  # instruct-train prompts to sample rollouts from
    prompts_per_step: int = 32  # completions per rollout (one per prompt)
    steps: int = 400  # optimizer steps (rollouts)
    inner_epochs: int = 4  # PPO reuses each rollout for several clipped update passes
    lr: float = 1e-6  # actor LR
    value_lr: float = 1e-5  # critic can learn faster than the actor
    warmup_steps: int = 20
    weight_decay: float = 0.0
    beta1: float = 0.9  # Adam betas
    beta2: float = 0.95
    grad_clip: float = 1.0
    clip_eps: float = 0.2  # PPO ratio + value clip range
    vf_coef: float = 0.5  # weight of the value loss in the total objective
    kl_beta: float = 0.02  # per-token KL(policy||ref) penalty folded into the reward
    gamma: float = 1.0  # discount (1.0: short episodes, no far-future discounting)
    lam: float = 0.95  # GAE lambda (bias/variance trade-off)
    temperature: float = 1.0
    top_k: int = 200
    top_p: float = 0.95
    max_new_tokens: int = 200
    rep_weight: float = REP_WEIGHT  # repetition-penalty weight in the terminal reward
    log_interval: int = 5
    ckpt_interval: int = 50
    seed: int = SEED


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
    policy, value_model, reference, tok, gpt_cfg, pool, ctx, device, cfg: "PPOConfig"
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
    """
    Sample one completion per prompt
    
    Return: The padded batch + rollout tensors.
    """
    idx = torch.randint(len(pool), (cfg.prompts_per_step,)).tolist()
    examples, rewards = [], []
    for pi in idx:
        prompt_text, prompt_ids = pool[pi]
        n_new = min(cfg.max_new_tokens, gpt_cfg.block_size - len(prompt_ids))
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
        with ctx:
            out = policy.generate(
                x, n_new, temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p
            )
        gen = out[0].tolist()[len(prompt_ids) :]
        inp, lab, story = build_rollout_example(prompt_ids, gen, tok.eos_id, tok)
        r = shaped_reward(prompt_text, story, rep_weight=cfg.rep_weight, n=REP_NGRAM)
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
    rewards, values, mask, old_logp, ref_logp, cfg: "PPOConfig"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token reward shaping + GAE (Generalized Advantage Estimation).

    Returns: Normalized advantages and returns.
    """
    N, T = mask.shape
    m = mask.float()
    r = (-cfg.kl_beta * (old_logp - ref_logp)) * m  # per-token KL penalty
    last_idx = (T - 1) - m.flip(1).argmax(1)  # last response position per sequence
    r[torch.arange(N, device=r.device), last_idx] += rewards  # terminal scalar reward

    adv = torch.zeros_like(values)
    lastgae = torch.zeros(N, device=values.device)
    for t in reversed(range(T)):
        next_val = (
            values[:, t + 1] if t + 1 < T else torch.zeros(N, device=values.device)
        )
        nonterm = m[:, t + 1] if t + 1 < T else torch.zeros(N, device=values.device)
        delta = r[:, t] + cfg.gamma * next_val * nonterm - values[:, t]
        lastgae = delta + cfg.gamma * cfg.lam * nonterm * lastgae
        adv[:, t] = lastgae
    adv = adv * m
    returns = (
        adv + values
    )  # critic target (only response positions are used in the loss)

    sel = mask.bool()
    adv[sel] = (adv[sel] - adv[sel].mean()) / (
        adv[sel].std() + 1e-8
    )  # normalize advantages
    return adv, returns


def ppo_losses(
    policy,
    value_model,
    x,
    y,
    mask,
    adv,
    returns,
    old_logp,
    old_values,
    ctx,
    cfg: "PPOConfig",
) -> tuple[torch.Tensor, float, float]:
    """Clipped actor surrogate + clipped value loss for one update pass."""
    with ctx:
        new_logp, _ = token_logprobs(policy, x, y)
        new_values = value_model(x).float()
    m = mask.float()
    n = m.sum().clamp(min=1)

    ratio = torch.exp(new_logp - old_logp)
    surr = torch.min(
        ratio * adv, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    )
    actor_loss = -(surr * m).sum() / n

    v_clipped = old_values + (new_values - old_values).clamp(
        -cfg.clip_eps, cfg.clip_eps
    )
    v_loss = torch.max((new_values - returns) ** 2, (v_clipped - returns) ** 2)
    value_loss = 0.5 * (v_loss * m).sum() / n

    loss = actor_loss + cfg.vf_coef * value_loss
    return loss, actor_loss.item(), value_loss.item()


def train_ppo(cfg: PPOConfig = PPOConfig()) -> None:
    """Run the PPO loop: rollout, GAE, clipped actor + critic updates."""
    torch.manual_seed(cfg.seed)
    device, ctx, scaler = setup_amp()

    init_ckpt = resolve_checkpoint(cfg.init_run, "best.pt", cfg.init_dir)
    policy, _ = load_checkpoint(init_ckpt, device)
    reference, _ = load_checkpoint(init_ckpt, device)
    reference.eval()
    reference.requires_grad_(False)
    critic_backbone, _ = load_checkpoint(init_ckpt, device)
    value_model = ValueModel(critic_backbone).to(device)
    gpt_cfg = policy.cfg

    opt_policy = configure_optimizers(
        policy, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
    )
    opt_value = configure_optimizers(
        value_model, cfg.weight_decay, cfg.value_lr, (cfg.beta1, cfg.beta2), device
    )

    tok = Tokenizer()
    max_prompt = gpt_cfg.block_size - cfg.max_new_tokens
    pool = load_prompt_pool(tok, max_prompt, cfg.prompt_pool)
    print(
        f"PPO init from {init_ckpt} ({policy.num_params():,} non-embedding params) | "
        f"prompts/step={cfg.prompts_per_step} inner={cfg.inner_epochs} "
        f"clip={cfg.clip_eps} beta={cfg.kl_beta} on {device}"
    )

    run_dir = new_run_dir(PPO_CKPT_DIR)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    print(f"run dir: {run_dir} | {cfg.steps:,} steps")

    ema_reward = None
    best_reward = -float("inf")
    start = time.time()
    policy.train()
    value_model.train()

    for step in range(cfg.steps):
        lr_p = cosine_lr(
            step, cfg.warmup_steps, cfg.steps, cfg.lr, cfg.lr
        )  # warmup then constant
        lr_v = cosine_lr(step, cfg.warmup_steps, cfg.steps, cfg.value_lr, cfg.value_lr)
        for grp in opt_policy.param_groups:
            grp["lr"] = lr_p
        for grp in opt_value.param_groups:
            grp["lr"] = lr_v

        x, y, mask, rewards, old_logp, ref_logp, old_values, resp_len = rollout(
            policy, value_model, reference, tok, gpt_cfg, pool, ctx, device, cfg
        )
        adv, returns = compute_gae(rewards, old_values, mask, old_logp, ref_logp, cfg)

        a_loss = v_loss = 0.0
        for _ in range(cfg.inner_epochs):
            loss, a_loss, v_loss = ppo_losses(
                policy,
                value_model,
                x,
                y,
                mask,
                adv,
                returns,
                old_logp,
                old_values,
                ctx,
                cfg,
            )
            scaler.scale(loss).backward()
            optimizer_step(
                scaler,
                [
                    (opt_policy, policy.parameters()),
                    (opt_value, value_model.parameters()),
                ],
                cfg.grad_clip,
            )

        mean_reward = rewards.mean().item()
        ema_reward = (
            mean_reward if ema_reward is None else 0.9 * ema_reward + 0.1 * mean_reward
        )

        if step % cfg.log_interval == 0:
            left = cfg.steps - step
            elapsed = (time.time() - start) / 60
            eta = left / ((step + 1) / max(1e-9, elapsed)) if step else 0.0
            print(
                f"step {step:>4}/{cfg.steps} ({left:,} left): reward {mean_reward:.3f} "
                f"(ema {ema_reward:.3f}) | actor {a_loss:+.4f} | value {v_loss:.4f} | "
                f"resp {resp_len:.0f}t | lr {lr_p:.1e}/{lr_v:.1e} | "
                f"{elapsed:.1f} min, ~{eta:.1f} left"
            )
            log_metrics(
                run_dir,
                {
                    "step": step,
                    "reward": round(mean_reward, 4),
                    "ema_reward": round(ema_reward, 4),
                    "actor_loss": round(a_loss, 4),
                    "value_loss": round(v_loss, 4),
                    "resp_len": round(resp_len, 1),
                    "lr": lr_p,
                },
            )

        if ema_reward > best_reward:
            best_reward = ema_reward
            save_checkpoint(best_path, policy, opt_policy, step, best_reward, gpt_cfg)
        if step % cfg.ckpt_interval == 0 or step == cfg.steps - 1:
            save_checkpoint(last_path, policy, opt_policy, step, best_reward, gpt_cfg)

    png = plot_series(
        run_dir,
        "step",
        [
            ("reward", ["reward", "ema_reward"]),
            ("value loss", ["value_loss"]),
            ("actor loss", ["actor_loss"]),
        ],
    )
    print(
        f"done. best ema reward {best_reward:.3f}. checkpoints in {run_dir}/"
        + (f" (curve: {png})" if png else "")
    )


if __name__ == "__main__":
    train_ppo()
