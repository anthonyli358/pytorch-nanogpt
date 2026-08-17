"""Post-training evaluation: DPO vs SFT win-rate + KL from the reference (step 13).

Grades the DPO policy against its SFT baseline on **held-out** instruct-valid
prompts (never the train prompts the preference pairs were built from). For each
prompt with a verifiable ``Words:`` field, sample K completions from each model,
score with ``verifiable_reward``, and take the mean as that model's per-prompt
score. Reports:

- **pass-rate** -- mean verifiable reward per model (did the stories include the
  required words?);
- **win-rate** -- fraction of prompts where the DPO mean reward beats SFT
  (with losses / ties), the head-to-head DPO improved over SFT;
- **KL(DPO || SFT)** on the DPO model's own samples -- the over-optimization
  gauge. It should climb only as far as the win-rate justifies; a large KL with
  flat win-rate is reward hacking, exactly the failure the README warns about.

The regression check (perplexity / grammar didn't collapse) stays in the
existing perplexity path (``src/evaluate.py``); this module covers the first two.
"""

import json
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch

from src.config import (
    DATA_DIR,
    SFT_CKPT_DIR,
    DPO_CKPT_DIR,
    DPO_MAX_LEN,
    PREF_MAX_NEW_TOKENS,
    PREF_MIN_NEW_TOKENS,
    WINRATE_SFT_RUN,
    WINRATE_DPO_RUN,
    WINRATE_NUM_PROMPTS,
    WINRATE_SAMPLES_PER_PROMPT,
    WINRATE_LOG_EVERY,
    WINRATE_SEED,
    WINRATE_RESULTS_FILE,
)
from src.data.sft_data import download_instruct, parse_records, build_example, pad_batch
from src.dpo_train import sequence_logprob
from src.make_preferences import sample_completions
from src.models.checkpoints import load_checkpoint, resolve_checkpoint, latest_run_dir
from src.models.tokenizer import Tokenizer
from src.reward import parse_instruction, verifiable_reward
from src.train import resolve_device_dtype


def _mean_reward(prompt: str, stories: list[str]) -> float | None:
    """Mean verifiable reward over a model's completions for one prompt."""
    scored = [verifiable_reward(prompt, s) for s in stories if s]
    scored = [r for r in scored if r is not None]
    return sum(scored) / len(scored) if scored else None


@torch.no_grad()
def _sequence_kl(policy, reference, stories, prompt, tok, max_len, ctx, device):
    """Estimate KL(policy || reference) on ``stories`` (samples from the policy).

    Returns ``(sum_kl_seq, sum_kl_per_tok, n)`` over the completions that fit the
    context, using the summed response-token log-prob gap ``logp_pi - logp_ref``.
    """
    examples = []
    for s in stories:
        ex = build_example(tok, prompt, s, max_len)
        if ex is not None:
            examples.append((torch.tensor(ex[0]), torch.tensor(ex[1])))
    if not examples:
        return 0.0, 0.0, 0
    x, y = pad_batch(examples, pad_id=tok.pad_id)
    x, y = x.to(device), y.to(device)
    with ctx:
        lp_pi = sequence_logprob(policy, x, y)
        lp_ref = sequence_logprob(reference, x, y)
    kl_seq = lp_pi - lp_ref  # (B,)
    n_resp = (y != -1).sum(dim=1).clamp(min=1)  # response tokens per sequence
    return kl_seq.sum().item(), (kl_seq / n_resp).sum().item(), len(examples)


def evaluate_winrate() -> None:
    """Sample SFT and DPO on held-out prompts; report pass-rate, win-rate, KL."""
    torch.manual_seed(WINRATE_SEED)
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )

    sft_ckpt = resolve_checkpoint(WINRATE_SFT_RUN, "best.pt", SFT_CKPT_DIR)
    dpo_ckpt = resolve_checkpoint(WINRATE_DPO_RUN, "best.pt", DPO_CKPT_DIR)
    sft, _ = load_checkpoint(sft_ckpt, device)
    dpo, _ = load_checkpoint(dpo_ckpt, device)
    sft.eval()
    dpo.eval()
    cfg = dpo.cfg
    tok = Tokenizer()
    max_len = min(DPO_MAX_LEN or cfg.block_size, cfg.block_size)
    max_prompt = cfg.block_size - PREF_MIN_NEW_TOKENS
    print(
        f"SFT  {sft_ckpt}\nDPO  {dpo_ckpt}\n"
        f"grading K={WINRATE_SAMPLES_PER_PROMPT} on held-out valid prompts, {device}"
    )

    paths = download_instruct(DATA_DIR)
    seen = wins = losses = ties = skipped = 0
    sft_pass = dpo_pass = 0.0
    kl_seq_sum = kl_tok_sum = 0.0
    kl_n = 0
    start = time.time()

    for prompt, _ in parse_records(paths["valid"]):
        if "words" not in parse_instruction(prompt):
            continue
        prompt_ids = tok.encode(prompt)
        if not prompt_ids or len(prompt_ids) > max_prompt:
            skipped += 1
            continue
        n_new = min(PREF_MAX_NEW_TOKENS, cfg.block_size - len(prompt_ids))
        k = WINRATE_SAMPLES_PER_PROMPT

        sft_stories = sample_completions(sft, tok, prompt_ids, k, n_new, ctx, device)
        dpo_stories = sample_completions(dpo, tok, prompt_ids, k, n_new, ctx, device)
        sft_r = _mean_reward(prompt, sft_stories)
        dpo_r = _mean_reward(prompt, dpo_stories)
        if sft_r is None or dpo_r is None:
            skipped += 1
            continue
        seen += 1
        sft_pass += sft_r
        dpo_pass += dpo_r
        if dpo_r > sft_r:
            wins += 1
        elif dpo_r < sft_r:
            losses += 1
        else:
            ties += 1

        s, t, n = _sequence_kl(dpo, sft, dpo_stories, prompt, tok, max_len, ctx, device)
        kl_seq_sum += s
        kl_tok_sum += t
        kl_n += n

        if seen % WINRATE_LOG_EVERY == 0:
            left = WINRATE_NUM_PROMPTS - seen
            elapsed = time.time() - start
            eta = left / (seen / elapsed) / 60 if seen else 0.0
            print(
                f"  {seen:,}/{WINRATE_NUM_PROMPTS:,} prompts ({left:,} left) | "
                f"win-rate {wins / seen:.3f} | {elapsed / 60:.1f} min elapsed, ~{eta:.1f} min left"
            )
        if seen >= WINRATE_NUM_PROMPTS:
            break

    seen = max(1, seen)
    results = {
        "sft_ckpt": str(sft_ckpt),
        "dpo_ckpt": str(dpo_ckpt),
        "prompts": seen,
        "samples_per_prompt": WINRATE_SAMPLES_PER_PROMPT,
        "sft_pass_rate": round(sft_pass / seen, 4),
        "dpo_pass_rate": round(dpo_pass / seen, 4),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / seen, 4),          # DPO strictly beats SFT
        "win_rate_incl_ties": round((wins + ties) / seen, 4),
        "kl_dpo_ref_per_seq": round(kl_seq_sum / max(1, kl_n), 4),
        "kl_dpo_ref_per_tok": round(kl_tok_sum / max(1, kl_n), 4),
        "prompts_skipped": skipped,
    }

    print(
        f"\n=== win-rate: DPO vs SFT ({seen:,} held-out prompts, "
        f"K={WINRATE_SAMPLES_PER_PROMPT}) ===\n"
        f"pass-rate   SFT {results['sft_pass_rate']:.3f}   DPO {results['dpo_pass_rate']:.3f}\n"
        f"win-rate    {results['win_rate']:.3f}  ({wins} win / {losses} loss / {ties} tie)\n"
        f"KL(DPO||SFT) {results['kl_dpo_ref_per_tok']:.3f} nats/tok "
        f"({results['kl_dpo_ref_per_seq']:.2f} nats/seq)"
    )

    dpo_run = Path(dpo_ckpt).parent if Path(dpo_ckpt).suffix == ".pt" else Path(dpo_ckpt)
    if not dpo_run.is_dir():
        dpo_run = latest_run_dir(DPO_CKPT_DIR) or Path(".")
    out_path = dpo_run / WINRATE_RESULTS_FILE
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    evaluate_winrate()
