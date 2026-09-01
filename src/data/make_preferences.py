"""
Generate DPO preference pairs from the SFT model.

Loads the finished SFT checkpoint and samples `K` completions at temperature
> 0 for diversity. Each completion is scored by the programmatic reward
"""

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import (
    DATA_DIR,
    SFT_CKPT_DIR,
    DPO_DATA_DIR,
    PAIRS_FILE,
    REP_WEIGHT,
    REP_NGRAM,
    SEED,
)
from src.data.sft_data import download_instruct, parse_records
from src.models.checkpoints import load_checkpoint, resolve_checkpoint
from src.models.tokenizer import Tokenizer
from src.eval.reward import parse_instruction, verifiable_reward, repetition_penalty
from src.training.common import resolve_device_dtype


@dataclass
class PrefConfig:
    """Preference-generation parameters."""

    init_run: str | None = None  # SFT run to sample from (None = latest)
    num_prompts: int = 5000  # instruct-train prompts (with a Words: field)
    samples_per_prompt: int = 4  # K completions per prompt
    temperature: float = 1.0  # > 0 for diversity across the K samples
    top_k: int = 200
    top_p: float = 0.95
    max_new_tokens: int = 256  # upper cap
    min_new_tokens: int = 48  # skip when the remaining context is smaller
    log_every: int = 200  # progress print every N prompts processed


@torch.no_grad()
def sample_completions(
    model,
    tok: Tokenizer,
    prompt_ids: list[int],
    k: int,
    max_new_tokens: int,
    ctx,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = 200,
    top_p: float | None = 0.95,
) -> list[str]:
    """
    Sample `k` completions for one prompt and return their decoded stories.

    Each completion is trimmed at the first EOS (the story boundary).
    The prompt is replicated into a batch of `k` so all samples generate in one
    pass. 

    After reward scoring, when the best and worst scores differ (a reward spread), the pair
    `(prompt, best, worst)` is emitted. Ties carry no learning signal and are skipped.
    """
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).expand(k, -1)
    with ctx:
        out = model.generate(
            x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p
        )
    stories = []
    for row in out.tolist():
        gen = row[len(prompt_ids) :]  # newly generated ids only
        if tok.eos_id in gen:
            gen = gen[: gen.index(tok.eos_id)]  # stop at the first story boundary
        stories.append(tok.decode(gen).strip())
    return stories


def make_preferences(cfg: PrefConfig = PrefConfig()) -> None:
    """Sample, score, and write preference pairs to `data/dpo/pairs.jsonl`."""
    torch.manual_seed(SEED)
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )

    init_ckpt = resolve_checkpoint(cfg.init_run, "best.pt", SFT_CKPT_DIR)
    model, _ = load_checkpoint(init_ckpt, device)
    model.eval()
    gpt_cfg = model.cfg
    tok = Tokenizer()
    # Keep prompt + completion within the model's context (the SFT training regime):
    # generate up to the remaining room, capped at cfg.max_new_tokens, and skip a
    # prompt only when too little room is left for a story (< cfg.min_new_tokens).
    max_prompt = gpt_cfg.block_size - cfg.min_new_tokens
    print(
        f"sampling from {init_ckpt} | K={cfg.samples_per_prompt} T={cfg.temperature} "
        f"top_k={cfg.top_k} top_p={cfg.top_p} on {device}"
    )

    paths = download_instruct(DATA_DIR)
    out_dir = Path(DPO_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / PAIRS_FILE

    seen = pairs = ties = skipped = 0
    start = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for prompt, _ in parse_records(paths["train"]):
            if "words" not in parse_instruction(prompt):
                continue  # no verifiable signal -> nothing to rank on
            prompt_ids = tok.encode(prompt)
            if not prompt_ids or len(prompt_ids) > max_prompt:
                skipped += 1
                continue

            n_new = min(cfg.max_new_tokens, gpt_cfg.block_size - len(prompt_ids))
            stories = sample_completions(
                model,
                tok,
                prompt_ids,
                cfg.samples_per_prompt,
                n_new,
                ctx,
                device,
                cfg.temperature,
                cfg.top_k,
                cfg.top_p,
            )
            # Rank by the SHAPED score (verifiable reward minus a repetition penalty)
            # so that on a word-inclusion tie the less-repetitive completion is
            # chosen -- teaching DPO against the looping the raw reward can't see.
            scored = []
            for s in stories:
                base = verifiable_reward(prompt, s)
                if base is None or not s:
                    continue
                shaped = base - REP_WEIGHT * repetition_penalty(s, REP_NGRAM)
                scored.append((shaped, base, s))
            seen += 1

            if len(scored) >= 2:
                best = max(scored, key=lambda t: t[0])
                worst = min(scored, key=lambda t: t[0])
                if best[0] > worst[0] + 1e-6:  # spread in the shaped score
                    fout.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "chosen": best[2],
                                "rejected": worst[2],
                                "chosen_reward": round(best[1], 4),
                                "rejected_reward": round(worst[1], 4),
                                "chosen_score": round(best[0], 4),
                                "rejected_score": round(worst[0], 4),
                            }
                        )
                        + "\n"
                    )
                    pairs += 1
                else:
                    ties += 1
            else:
                ties += 1

            if seen % cfg.log_every == 0:
                left = cfg.num_prompts - seen
                elapsed = time.time() - start
                eta_min = left / (seen / elapsed) / 60 if seen else 0.0
                print(
                    f"  {seen:,}/{cfg.num_prompts:,} prompts ({left:,} left) | "
                    f"{pairs:,} pairs | {ties:,} ties | {skipped:,} skipped | "
                    f"{elapsed / 60:.1f} min elapsed, ~{eta_min:.1f} min left"
                )
            if seen >= cfg.num_prompts:
                break

    print(
        f"done. {pairs:,} pairs written to {out_path} "
        f"({ties:,} ties skipped, {skipped:,} prompts too long, {seen:,} sampled)"
    )


if __name__ == "__main__":
    make_preferences()
