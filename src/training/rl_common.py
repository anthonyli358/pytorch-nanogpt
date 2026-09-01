"""
Shared machinery for the online RL trainers (GRPO, PPO).
"""

import torch
import torch.nn.functional as F

from src.config import DATA_DIR
from src.data.sft_data import download_instruct, parse_records
from src.models.tokenizer import Tokenizer
from src.reward import parse_instruction


def token_logprobs(model, x: torch.Tensor, y: torch.Tensor):
    """
    Per-token log-probs of the targets and the response mask.

    Passing `y` (labels with `-1` over prompt + padding) as targets makes
    `forward` return the full `(B, T, vocab)` logits.

    Returns:
        `(logp, mask)` of each `(B, T)`. The log-prob of the label token at every
        position, and the boolean response mask (`y != -1`).
    """
    logits, _ = model(x, y)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = y != -1
    tok = torch.gather(logp, -1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return tok, mask


def sequence_logprob(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Summed response-token log-prob per sequence, shape `(B,)`.

    The masked sum of :func:`token_logprobs`. This is the quantity the DPO objective
    contrasts between chosen and rejected against the frozen reference.
    """
    tok, mask = token_logprobs(model, x, y)
    return (tok * mask).sum(dim=-1)


def load_prompt_pool(tok: Tokenizer, max_prompt: int, limit: int) -> list[tuple[str, list[int]]]:
    """
    Pre-tokenize instruct-train prompts with a `Words:` field into an in-memory pool.

    Keeps the original prompt *text* alongside the `ids:`. The reward is scored on the
    text (SentencePiece decode collapses the newlines the `Words:` regex needs).
    """
    paths = download_instruct(DATA_DIR)
    pool = []
    for prompt, _ in parse_records(paths["train"]):
        if "words" not in parse_instruction(prompt):
            continue
        ids = tok.encode(prompt)
        if ids and len(ids) <= max_prompt:
            pool.append((prompt, ids))
        if len(pool) >= limit:
            break
    print(f"prompt pool: {len(pool):,} instruct prompts with a Words: field")
    return pool


def build_rollout_example(prompt_ids, gen_ids, eos_id, tok):
    """
    Turn a generated sequence into `(input_ids, labels, story_text)`.

    The response is the generated tokens up to (and including) the first `EOS`
    tokens after EOS are dropped. `labels` mask the prompt to `-1` (as in SFT),
    so log-probs cover only the response the reward is computed on.
    """
    end = gen_ids.index(eos_id) if eos_id in gen_ids else len(gen_ids)
    resp = gen_ids[:end] + ([eos_id] if eos_id in gen_ids else [])
    seq = prompt_ids + resp
    input_ids, labels = seq[:-1], seq[1:]
    for j in range(len(prompt_ids) - 1):  # mask targets that predict a prompt token
        labels[j] = -1
    return input_ids, labels, tok.decode(gen_ids[:end])
