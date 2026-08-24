from pathlib import Path

import torch

from src.config import (
    CKPT_DIR,
    CKPT_RUN,
    DEVICE,
    SAMPLE_PROMPTS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

from src.models.checkpoints import load_checkpoint, resolve_checkpoint
from src.models.tokenizer import Tokenizer


def sample_story(
    model,
    tok: Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int | None = TOP_K,
    top_p: float | None = TOP_P,
) -> str:
    """Generate one continuation for prompt and trim at the first EOS.

    Args:
        model: A model in eval mode.
        tok: The tokenizer.
        prompt: Seed text. Empty prompt starts from the <|endoftext|> id.
        device: Target device.
        max_new_tokens: Tokens to generate.
        temperature: Sampling temperature (<= 0 is greedy).
        top_k: Top-k cutoff, or None.
        top_p: Nucleus cutoff, or None.

    Returns:
        The prompt plus generated text, up to (excluding) the first EOS.
    """
    ids = tok.encode(prompt) or [tok.eos_id]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    out = model.generate(
        x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p
    )[0].tolist()

    gen = out[len(ids) :]  # newly generated ids
    if tok.eos_id in gen:
        gen = gen[: gen.index(tok.eos_id)]  # stop at the first story boundary
    return tok.decode(ids + gen)


def generate() -> None:
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(
        resolve_checkpoint(CKPT_RUN, "best.pt", CKPT_DIR), device
    )
    model.eval()
    tok = Tokenizer()
    print(
        f"loaded checkpoint (trained {ckpt['step']} steps), "
        f"sampling T={TEMPERATURE} top_k={TOP_K} top_p={TOP_P}\n"
    )

    for prompt in SAMPLE_PROMPTS:
        story = sample_story(model, tok, prompt, device)
        print(f"--- prompt: {prompt!r} ---\n{story}\n")


if __name__ == "__main__":
    generate()
