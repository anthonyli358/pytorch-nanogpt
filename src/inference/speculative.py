import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from src.config import (
    SPEC_TARGET_DIR,
    SPEC_TARGET_RUN,
    SPEC_DRAFT_DIR,
    SPEC_DRAFT_RUN,
    SPEC_GAMMA,
    SPEC_MAX_NEW_TOKENS,
    SAMPLE_PROMPTS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from src.models.checkpoints import load_checkpoint, resolve_checkpoint
from src.models.tokenizer import Tokenizer
from src.training.common import resolve_device_dtype


def token_dist(
    logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """Next-token probability vector after temperature / top-k / top-p filtering.

    Returns a normalized `(vocab,)` distribution, so draft and target distributions 
    are comparable and the acceptance ratio `p(x)/q(x)` is well defined. 
    Temperature <= 0 gives a one-hot (greedy) distribution.
    """
    if temperature <= 0.0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    logits = logits / temperature
    if top_k is not None:
        thresh = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()  # keep the first token past the threshold
        remove[0] = False
        logits = logits.clone()
        logits[sorted_idx[remove]] = float("-inf")
    return F.softmax(logits, dim=-1)


def _target_logits(target, seq: torch.Tensor) -> torch.Tensor:
    """Full ``(T, vocab)`` logits for a single sequence in one forward pass."""
    return target.lm_head(target.hidden(seq))[0]  # hidden() is the shared backbone seam


@torch.no_grad()
def speculative_generate(
    target,
    draft,
    prompt_ids: list[int],
    max_new_tokens: int,
    gamma: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: str,
    eos_id: int | None = None,
):
    """Generate from `target` using `draft` proposals; return `(ids, stats)`.

    `stats` reports `target_calls` (target forward passes), `new_tokens`, and
    `mean_accept_len` = tokens per target call. The speedup factor when the
    draft is much cheaper than the target.
    """
    cur = list(prompt_ids)
    new_tokens = target_calls = accepted = 0
    done = False
    # Sliding window: leave room for gamma drafted tokens so every forward fits the context.
    block = min(target.cfg.block_size, draft.cfg.block_size)
    keep = block - gamma

    while new_tokens < max_new_tokens and not done:
        window = cur[-keep:] if len(cur) > keep else list(cur)
        base_len = len(window)

        # 1) Draft proposes gamma tokens autoregressively, recording its distributions q.
        drafted: list[int] = []
        q_dists: list[torch.Tensor] = []
        dctx = torch.tensor(window, dtype=torch.long, device=device)[None, :]
        for _ in range(gamma):
            dlogits, _ = draft(dctx)  # inference path -> last-position logits
            q = token_dist(dlogits[0, -1], temperature, top_k, top_p)
            tok = int(torch.multinomial(q, 1))
            drafted.append(tok)
            q_dists.append(q)
            dctx = torch.cat([dctx, torch.tensor([[tok]], device=device)], dim=1)

        # 2) Target verifies all gamma proposals in ONE forward pass over the window.
        seq = torch.tensor(window + drafted, dtype=torch.long, device=device)[None, :]
        tlogits = _target_logits(target, seq)  # (base_len + gamma, vocab)
        target_calls += 1
        base = base_len - 1  # logits[base + i] predicts drafted[i]

        # 3) Accept the longest consistent prefix; correct-and-stop on the first reject.
        rejected = False
        for i in range(gamma):
            p = token_dist(tlogits[base + i], temperature, top_k, top_p)
            x = drafted[i]
            ratio = (p[x] / q_dists[i][x]).item() if q_dists[i][x] > 0 else 0.0
            if torch.rand(1, device=device).item() < min(1.0, ratio):
                cur.append(x)
                new_tokens += 1
                accepted += 1
                if eos_id is not None and x == eos_id:
                    done = True
                    break
                if new_tokens >= max_new_tokens:
                    done = True
                    break
            else:
                corr = torch.clamp(p - q_dists[i], min=0.0)  # residual distribution
                corr = p if corr.sum() <= 0 else corr / corr.sum()
                tok = int(torch.multinomial(corr, 1))
                cur.append(tok)
                new_tokens += 1
                if eos_id is not None and tok == eos_id:
                    done = True
                rejected = True
                break

        # 4) All gamma accepted -> the target's next-position logits give a free bonus token.
        if not rejected and not done and new_tokens < max_new_tokens:
            p = token_dist(tlogits[base + gamma], temperature, top_k, top_p)
            tok = int(torch.multinomial(p, 1))
            cur.append(tok)
            new_tokens += 1
            if eos_id is not None and tok == eos_id:
                done = True

    stats = {
        "new_tokens": new_tokens,
        "target_calls": target_calls,
        "mean_accept_len": new_tokens / max(1, target_calls),
        "accept_rate": accepted / max(1, target_calls * gamma),
    }
    return cur, stats


def compare_decoding(
    prompts: list[str] | None = None,
    gamma: int = SPEC_GAMMA,
    max_new_tokens: int = SPEC_MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int | None = TOP_K,
    top_p: float | None = TOP_P,
) -> list[dict]:
    """
    Load target + draft checkpoints, generate, and report throughput vs vanilla.

    A small draft model proposes `gamma` tokens autoregressively.
    A larger target model verifies all of them in a single forward pass and
    accepts the longest prefix that's consistent with its own distribution. On the
    first mismatch it resamples from a corrected distribution and discards the rest

    Args:
        prompts: Prompts to benchmark, or None to use ``SAMPLE_PROMPTS``.
        gamma: Draft tokens proposed per target verification pass.
        max_new_tokens: Tokens to generate per prompt.
        temperature: Sampling temperature (``<= 0`` is greedy).
        top_k: Top-k filter, or None.
        top_p: Top-p (nucleus) filter, or None.

    Returns:
        Per-prompt stats dicts (empty if no draft checkpoint is available), each
        with the prompt, the ``speculative_generate`` stats, both timings (ms),
        and the measured speedup.
    """
    prompts = SAMPLE_PROMPTS if prompts is None else prompts
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )

    target, _ = load_checkpoint(
        resolve_checkpoint(SPEC_TARGET_RUN, "best.pt", SPEC_TARGET_DIR), device
    )
    try:
        draft_path = resolve_checkpoint(SPEC_DRAFT_RUN, "best.pt", SPEC_DRAFT_DIR)
        draft, _ = load_checkpoint(draft_path, device)
    except FileNotFoundError:
        print(
            f"No draft model under {SPEC_DRAFT_DIR!r}. Train a small one (a reduced "
            f"GPTConfig, e.g. n_layer=2, d_model=128) on the same corpus, then point "
            f"SPEC_DRAFT_DIR at it. The algorithm and its correctness tests run without "
            f"one -- see `python -m tests.speculative_test`."
        )
        return []
    target.eval()
    draft.eval()
    tok = Tokenizer()
    print(
        f"target {target.num_params():,} params | draft {draft.num_params():,} params | "
        f"gamma={gamma} on {device}\n"
    )

    results = []
    for prompt in prompts:
        ids = tok.encode(prompt) or [tok.eos_id]

        with ctx:
            t0 = time.time()
            out_v = target.generate(
                torch.tensor(ids, device=device)[None, :],
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )[0].tolist()
            t_vanilla = time.time() - t0

            t0 = time.time()
            out_s, stats = speculative_generate(
                target,
                draft,
                ids,
                max_new_tokens,
                gamma,
                temperature,
                top_k,
                top_p,
                device,
                eos_id=tok.eos_id,
            )
            t_spec = time.time() - t0

        speedup = t_vanilla / max(1e-9, t_spec)
        print(f"--- prompt: {prompt!r} ---")
        print(f"speculative: {tok.decode(out_s)}")
        print(
            f"  vanilla {t_vanilla * 1000:.0f} ms ({len(out_v) - len(ids)} tok) | "
            f"speculative {t_spec * 1000:.0f} ms ({stats['new_tokens']} tok) | "
            f"{stats['mean_accept_len']:.2f} tok/target-call | "
            f"accept-rate {stats['accept_rate']:.2f} | speedup {speedup:.2f}x\n"
        )
        results.append(
            {
                "prompt": prompt,
                **stats,
                "vanilla_ms": t_vanilla * 1000,
                "spec_ms": t_spec * 1000,
                "speedup": speedup,
            }
        )
    return results


if __name__ == "__main__":
    compare_decoding()
