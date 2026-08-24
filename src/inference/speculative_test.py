"""Correctness tests for speculative decoding (no GPU, no checkpoints needed).

Two properties, on tiny random models:

1. **Unbiased.** For *any* draft ``q`` and target ``p``, one speculative step is
   distributed exactly as ``p``. Verified by sampling many single steps with a
   *different* random draft and comparing the empirical token distribution to the
   target's -- the whole point of the accept/reject correction.
2. **All-accept.** When the draft *is* the target, ``q == p`` so every proposal is
   accepted: zero rejections and ``gamma + 1`` tokens per target forward.
"""

import torch

from src.config import GPTConfig
from src.models.gpt import GPT
from src.inference.speculative import speculative_generate, token_dist


def _tiny(seed: int) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=16, block_size=32, n_layer=2, n_head=2, d_model=32)).eval()


def test_unbiased(n_samples: int = 20_000, tol: float = 0.03) -> None:
    """Speculative output distribution must match the target's, for a mismatched draft."""
    torch.manual_seed(0)
    target, draft = _tiny(1), _tiny(2)  # deliberately different models
    ctx = [3, 7, 1, 9]

    with torch.no_grad():
        tlogits, _ = target(torch.tensor(ctx)[None, :])
        p = token_dist(tlogits[0, -1], temperature=1.0, top_k=None, top_p=None)

    counts = torch.zeros(16)
    for _ in range(n_samples):
        out, _ = speculative_generate(
            target, draft, ctx, max_new_tokens=1, gamma=1,
            temperature=1.0, top_k=None, top_p=None, device="cpu",
        )
        counts[out[len(ctx)]] += 1
    emp = counts / counts.sum()
    tv = 0.5 * (emp - p).abs().sum().item()  # total-variation distance
    print(f"[unbiased] TV(empirical, target) = {tv:.4f}  (tol {tol})  -> {'PASS' if tv < tol else 'FAIL'}")
    assert tv < tol, f"speculative distribution deviates from target (TV={tv:.4f})"


def test_all_accept(gamma: int = 4, max_new: int = 40) -> None:
    """With draft == target, nothing is ever rejected: gamma+1 tokens per target call."""
    model = _tiny(1)
    _, stats = speculative_generate(
        model, model, [3, 7, 1, 9], max_new_tokens=max_new, gamma=gamma,
        temperature=1.0, top_k=None, top_p=None, device="cpu", eos_id=None,
    )
    print(f"[all-accept] accept_rate={stats['accept_rate']:.3f}  "
          f"mean_accept_len={stats['mean_accept_len']:.2f} (expect {gamma + 1})  "
          f"-> {'PASS' if stats['accept_rate'] == 1.0 else 'FAIL'}")
    assert stats["accept_rate"] == 1.0, "draft==target should never reject"
    assert abs(stats["mean_accept_len"] - (gamma + 1)) < 1e-6


def run_tests() -> None:
    test_all_accept()
    test_unbiased()
    print("all speculative-decoding tests passed")


if __name__ == "__main__":
    run_tests()
