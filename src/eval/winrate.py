import json
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import torch

from src.config import (
    DATA_DIR,
    DPO_MAX_LEN,
    CONTEXT_LEN,
    EVAL_BATCH_SIZE,
    EVAL_MAX_BATCHES,
    PREF_MAX_NEW_TOKENS,
    PREF_MIN_NEW_TOKENS,
    REP_NGRAM,
    WINRATE_BASELINE,
    WINRATE_MODELS,
    WINRATE_NUM_PROMPTS,
    WINRATE_SAMPLES_PER_PROMPT,
    WINRATE_SAMPLE_DUMP,
    WINRATE_LOG_EVERY,
    WINRATE_SEED,
    WINRATE_RESULTS_FILE,
)
from src.data.sft_data import download_instruct, parse_records
from src.data.common import build_example, pad_batch
from src.eval.perplexity import evaluate_split
from src.training.rl_common import sequence_logprob
from src.make_preferences import sample_completions
from src.models.checkpoints import load_checkpoint, resolve_checkpoint, latest_run_dir
from src.models.tokenizer import Tokenizer
from src.reward import parse_instruction, verifiable_reward, distinct_ngram_ratio
from src.training.common import resolve_device_dtype


def resolve_stage(label: str, ckpt_dir: str, run) -> Path | None:
    """Resolve a stage to its best.pt path, or None if the dir has no runs yet."""
    if run is None and latest_run_dir(ckpt_dir) is None:
        return None
    try:
        return resolve_checkpoint(run, "best.pt", ckpt_dir)
    except FileNotFoundError:
        return None


def held_out_prompts(
    tok: Tokenizer, cfg, max_prompt: int
) -> Iterator[tuple[str, list[int], int]]:
    """Yield (prompt, prompt_ids, n_new) for valid prompts with a Words: field."""
    paths = download_instruct(DATA_DIR)
    for prompt, _ in parse_records(paths["valid"]):
        if "words" not in parse_instruction(prompt):
            continue
        ids = tok.encode(prompt)
        if not ids or len(ids) > max_prompt:
            continue
        yield prompt, ids, min(PREF_MAX_NEW_TOKENS, cfg.block_size - len(ids))


def _mean_reward(prompt: str, stories: list[str]) -> float | None:
    """Mean verifiable reward over a model's completions for one prompt."""
    scored = [verifiable_reward(prompt, s) for s in stories if s]
    scored = [r for r in scored if r is not None]
    return sum(scored) / len(scored) if scored else None


@torch.no_grad()
def _sequence_kl(
    policy, reference, stories, prompt, tok, max_len, ctx, device
) -> tuple[float, float, int]:
    """Estimate KL(policy || reference) on stories (samples from the policy)."""
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
    kl_seq = lp_pi - lp_ref
    n_resp = (y != -1).sum(dim=1).clamp(min=1)
    return kl_seq.sum().item(), (kl_seq / n_resp).sum().item(), len(examples)


def run_winrate(models, base, tok, cfg, ctx, device, n_prompts, max_len, max_prompt) -> dict:
    """Grade every stage in `models` against the `base` baseline in one sweep.

    `models` maps label -> GPT (including the baseline). Returns per-stage
    pass-rate, and per-candidate win-rate / KL against the baseline.
    """
    names = list(models)
    cand = [n for n in names if n != base]
    k = WINRATE_SAMPLES_PER_PROMPT
    seen = skipped = 0
    pass_sum = {n: 0.0 for n in names}
    distinct_sum = {n: 0.0 for n in names}  # mean distinct-n-gram ratio (higher = less looping)
    wins = {c: 0 for c in cand}
    losses = {c: 0 for c in cand}
    ties = {c: 0 for c in cand}
    kl_tok = {c: 0.0 for c in cand}
    kl_n = {c: 0 for c in cand}
    start = time.time()

    for prompt, prompt_ids, n_new in held_out_prompts(tok, cfg, max_prompt):
        stories = {
            n: sample_completions(models[n], tok, prompt_ids, k, n_new, ctx, device)
            for n in names
        }
        rmean = {n: _mean_reward(prompt, stories[n]) for n in names}
        if any(v is None for v in rmean.values()):
            skipped += 1
            continue
        seen += 1
        for n in names:
            pass_sum[n] += rmean[n]
            ds = [distinct_ngram_ratio(s, REP_NGRAM) for s in stories[n] if s]
            distinct_sum[n] += sum(ds) / len(ds) if ds else 1.0
        for c in cand:
            if rmean[c] > rmean[base]:
                wins[c] += 1
            elif rmean[c] < rmean[base]:
                losses[c] += 1
            else:
                ties[c] += 1
            _, t, nn = _sequence_kl(models[c], models[base], stories[c], prompt, tok, max_len, ctx, device)
            kl_tok[c] += t
            kl_n[c] += nn

        if seen % WINRATE_LOG_EVERY == 0:
            left = n_prompts - seen
            elapsed = time.time() - start
            eta = left / (seen / elapsed) / 60 if seen else 0.0
            wr = " ".join(f"{c} {wins[c] / seen:.2f}" for c in cand)
            print(f"  {seen:,}/{n_prompts:,} ({left:,} left) | win-rate {wr} | "
                  f"{elapsed / 60:.1f} min, ~{eta:.1f} left")
        if seen >= n_prompts:
            break

    seen = max(1, seen)
    stages = {
        n: {"pass_rate": round(pass_sum[n] / seen, 4),
            "distinct2": round(distinct_sum[n] / seen, 4)}
        for n in names
    }
    for c in cand:
        stages[c].update({
            "win_rate": round(wins[c] / seen, 4),
            "win_rate_incl_ties": round((wins[c] + ties[c]) / seen, 4),
            "wins": wins[c], "losses": losses[c], "ties": ties[c],
            "kl_vs_base_per_tok": round(kl_tok[c] / max(1, kl_n[c]), 4),
        })

    print(f"\n--- win-rate vs {base} ({seen:,} held-out prompts, K={k}) ---")
    print(f"{'stage':<6} {'pass-rate':>10} {'distinct2':>10} {'win-rate':>9} {'>=base':>7} {'KL/tok':>8}")
    print(f"{base:<6} {stages[base]['pass_rate']:>10.3f} {stages[base]['distinct2']:>10.3f} "
          f"{'--':>9} {'--':>7} {'--':>8}")
    for c in cand:
        s = stages[c]
        print(f"{c:<6} {s['pass_rate']:>10.3f} {s['distinct2']:>10.3f} {s['win_rate']:>9.3f} "
              f"{s['win_rate_incl_ties']:>7.3f} {s['kl_vs_base_per_tok']:>8.3f}")
    return {"baseline": base, "prompts": seen, "samples_per_prompt": k, "stages": stages}


def run_regression(models, base, ctx, device) -> dict:
    """Validation perplexity per stage vs the baseline (fluency regression check)."""
    ppl = {}
    for n, m in models.items():
        r = evaluate_split(m, "valid", CONTEXT_LEN, EVAL_BATCH_SIZE, device, ctx, EVAL_MAX_BATCHES)
        ppl[n] = r["perplexity"]
    base_ppl = ppl[base]
    print(f"\n--- regression: validation perplexity (vs {base}) ---")
    print(f"{'stage':<6} {'val ppl':>9} {'delta':>9} {'':>8}")
    out = {}
    for n in models:
        delta = ppl[n] - base_ppl
        pct = 100.0 * delta / base_ppl
        flag = "" if n == base else ("[regressed]" if pct > 5 else "[ok]")
        print(f"{n:<6} {ppl[n]:>9.3f} {delta:>+9.3f} {flag:>8}")
        out[n] = {"val_ppl": round(ppl[n], 3), "val_ppl_delta": round(delta, 3),
                  "val_ppl_delta_pct": round(pct, 2)}
    return out


@torch.no_grad()
def _greedy_story(model, tok, prompt_ids, n_new, ctx, device) -> str:
    """Greedy (deterministic) continuation, trimmed at the first EOS."""
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    with ctx:
        out = model.generate(x, n_new, temperature=0.0)[0].tolist()
    gen = out[len(prompt_ids):]
    if tok.eos_id in gen:
        gen = gen[: gen.index(tok.eos_id)]
    return tok.decode(gen).strip()


def run_samples(models, tok, cfg, ctx, device, n_dump, max_prompt) -> None:
    """Print greedy generations from every stage side by side for a few prompts."""
    print(f"\n--- qualitative: {n_dump} greedy generations per stage ---")
    for i, (prompt, prompt_ids, n_new) in enumerate(held_out_prompts(tok, cfg, max_prompt)):
        if i >= n_dump:
            break
        words = parse_instruction(prompt).get("words", [])
        print(f"\n[{i + 1}] Words: {', '.join(words)}")
        for n, m in models.items():
            story = _greedy_story(m, tok, prompt_ids, n_new, ctx, device)
            print(f"  {n:<5} (r={verifiable_reward(prompt, story):.2f}): {story}")


def run_eval(
    n_prompts: int = WINRATE_NUM_PROMPTS,
    n_samples: int = WINRATE_SAMPLE_DUMP,
    only: set[str] | None = None,
    skip_winrate: bool = False,
    skip_regression: bool = False,
) -> dict:
    """Grade the SFT baseline against every present post-training stage.

    Resolves the baseline plus each candidate stage that has a run, loads them,
    and runs the win-rate / KL, perplexity-regression, and qualitative checks,
    writing the combined results to ``eval_results/``.

    Args:
        n_prompts: Held-out prompts to grade for win-rate.
        n_samples: Side-by-side greedy generations to print (0 to skip).
        only: Restrict to these stage labels, or None for every present stage.
        skip_winrate: Skip the win-rate / KL sweep.
        skip_regression: Skip the perplexity regression check.

    Returns:
        The results dict that was also written to ``eval_results/``.

    Raises:
        FileNotFoundError: The baseline stage has no runs yet.
        SystemExit: No candidate stages exist to grade (train one first).
    """
    torch.manual_seed(WINRATE_SEED)
    device, pt_dtype = resolve_device_dtype()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = (
        torch.autocast(device_type=device_type, dtype=pt_dtype)
        if pt_dtype is not torch.float32
        else nullcontext()
    )

    base_label, base_dir, base_run = WINRATE_BASELINE
    base_path = resolve_stage(base_label, base_dir, base_run)
    if base_path is None:
        raise FileNotFoundError(f"baseline {base_label} has no runs under {base_dir}")

    paths = {base_label: base_path}
    for label, ckpt_dir, run in WINRATE_MODELS:
        if only and label not in only:
            continue
        p = resolve_stage(label, ckpt_dir, run)
        if p is None:
            print(f"skip {label}: no runs under {ckpt_dir}")
        else:
            paths[label] = p
    if len(paths) < 2:
        raise SystemExit("no candidate stages to grade (train DPO/GRPO/PPO first)")

    models = {}
    for label, p in paths.items():
        m, _ = load_checkpoint(p, device)
        m.eval()
        models[label] = m
        print(f"{label:<5} {p}")
    cfg = models[base_label].cfg
    tok = Tokenizer()
    max_len = min(DPO_MAX_LEN or cfg.block_size, cfg.block_size)
    max_prompt = cfg.block_size - PREF_MIN_NEW_TOKENS
    print(f"device {device} | baseline {base_label} | candidates {list(models)[1:]}")

    results = {"baseline": base_label, "checkpoints": {k: str(v) for k, v in paths.items()}}
    if not skip_winrate:
        results["winrate"] = run_winrate(models, base_label, tok, cfg, ctx, device,
                                         n_prompts, max_len, max_prompt)
    if not skip_regression:
        results["regression"] = run_regression(models, base_label, ctx, device)
    if n_samples > 0:
        run_samples(models, tok, cfg, ctx, device, n_samples, max_prompt)

    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / WINRATE_RESULTS_FILE
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")
    return results


if __name__ == "__main__":
    run_eval()
