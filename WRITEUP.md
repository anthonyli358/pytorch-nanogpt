# TinyStories SLM: an end-to-end small language model, from pretraining through the full post-training stack

A ~14M-parameter decoder-only GPT built from scratch on the TinyStories corpus,
then taken through the entire modern post-training pipeline: **SFT → DPO →
GRPO / PPO**, graded by a verifiable-reward evaluation harness.

**The thesis, up front.** On a model this small the quality ceiling is low, so
the goal was never a benchmark number — it was to build and understand the
*machinery* end to end, and to see the failure modes (reward hacking,
over-optimization) with my own eyes on a model I could train in an afternoon.
TinyStories earns its place because its instruction variant hands you
**verifiable rewards for free** — did the story contain the required words? —
which turns the whole RL stack into a clean, model-free sandbox.

---

## TL;DR — what I learned by building it

- **Base → SFT is the capability cliff.** Word-inclusion pass-rate jumps from
  **0.18 → 0.83** with supervised fine-tuning. Everything after — DPO, GRPO,
  PPO — is refinement on top of that one big step.
- **Reward hacking is real and sneaky.** A verifiable "include these words"
  reward, optimized too hard, teaches the model to *cram words into repetitive,
  incoherent prose* while the pass-rate sits pinned at ~1.0. You cannot see it
  in the reward; you see it only when a quality metric rides alongside.
- **Validation loss is a trap for RL over-optimization.** DPO for 2+ epochs kept
  the win-rate high while blowing validation perplexity up **+42%**. One epoch:
  **+2.9%**. The regression check caught what the training objective couldn't.
- **KL-on-samples can understate the damage.** The measured KL from the
  reference was a modest 0.25 nats/token even as perplexity on natural text rose
  42% — the held-out perplexity check exposed drift the KL estimate missed.
- **Shaping the reward fixed the repetition.** Selecting preference pairs by a
  repetition-penalized score lifted DPO's diversity (distinct-2gram) from *below*
  SFT to **0.930 — above SFT's 0.905** and second only to the base, while keeping
  the **highest** pass-rate (0.967). The subtler, harder-to-separate pairs cost a
  higher training loss and bought real diversity — a deliberate, measured trade.
- **The cheapest method won; the heaviest underperformed.** Offline **DPO** beat
  both online RL methods on pass-rate *and* diversity. **PPO** — the full
  actor-critic stack — barely moved off SFT (win-rate 0.49, KL 0.002): the
  textbook "PPO is finicky and heavy" outcome, seen firsthand.

---

## Part I — Pretraining

### Architecture

A decoder-only GPT, pruned down from a seq2seq transformer (kept causal
self-attention, dropped cross-attention and the encoder entirely).

| | |
|---|---|
| `d_model` | 384 |
| layers | 6 |
| heads | 6 (head dim 64) |
| context | 256 |
| vocab | 8,000 |
| dropout | 0.0 |
| params | ~10.6M in the transformer blocks; ~13.8M with the tied token embedding |

Standard modern choices: pre-norm blocks, learned absolute positions, final
LayerNorm, and an LM head **weight-tied** to the token embedding. Dropout is off
— with ~530M training tokens against ~14M params, overfitting isn't the risk.

### Tokenizer

A SentencePiece **BPE**, vocab 8k, trained on 2M sampled lines of the train
split (BPE merges saturate fast on a corpus this repetitive, so more lines don't
change the vocabulary). Two deliberate choices:

- **`byte_fallback=True`** — unseen characters fall back to bytes, so the
  tokenizer never emits `<unk>`.
- **`<|endoftext|>` as a single user-defined symbol**, doubling as the document
  / EOS boundary. Native BOS/EOS are disabled; the model uses the EOS id.

### Data & packing

The whole corpus is encoded once to token ids and written as a flat `uint16`
memmap per split (8k vocab fits `uint16`), EOS between stories — the nanoGPT
pattern: pre-tokenize once, then sample random windows at train time.

| split | tokens | stories |
|---|---|---|
| train | 530,386,257 | 2,717,700 |
| val | 5,355,994 | 27,631 |

The story token-length distribution settled the context length: p50 **173**,
p90 **263**, p95 364, and **89.3%** of stories fit in ≤256 tokens. 256 it is.

### Training loop

AdamW (β 0.9/0.95, weight decay 0.1), cosine decay with linear warmup, grad clip
1.0, bf16 autocast, gradient accumulation for a large effective batch.

- batch 64 × accum 8 × context 256 ≈ **131k tokens/step**
- 10,000 steps ≈ **2.5 epochs** over 530M tokens (one epoch ≈ 4,046 steps)
- **Stateless sampling with replacement** — random windows drawn straight off the
  memmap, so there are no epochs to track and resume is trivial (minGPT-style).
  Tokens get seen at varying positions, which adds a little diversity.

**Result:** validation loss **1.300** → perplexity **≈3.67**.

---

## Part II — Post-training

The heart of the project. Ordered offline/simple → online/hard, each stage
independently useful, each regularizing against the SFT model as its reference.

```
Base  ──SFT──▶  SFT  ──DPO──▶  DPO
                 │
                 ├──GRPO──▶  GRPO   (online, verifiable reward)
                 └──PPO───▶  PPO    (full actor-critic, the classic stack)
```

### Step 1 — SFT: teaching the instruction schema

`TinyStories-Instruct` prefaces each story with constraints — a Summary, a list
of required **Words**, a Sentence to include, feature flags (dialogue / bad
ending / moral / plot twist) — followed by `Story:` and the story.

- **Method:** the *same* next-token loss as pretraining, but the **prompt is
  masked** (`labels = -1` up to and including `Story:`) so only the response
  (+EOS) contributes. Init from the pretrained checkpoint; reuse the optimizer /
  autocast / clipping machinery.
- **Data discipline:** examples longer than the context are **dropped, never
  truncated** — the model only ever sees complete stories.
- **Schedule:** 1 epoch (~50k steps over ~1.6M examples), LR 3e-4 (below the 6e-4
  pretraining peak), cosine + warmup, early-stop on masked val loss.

**Result:** masked-response val loss **1.176**. But the loss number isn't the
story — the *capability* is. In the evaluation harness, word-inclusion
pass-rate goes **Base 0.18 → SFT 0.83**: SFT is what makes instruction-following
possible at all. The cost, honestly reported: plain-story perplexity rises
**3.67 → 8.84** as the model specializes to the instruct distribution.

**Why it matters twice:** the SFT model is both the deliverable *and* the frozen
reference policy every later RL/DPO stage measures its KL against.

### Step 2 — Reward design: verifiable, then shaped

The pivot everything downstream depends on. The cheapest, cleanest signal is
**verifiable**: the fraction of required `Words:` present in the story (lenient
prefix match, set semantics — each word counts once, so repeating a word buys
nothing). Zero models, zero labels, range [0, 1]. LLM-as-judge and a trained
reward model were deliberately skipped — the verifiable reward is the right fit
here and keeps the RL loop model-free.

But a pure word-inclusion reward is **gameable** (see the DPO story below), so it
gets **shaped**:

```
shaped_reward = verifiable_reward − 0.5 · repetition_penalty
repetition_penalty = 1 − (distinct 2-grams / total 2-grams)
```

Now "include the words **and** stay diverse" is the objective — cramming and
looping stop paying. This shaped signal drives DPO pair selection and the online
RL rewards.

### Step 3 — DPO, and a reward-hacking war story

DPO needs preference pairs. For each instruct prompt I sample **K=4** completions
(temperature > 0 for diversity), score each with the shaped reward, and emit
`(prompt, chosen, rejected)` on a reward spread — chosen being the
higher-scoring (and, on a word-inclusion tie, the *less repetitive*) completion.
Ties carry no signal and are skipped.

The loss is the standard offline objective against the **frozen SFT reference**,
with the KL leash baked in:

```
L = −log σ( β · [ (logπθ(y_w) − logπref(y_w)) − (logπθ(y_l) − logπref(y_l)) ] ),   β = 0.1
```

No reward model, no sampling loop, no critic — a classification-style loss. Most
stable, easiest to debug.

**The war story.** My first instinct — a small preference set, so train a couple
of epochs — was wrong in an instructive way:

| DPO run | val ppl vs SFT | KL/tok | win-rate | verdict |
|---|---|---|---|---|
| 2+ epochs | **+42%** | 0.25 | ~1.0 | reward-hacked |
| 1 epoch | **+2.9%** | 0.04 | 0.744 | healthy |

At 2+ epochs the win-rate looked *great* while the model quietly learned to jam
required words into repetitive, incoherent stories — "*there was a new waffle …
they liked to eat the new waffle … very good at making waffle*" — pass-rate 1.0,
prose in ruins. Two lessons fell out of this:

1. **Validation loss will not save you.** It keeps dropping as the implicit
   reward climbs; the early-stop-on-val-loss guard is blind to this failure. The
   **regression check** (held-out perplexity) is what caught it.
2. **KL-on-samples understated the damage.** 0.25 nats/token looked benign next
   to a 42% perplexity blow-up. Measuring KL on the reward-passing completions
   doesn't capture how far the distribution shifted on *natural* text.

The one-epoch model was shipped as `dpo.pt`.

### Step 4 — GRPO (online, verifiable reward)

*Group Relative Policy Optimization — PPO minus the critic.* Each step:

1. **Rollout** — sample a *group* of G completions per prompt from the policy.
2. **Reward** — score each with the shaped reward.
3. **Advantage** — normalize each reward *within its group*:
   `A = (r − mean) / std`. The group mean is the baseline; **no value network**.
   A group whose completions all score the same has zero advantage and
   contributes no gradient.
4. **Update** — a clipped PPO surrogate on the response tokens, with a per-token
   KL leash to the frozen reference (k3 estimator).

Policy and reference both init from SFT. This is the roadmap's recommended path
— light, and a natural fit for verifiable rewards. **Result:** pass-rate 0.950,
diversity 0.920 (above SFT), win-rate 0.694 (≥ SFT on 95% of prompts) — nearly
DPO's numbers at a **third of the KL** (0.011), the most conservative improver.

### Step 5 — PPO (the full classic stack)

The "build the whole thing for the education" path. Everything GRPO drops, added
back: a **critic** (a second GPT backbone with a scalar value head), a per-token
KL-penalty reward with the shaped reward on the terminal token, **GAE**
advantages over the value baseline, and a clipped actor + clipped value loss on
separate optimizers. The trained reward model (the classic RLHF third stage) is
skipped — PPO optimizes the verifiable reward directly. **Result:** the weakest
of the three — pass-rate 0.867 (barely above SFT's 0.826), win-rate 0.492 (a
coin flip vs SFT), KL 0.002 (it barely moved). With only 400 steps and a critic
learning from scratch, early advantages are too noisy to drive confident updates.
This is exactly the outcome the roadmap predicted — heavy, finicky to stabilize —
and it's the empirical case for preferring GRPO on verifiable rewards.

---

## Evaluation

Post-training is graded on **held-out** instruct-valid prompts (never the train
prompts the preference pairs came from), in a single command, across three legs:

- **Win-rate** — verifiable pass-rate; does the stage beat SFT on this prompt?
- **KL from the reference** — the over-optimization gauge.
- **Regression** — validation perplexity + a **distinct-2gram** diversity
  number + a side-by-side greedy sample dump, so a quality collapse can't hide
  behind a healthy pass-rate.

The whole ladder, one 500-prompt held-out sweep (K=4 completions per prompt):

| stage | pass-rate | distinct-2 | win-rate vs SFT | KL/tok |
|---|---|---|---|---|
| Base | 0.177 | **0.971** | 0.000 (≥ SFT on 0.004) | 1.822 |
| SFT | 0.826 | 0.905 | — (baseline) | — |
| **DPO** | **0.967** | **0.930** | **0.736** (≥ SFT on 0.970) | 0.028 |
| GRPO | 0.950 | 0.920 | 0.694 (≥ SFT on 0.948) | 0.011 |
| PPO | 0.867 | 0.907 | 0.492 (≥ SFT on 0.754) | 0.002 |

What the table says:

- **The base model is the *most* diverse** (distinct-2 0.971) precisely *because*
  it ignores the task — free-flowing stories instead of crammed words (pass-rate
  0.18). So "less repetition" alone isn't the goal; the target is diversity
  **given** instruction-following. The base sets the diversity ceiling.
- **DPO wins outright** — the highest pass-rate (0.967) *and* the best diversity
  of any trained model (0.930, clearing SFT's 0.905), at a tiny KL (0.028). The
  shaped-pair selection did its job: pre-shaping DPO sat *below* SFT on diversity;
  now it clears it. The cheapest, simplest post-training method came out on top.
- **GRPO is a close, conservative second** — nearly DPO's numbers at a third of
  the KL (0.011): most of the gain, least drift from the reference.
- **PPO underperformed** — a near-coin-flip against SFT (0.492) and essentially
  no movement (KL 0.002). The heaviest method delivered the least in this budget.

---

## Speculative decoding (Part I, step 8)

A throughput extension, orthogonal to the post-training story: a small **draft**
model proposes γ tokens, the **target** verifies them in one forward pass and
accepts the longest prefix consistent with its own distribution (rejection-
sampling correction on the first mismatch). The output is distributed *exactly*
as sampling from the target alone — the draft affects only speed. The algorithm
and its correctness are done: a unit test confirms the output distribution
matches the target's to **total-variation 0.012**, and draft==target accepts
every token (γ+1 per forward).

**Measured — an honest negative result.** With a trained 1.4M-param draft against
the 13.7M target (γ=4), there is **no speedup**: wall-clock **0.5–1.3×**, a net
slowdown on most prompts. Two compounding reasons: (1) the target's forward is
already sub-millisecond, so the per-token Python overhead of the accept/reject
loop — distribution filtering, sampling, and a GPU sync *per position* —
outweighs the savings from fewer forwards; (2) the small draft mimics the target
poorly, so acceptance is only ~0.20 (about **1.8 tokens per target forward**
against the γ+1 = 5 ceiling). Speculative decoding pays off when the target
forward is the bottleneck — large models — not at 14M. The technique is *correct*
(the output is provably a target sample); the economics simply require scale.

## What's next

- **Loss-curve figures** — the base/SFT runs predate metrics logging, so the
  curves would need a (seeded, reproducible) retrain; all newer stages log
  `metrics.csv` + `loss.png` automatically.
- **Stronger reward shaping** — DPO's diversity gain rode on ~26% repetition-
  tiebreak pairs; raising the repetition weight (or the pair-emit threshold)
  could push distinct-2 further toward the base ceiling.

---

## The one-paragraph takeaway

On a ~14M-parameter model the scores are never going to impress anyone — and
that was the point. The value was in wiring up the complete stack (tokenizer →
pretraining → SFT → verifiable rewards → DPO → GRPO → PPO → a real eval harness),
and in watching the textbook failure modes actually happen: the reward getting
hacked, validation loss lying about it, KL understating the drift, and a simple
diversity metric plus a held-out perplexity check being the things that told the
truth. That's a much more durable education than a good number would have been.
