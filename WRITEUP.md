# pytorch-nanogpt

Pytorch implementation of a decoder-only [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) trained from scratch, then taken through the full modern post-training stack: **SFT → DPO → GRPO → PPO**.

Building a ~14M-parameter small language model end-to-end on the [TinyStories](https://arxiv.org/abs/2305.07759) corpus, graded by a verifiable-reward evaluation harness.

**The thesis, up front.** On a model this small the quality ceiling is low, so the goal was never a benchmark number — it was to build and understand the *machinery* end to end, and to watch the failure modes (reward hacking, over-optimization) happen with my own eyes on a model I could train in an afternoon. TinyStories earns its place because its instruction variant hands you **verifiable rewards for free** — did the story contain the required words? — which turns the whole RL stack into a clean, model-free sandbox.

## Results

The whole ladder, evaluated in one 500-prompt held-out sweep (K=4 completions per prompt), each post-training stage regularised against the SFT model as its frozen reference:

| stage | pass-rate | distinct-2 | win-rate vs SFT | KL/tok |
|---|---|---|---|---|
| Base | 0.177 | **0.971** | 0.000 (≥ SFT on 0.004) | 1.822 |
| SFT | 0.826 | 0.905 | — (baseline) | — |
| **DPO** | **0.967** | **0.930** | **0.736** (≥ SFT on 0.970) | 0.028 |
| GRPO | 0.950 | 0.920 | 0.694 (≥ SFT on 0.948) | 0.011 |
| PPO | 0.867 | 0.907 | 0.492 (≥ SFT on 0.754) | 0.002 |

- **pass-rate** — the verifiable reward: fraction of the required `Words:` present in the story.
- **distinct-2** — distinct 2-grams / total 2-grams, a diversity proxy that catches repetitive, looping prose.
- **win-rate** — does the stage beat SFT on a given held-out prompt?
- **KL/tok** — divergence from the frozen SFT reference; the over-optimization gauge.

What the table says:

- **Base → SFT is the capability cliff.** Word-inclusion pass-rate jumps **0.18 → 0.83** with supervised fine-tuning alone. Everything after — DPO, GRPO, PPO — is refinement on top of that one big step.
- **The base model is the *most* diverse** (distinct-2 0.971) precisely *because* it ignores the task — free-flowing stories instead of crammed-in words (pass-rate 0.18). So "less repetition" alone isn't the goal; the target is diversity **given** instruction-following. The base sets the diversity ceiling.
- **DPO wins outright** — the highest pass-rate (0.967) *and* the best diversity of any trained model (0.930, clearing SFT's 0.905), at a tiny KL (0.028). The cheapest, simplest post-training method came out on top.
- **GRPO is a close, conservative second** — nearly DPO's numbers at a third of the KL (0.011): most of the gain, least drift from the reference.
- **PPO underperformed** — a near-coin-flip against SFT (0.492) and essentially no movement (KL 0.002). The heaviest method delivered the least in this budget; its training curve shows exactly why (Step 11).

And the failure modes that made the project worth building:

- **Reward hacking is real and sneaky.** A verifiable "include these words" reward, optimized too hard, teaches the model to *cram words into repetitive, incoherent prose* while the pass-rate sits pinned at ~1.0. You cannot see it in the reward; you see it only when a quality metric rides alongside.
- **Validation loss is a trap for RL over-optimization.** DPO for 2+ epochs kept the win-rate high while blowing validation perplexity up **+42%**. One epoch: **+2.9%**. The regression check caught what the training objective couldn't.
- **KL-on-samples can understate the damage.** The measured KL from the reference looked modest even as perplexity on natural text rose 42% — the held-out perplexity check exposed drift the KL estimate missed.

## Getting Started

1. For GPU, go to the [pytorch](https://pytorch.org/get-started/locally/) website and select the local installs to get the bash command.

2. To use this repo, [install uv](https://docs.astral.sh/uv/getting-started/installation/). We use the [pytorch index](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) for CUDA 12.6 [in this project](pyproject.toml).

3. Now install dependencies.

```bash
uv sync
```

4. The pipeline is driven from [src/main.py](src/main.py): uncomment one stage at a time and run it as a module. The stages are ordered by dependency — download → tokenizer → pack → pretrain → SFT → preferences → DPO/GRPO/PPO → eval — and each writes a checkpoint the next stage reads.

```bash
uv run python -m src.main
```

5. TinyStories is fetched straight from the HF Hub (`download_data()`), so there's no manual dataset step. Two numbers ripple through everything from packing onward and are pinned in [src/config.py](src/config.py): **vocab size (8k)** and **context length (256)**.

6. Each training run saves the best checkpoint (by validation loss) under `checkpoints/<stage>/<timestamp>/`, alongside a `metrics.csv` and loss/diagnostic curves. Post-training stages init from the SFT checkpoint, so run SFT before DPO/GRPO/PPO.

7. Evaluate the whole ladder — win-rate, KL, and the perplexity/diversity regression check — in a single command via `run_eval()`, and compare vanilla vs speculative decoding via `compare_decoding()`. During development, check shapes and the speculative-decoding correctness proof with the [tests](tests/speculative_test.py).

```bash
uv run pytest
```

## Development

This repo implements a GPT and its post-training stack from scratch for understanding. In production you'd reach for `torch.nn.functional.scaled_dot_product_attention` and a library like TRL for the fused kernels and battle-tested trainers.

The most important takeaways from this exercise:

- **Decoder-only, because story generation is continuation, not translation.** There's no clean input → output; prompt and story are the same stream of text, so the [model](src/models/gpt.py) drops cross-attention and the encoder entirely and keeps only causal self-attention.
- **SFT is the single biggest capability gain.** The Base→SFT jump (pass-rate 0.18 → 0.83) dwarfs everything the RL methods add on top. Get [SFT](src/training/sft.py) right first.
- **A verifiable reward removes the reward model.** Because TinyStories-Instruct tells you the required words, the [reward](src/eval/reward.py) is a pure function — no LLM judge, no Bradley-Terry model to train. This is what keeps the whole RL loop model-free.
- **A pure verifiable reward is gameable, so shape it.** Subtracting a repetition penalty is what stops the model cramming words into looping prose. Cramming stops paying.
- **You cannot trust the training objective to tell you when RL has gone wrong.** A held-out **perplexity** regression check plus a **diversity** metric caught over-optimization that both validation loss and sampled KL missed.
- **On verifiable rewards, GRPO beats PPO by construction.** A learned critic collapses to the near-constant reward and starves the actor of advantage; GRPO's group-relative baseline sidesteps that. The cheapest methods won.

## Walkthrough

Rather than split results from implementation, each step below covers the method, how it was trained, and what came out — Part I builds the base model, Part II is the post-training stack that's the heart of the project.

```
Base  ──SFT──▶  SFT  ──DPO──▶  DPO
                 │
                 ├──GRPO──▶  GRPO   (online, verifiable reward)
                 └──PPO───▶  PPO    (full actor-critic, the classic stack)
```

---

### Part I — Pretraining

#### 1. Data

Raw `.txt` per split is pulled from the HF Hub with `<|endoftext|>` separators intact ([src/data/data.py](src/data/data.py)). The download is cache-backed, so re-runs are free.

| split | tokens | stories |
|---|---|---|
| train | 530,386,257 | 2,717,700 |
| val | 5,355,994 | 27,631 |

#### 2. Tokenizer

A SentencePiece **BPE** ([src/models/tokenizer.py](src/models/tokenizer.py)), vocab 8k, `model_type=bpe`, trained on 2M lines sampled from the ~14.6M-line train split. BPE merge frequencies saturate fast on a corpus this repetitive, so more lines don't change the vocabulary. Byte-Pair Encoding iteratively merges the most frequent adjacent pair into a new token until the vocabulary is full — e.g. if many words end in "er" (lower, higher, faster), the frequent `e`,`r` pair merges into a single `er` token. Three deliberate choices:

- **`byte_fallback=True`** — unseen characters fall back to bytes, so the tokenizer never emits `<unk>`. This is the single biggest quality lever over a word-level tokenizer, which would cap rare words at `<unk>`.
- **`<|endoftext|>` as a single user-defined symbol** (never split by BPE), doubling as the document / EOS boundary. Native BOS/EOS are disabled; the model uses the EOS id.
- **`max_sentence_length=8192`** — raised from the 4096-byte default so no full stories are dropped when learning merges. `encode()` itself is never length-limited.

#### 3. Preprocess / pack

The whole corpus is encoded once to token ids and written as a flat `uint16` memmap per split (8k vocab fits `uint16`), EOS between stories ([src/data/pack.py](src/data/pack.py)). This is the nanoGPT pattern: pre-tokenize once, then sample random windows at train time.

The story token-length distribution settled the context length — p50 **173**, p90 **263**, p95 364, and **89.3%** of stories fit in ≤256 tokens. So 256 it is: long enough to hold nearly every story whole, short enough to keep the positional table and attention cheap.

#### 4. Model

A decoder-only GPT ([src/models/gpt.py](src/models/gpt.py)), pruned down from a seq2seq transformer — kept causal self-attention ([src/models/attention.py](src/models/attention.py)), dropped cross-attention and the encoder entirely.

| | |
|---|---|
| `d_model` | 384 |
| layers | 6 |
| heads | 6 (head dim 64) |
| context | 256 |
| vocab | 8,000 |
| dropout | 0.0 |
| params | ~10.6M in the transformer blocks; ~13.8M with the tied token embedding |

Standard modern choices ([src/models/block.py](src/models/block.py)): pre-norm blocks, learned absolute positions, a final LayerNorm, and an LM head **weight-tied** to the token embedding. Head dim 64 is the universal choice (`384 / 6`). Dropout is off — with ~530M training tokens against ~14M params, overfitting isn't the risk.

#### 5. Training loop

AdamW (β 0.9/0.95, weight decay 0.1), cosine decay with linear warmup, grad clip 1.0, bf16 autocast, and gradient accumulation for a large effective batch ([src/training/pretrain.py](src/training/pretrain.py)). Gradient accumulation simulates a batch the GPU can't fit: run several smaller microbatches, sum the gradients, and step once.

- batch 64 × accum 8 × context 256 ≈ **131k tokens/step**
- 10,000 steps ≈ **2.5 epochs** over 530M tokens (one epoch ≈ 4,046 steps)
- **Stateless sampling with replacement** — random windows drawn straight off the memmap, following [Karpathy's minGPT](https://github.com/karpathy/minGPT). There are no epochs to track, resume is trivial and decoupled from dataset size, and tokens get seen at varying positions, which adds a little diversity. The trade-off is no coverage guarantee — fine here, where we have far more data than training budget.

**Result:** validation loss **1.300** → perplexity **≈3.67**.

<p align="left">
    <img src="checkpoints/base/20260824_170504/loss.png" alt="pretraining loss" width="600"/>
</p>

#### 6. Sampling

Autoregressive decode with temperature + top-k/top-p ([src/inference/generate.py](src/inference/generate.py)) — sampling, not beam search, because this is open-ended generation: prompt in → story out.

---

### Part II — Post-training

The heart of the project. Ordered offline/simple → online/hard, each stage independently useful, each regularizing against the SFT model as its reference. RLHF isn't a separate method here — it's the umbrella for SFT → reward model → PPO; DPO and PPO/GRPO are alternative routes from the *same* preference data, and GRPO is just PPO minus the critic.

#### 7. SFT — teaching the instruction schema

`TinyStories-Instruct` prefaces each story with constraints — a Summary, a list of required **Words**, a Sentence to include, feature flags (dialogue / bad ending / moral / plot twist) — followed by `Story:` and the story.

- **Method** ([src/training/sft.py](src/training/sft.py)): the *same* next-token loss as pretraining, but the **prompt is masked** (`labels = -1` up to and including `Story:`, see [src/data/sft_data.py](src/data/sft_data.py)) so only the response (+EOS) contributes. Init from the pretrained checkpoint; reuse the optimizer / autocast / clipping machinery.
- **Data discipline:** examples longer than the context are **dropped, never truncated** — the model only ever sees complete stories.
- **Schedule:** 1 epoch (~50k steps over ~1.6M examples), LR 3e-4 (below the 6e-4 pretraining peak), cosine + warmup, early-stop on masked val loss.

**Result:** masked-response val loss **1.176**. But the loss number isn't the story — the *capability* is. Word-inclusion pass-rate goes **Base 0.18 → SFT 0.83**: SFT is what makes instruction-following possible at all. The cost, honestly reported: plain-story perplexity rises **3.67 → 8.84** as the model specializes to the instruct distribution.

**Why it matters twice:** the SFT model is both the deliverable *and* the frozen reference policy every later RL/DPO stage measures its KL against.

#### 8. Reward design — verifiable, then shaped

The pivot everything downstream depends on ([src/eval/reward.py](src/eval/reward.py)). The cheapest, cleanest signal is **verifiable**: the fraction of required `Words:` present in the story (lenient prefix match, set semantics — each word counts once, so repeating a word buys nothing). Zero models, zero labels, range [0, 1]. LLM-as-judge and a trained reward model were deliberately skipped — the verifiable reward is the right fit here and keeps the RL loop model-free.

But a pure word-inclusion reward is **gameable** (see the DPO war story), so it gets **shaped**:

$$r_\text{shaped} = r_\text{verifiable} - 0.5 \cdot r_\text{penalty}, \qquad r_\text{penalty} = 1 - \frac{\text{distinct 2-grams}}{\text{total 2-grams}}$$

Now "include the words **and** stay diverse" is the objective — cramming and looping stop paying. This shaped signal drives DPO pair selection and the online RL rewards.

#### 9. DPO — and a reward-hacking war story

DPO needs preference pairs. For each instruct prompt I sample **K=4** completions (temperature > 0 for diversity), score each with the shaped reward, and emit `(prompt, chosen, rejected)` on a reward spread — chosen being the higher-scoring (and, on a word-inclusion tie, the *less repetitive*) completion ([src/data/make_preferences.py](src/data/make_preferences.py) → `data/dpo/pairs.jsonl`). Ties carry no signal and are skipped.

The loss ([src/training/dpo.py](src/training/dpo.py)) is the standard offline objective against the **frozen SFT reference**, with the KL leash baked in:

$$L = -\log \sigma\!\left( \beta \left[ \big(\log \pi_\theta(y_w) - \log \pi_\text{ref}(y_w)\big) - \big(\log \pi_\theta(y_l) - \log \pi_\text{ref}(y_l)\big) \right] \right), \quad \beta = 0.1$$

where each $\log \pi(y)$ is the summed log-prob of the response tokens. No reward model, no sampling loop, no critic — a classification-style loss. Most stable, easiest to debug.

**The war story.** My first instinct — a small preference set, so train a couple of epochs — was wrong in an instructive way:

| DPO run | val ppl vs SFT | KL/tok | win-rate | verdict |
|---|---|---|---|---|
| 2+ epochs | **+42%** | 0.25 | ~1.0 | reward-hacked |
| 1 epoch | **+2.9%** | 0.04 | 0.744 | healthy |

At 2+ epochs the win-rate looked *great* while the model quietly learned to jam required words into repetitive, incoherent stories — "*there was a new waffle … they liked to eat the new waffle … very good at making waffle*" — pass-rate 1.0, prose in ruins. Two lessons fell out:

1. **Validation loss will not save you.** It keeps dropping as the implicit reward climbs; the early-stop-on-val-loss guard is blind to this failure. The **regression check** (held-out perplexity) is what caught it.
2. **KL-on-samples understated the damage.** 0.25 nats/token looked benign next to a 42% perplexity blow-up. Measuring KL on the reward-passing completions doesn't capture how far the distribution shifted on *natural* text.

The one-epoch model was shipped as `dpo.pt`. Its diversity gain was deliberate: selecting pairs by the repetition-penalized score lifted distinct-2gram from *below* SFT to **0.930 — above SFT's 0.905**, while keeping the **highest** pass-rate (0.967). The subtler, harder-to-separate pairs cost a higher training loss and bought real diversity — a measured trade.

<p align="left">
    <img src="checkpoints/dpo/20260824_185134/curves.png" alt="DPO training curves" width="600"/>
</p>

#### 10. GRPO — online, verifiable reward

*Group Relative Policy Optimization — PPO minus the critic* ([src/training/grpo.py](src/training/grpo.py), shared RL plumbing in [src/training/rl_common.py](src/training/rl_common.py)). Each step:

1. **Rollout** — sample a *group* of G completions per prompt from the policy.
2. **Reward** — score each with the shaped reward.
3. **Advantage** — normalize each reward *within its group*, $A = (r - \mu_\text{group}) / \sigma_\text{group}$. The group mean is the baseline; **no value network**. A group whose completions all score the same has zero advantage and contributes no gradient.
4. **Update** — a clipped PPO surrogate on the response tokens, with a per-token KL leash to the frozen reference (k3 estimator).

Policy and reference both init from SFT. This is the recommended path — light, and a natural fit for verifiable rewards.

**Result:** pass-rate 0.950, diversity 0.920 (above SFT), win-rate 0.694 (≥ SFT on 95% of prompts) — nearly DPO's numbers at a **third of the KL** (0.011), the most conservative improver. The curves show reward climbing 0.80 → 0.91 with KL rising in a controlled 0 → 0.016.

<p align="left">
    <img src="checkpoints/grpo/20260820_165555/curves.png" alt="GRPO training curves" width="600"/>
</p>

#### 11. PPO — the full classic stack

The "build the whole thing for the education" path ([src/training/ppo.py](src/training/ppo.py)). Everything GRPO drops, added back: a **critic** (a second GPT backbone with a scalar value head), a per-token KL-penalty reward with the shaped reward on the terminal token, **GAE** advantages over the value baseline, and a clipped actor + clipped value loss on separate optimizers. The trained reward model (the classic RLHF third stage) is skipped — PPO optimizes the verifiable reward directly.

**Result:** the weakest of the three — pass-rate 0.867 (barely above SFT's 0.826), win-rate 0.492 (a coin flip vs SFT), KL 0.002 (it barely moved).

**The training curve shows exactly why.** The critic's value loss collapses from 1.13 to ~0 within the first ~30 steps and flatlines — it trivially learned to predict the near-constant reward (~0.8 on almost every rollout, because the verifiable reward is low-variance). Once the value baseline matches the reward, the **GAE advantages vanish**: $A = \text{return} - \text{value} \approx 0$. With no advantage signal the actor gets essentially no gradient and stalls — actor loss hovers at ~−0.002 and reward stays flat and noisy for the remaining 370 steps. That's the mechanism behind "PPO is heavy and finicky," made concrete: a critic that fits a flat reward starves its own actor. GRPO sidesteps it — its group-relative baseline normalizes *within* each prompt's samples, so even a low-variance reward yields usable advantages. This is the empirical case for preferring GRPO on verifiable rewards.

<p align="left">
    <img src="checkpoints/ppo/20260824_111159/curves.png" alt="PPO training curves — value-loss collapse" width="600"/>
</p>

#### 12. Evaluation

Post-training is graded on **held-out** instruct-valid prompts (never the train prompts the preference pairs came from), in a single command ([src/eval/winrate.py](src/eval/winrate.py)), across three legs:

- **Win-rate** — verifiable pass-rate; does the stage beat SFT on this prompt?
- **KL from the reference** — the over-optimization gauge.
- **Regression** ([src/eval/perplexity.py](src/eval/perplexity.py)) — validation perplexity + a **distinct-2gram** diversity number + a side-by-side greedy sample dump, so a quality collapse can't hide behind a healthy pass-rate.

The headline ladder is in [Results](#results) above. The one number that matters most: the base model's high diversity (0.971) is *not* a win — it comes from ignoring the task. The real target is diversity **given** instruction-following, and DPO's 0.930 at a 0.967 pass-rate is the best balance struck.

#### 13. Speculative decoding

A throughput extension, orthogonal to the post-training story ([src/inference/speculative.py](src/inference/speculative.py)): a small **draft** model ([src/training/draft.py](src/training/draft.py)) proposes γ tokens, the **target** verifies them in one forward pass and accepts the longest prefix consistent with its own distribution (rejection-sampling correction on the first mismatch). The output is distributed *exactly* as sampling from the target alone — the draft affects only speed. A [unit test](tests/speculative_test.py) confirms the output distribution matches the target's to **total-variation 0.012**, and draft==target accepts every token (γ+1 per forward).

**Measured — an honest negative result.** With a trained 1.4M-param draft against the 13.7M target (γ=4), there is **no speedup**: wall-clock **0.5–1.3×**, a net slowdown on most prompts. Two compounding reasons: (1) the target's forward is already sub-millisecond, so the per-token Python overhead of the accept/reject loop — distribution filtering, sampling, and a GPU sync *per position* — outweighs the savings from fewer forwards; (2) the small draft mimics the target poorly, so acceptance is only ~0.20 (about **1.8 tokens per target forward** against the γ+1 = 5 ceiling). Speculative decoding pays off when the target forward is the bottleneck — large models — not at 14M. The technique is *correct* (the output is provably a target sample); the economics simply require scale.

## What's next

- **Loss-curve figures for base/SFT** — the earliest runs predate metrics logging; the newer stages log `metrics.csv` + curves automatically.
- **Stronger reward shaping** — DPO's diversity gain rode on ~26% repetition-tiebreak pairs; raising the repetition weight (or the pair-emit threshold) could push distinct-2 further toward the base ceiling.

## The one-paragraph takeaway

On a ~14M-parameter model the scores are never going to impress anyone — and that was the point. The value was in wiring up the complete stack (tokenizer → pretraining → SFT → verifiable rewards → DPO → GRPO → PPO → a real eval harness), and in watching the textbook failure modes actually happen: the reward getting hacked, validation loss lying about it, KL understating the drift, and a simple diversity metric plus a held-out perplexity check being the things that told the truth. That's a much more durable education than a good number would have been.

## Resources

- Radford et al., [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT, 2018)
- Eldan & Li, [*TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*](https://arxiv.org/abs/2305.07759) (2023)
- Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) (DPO, 2023)
- Shao et al., [*DeepSeekMath*](https://arxiv.org/abs/2402.03300) (GRPO, 2024)
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (PPO, 2017)
- Leviathan et al., [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192) (2023)
- Karpathy, [minGPT](https://github.com/karpathy/minGPT) for the stateless-sampling training pattern
- [SentencePiece](https://github.com/google/sentencepiece) for subword tokenisation


stage	curves.png shows
GRPO	reward 0.80→0.91 ↑, KL 0→0.016 ↑ (controlled), groups active
PPO	reward flat/noisy, value-loss collapse (the underperformance smoking gun), tiny actor loss
DPO	loss ↓, margin ↑ to +4.4, accuracy ↑ to ~0.8



--- win-rate vs SFT (500 held-out prompts, K=4) ---
stage   pass-rate  distinct2  win-rate  >=base   KL/tok
SFT         0.837      0.907        --      --       --
Base        0.154      0.974     0.000   0.002    2.139
DPO         0.967      0.930     0.690   0.952    0.194
GRPO        0.950      0.920     0.674   0.934    0.174
PPO         0.867      0.907     0.444   0.728    0.154

--- regression: validation perplexity (vs SFT) ---
stage    val ppl     delta         
SFT        9.486    +0.000         
Base       3.688    -5.797     [ok]
DPO        8.882    -0.604     [ok]
GRPO       8.878    -0.608     [ok]
PPO        8.871    -0.615     [ok]

--- qualitative: 6 greedy generations per stage ---
downloading TinyStories-Instruct-train.txt ...
downloading TinyStories-Instruct-valid.txt ...

[1] Words: escape, war, tall
  SFT   (r=1.00): Tom and Lily were playing in the forest. They saw a big, tall tree house. They wanted to go inside. "Let's go in!" Tom said. "No, it's too scary!" Lily said. They ran and ran. They saw a big, scary house. They heard a loud noise. "Who is there?" Lily asked. "It's a war!" Tom said. "We need to escape!" They ran and ran. They saw a big tree house. They went inside. They saw a big tree house. They heard a voice. "Hello, I am Ben. Do you want to come in?" Ben said. "Yes, please!" Tom said. They were happy. They said "We are friends. We will stay here with you." They hugged Ben. They were not scared anymore. They were safe. They said "Thank you, Ben. You are a good friend." They went back to the forest. They played and had fun.
  Base  (r=0.00): army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, army, ar
  DPO   (r=1.00): Once upon a time, there was a tall man named Tom. He lived in a small house in the forest. One day, Tom heard a loud noise outside. He went to the window and saw a big war. The war was very scary and Tom was scared. Tom saw a little girl named Lily in the forest. She was crying and crying. Tom asked, "Why are you crying, Lily?" Lily said, "I want to escape from the scary house and find a place to live." Tom wanted to help Lily, so he opened the door and went inside. Tom and Lily walked and walked until they found a big tree house. They went inside and met a kind old man named Ben. Ben said, "Hello, I am Ben. I live in this house. Do you want to stay with me?" Tom and Lily were happy to have a new friend. They all lived together and had fun in the forest.
  GRPO  (r=1.00): Once upon a time, there was a tall man named Tom. Tom lived in a small house in the forest. One day, Tom heard a loud noise outside. He went to see what it was. Tom saw a big war with many people. They were all scared and wanted to escape. Tom said, "I am scared. I want to escape." Tom went to the big house and saw a tall tree. He said, "Hello, tree! I am Tom. I want to escape. Can you help me?" The tree said, "Yes, Tom. I can help you. I am a nice tree. I will keep you safe." Tom and the tree went inside the house. They met a nice man named Ben. Ben said, "I am Ben. I live in this house. Do you want to stay with me?" Tom and Ben were happy. They stayed with Ben and had a fun day.
  PPO   (r=1.00): Tom and Lily are friends. They like to play in the forest. They see a big house in the distance. They want to go see it. "Let's go!" Tom says. "We can escape from the house." "OK!" Lily says. They run to the house. They are brave. They see a big door. It is dark and scary. They hear a loud noise. They are scared. They run to the door. They open the door. They see a big tree. It is a house. It is very tall. It has many windows. It is dark and scary. "Who is in there?" Tom asks. "I am Ben. He is a war. He is a big war. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean. He is mean.

[2] Words: meet, waffle, new
  SFT   (r=0.67): Once upon a time, there was a little girl named Lily. She got a new waffle for breakfast. It was big and round and had lots of colors. Lily was so happy to eat it. She ate it all up and it was very yummy. After breakfast, Lily went to play outside. She saw a new friend named Timmy. Timmy was also eating a waffle. Lily said, "Hi Timmy, do you want to play with me?" Timmy said, "Yes, I do!" They played together and had so much fun. After playing, Lily and Timmy went back inside. Lily's mom asked, "Did you have fun today?" Lily said, "Yes, I met a new friend named Timmy. He was my new friend!" Her mom smiled and said, "That sounds like a great day." Summary: Lily gets a new waffle for breakfast and plays with a new friend named Timmy, making it a fun day.
  Base  (r=0.33): Words, new friends.
  DPO   (r=1.00): Once upon a time, there was a little girl named Lily. She had a new waffle for breakfast. She was very happy and excited to eat it. But when she went to the kitchen, she saw a new waffle on the table. She didn't know where it came from, but she thought it looked yummy. Lily decided to go outside and meet her new friend, Timmy. She walked up to him and said, "Hi Timmy, do you want to meet me?" Timmy smiled and said, "Yes, I do! I'm very happy to meet you." Lily was happy too because she had a new friend. They sat together and ate their waffles. They talked about their favorite things and had a great time. When they were done, they said goodbye and promised to meet again soon. Lily went home with a big smile on her face, happy that she had met her new friend. Summary: Lily meets her new friend Timmy and they enjoy eating waffles together before saying goodbye.
  GRPO  (r=1.00): Once upon a time, there was a little girl named Lily. She had a new waffle for breakfast. She was very happy and ate it all up. After breakfast, she went outside to play. She saw a new friend in the park. Her new friend was a little boy named Max. They played together and had lots of fun. Lily was happy to meet Max and they became good friends. Summary: Lily makes a new friend named Max while playing in the park and they become good friends.
  PPO   (r=1.00): Once upon a time, there was a little girl named Lily. She loved waffles more than anything in the world. One day, she went to the park to play with her friends. While playing, she saw a new friend. It was a big, fluffy dog. Lily was so happy to meet the new friend. They played together all day long. When it was time to go home, Lily said goodbye to her new friend. She was sad to leave her new friend, but she knew she would see him again soon. Summary: Lily meets a new friend at the park and they play together all day, but Lily is sad to leave her new friend.

[3] Words: call, leak, foolish
  SFT   (r=0.67): Lily and Ben are playing in the garden. They have a bucket and a shovel. They want to make a big mess. "Look, I have a bucket!" Lily says. She runs to the bucket and starts to fill it with water. "Look, I have a bucket!" Ben says. He has a bucket too. He runs to the bucket and fill it with water. But he does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He does not see the leak. He is foolish.
  Base  (r=0.00): Water is not a toy, it is a living thing. It can be dangerous and hurt you. You have to respect the things that are in the garden. Do you understand?" Lily and Ben nod and say, "Yes, Mum. We understand. We're sorry. We won't do it again." Mum says, "Good. Now, let's go inside and have some cookies. And maybe we can find some real toys to play with in the garden. How about that?" Lily and Ben smile and say, "Yes, Mum. That sounds fun. Thank you, Mum." They hug Mum and say, "You're welcome, Mum. We love you." They go inside and have some cookies. They are happy and safe.
  DPO   (r=1.00): Lily and Ben are playing in the garden. They have a bucket and a shovel. They fill the bucket with water and make a big puddle. They splash and laugh. But then they hear a leak. It makes a big leak. "What is that?" Lily asks. "I don't know," Ben says. "Let's call the water." They run to the bucket and turn it. They hear a voice. It is Mom. She is not happy. "What are you doing?" Mom asks. "We are foolish water. We don't know. We need to tell Mom. She is not good. She is not good. She is foolish. She is not careful. She is not careful. She is foolish. She is not careful. She is foolish. She is not careful. She is foolish. She is not careful. She is foolish.
  GRPO  (r=1.00): Lily and Ben are playing in the garden. They have a bucket and a bucket. They fill it with water and splash it on the flowers. They are having fun. But then they see a leak in the bucket. It is a big leak. It makes a lot of water. Lily and Ben are foolish. They do not know what to do. They think they should call their mom. "Mom, can you hear us?" Lily asks. "Yes, I hear you. But don't be foolish. We need to tell her what to do. She is wet and cold." Mom says. She comes to the garden and sees the leak. She is not happy. She says they are foolish. She says they should not play with water. She says they are not careful. She says they are foolish. Lily and Ben are sad. They say sorry to their mom. They go inside. They are sorry. They will.
  PPO   (r=1.00): Lily and Ben are playing in the garden. They have a bucket and a shovel. They fill the bucket with water and make a big puddle. They splash and laugh. But then they see a leak in the bucket. It is a leak. It makes water everywhere. "Look, Ben, the leak is wet!" Lily says. She runs to get her bucket and starts to fill it with water. But Ben is not careful. He does not see the leak. He runs to the bucket and starts to splash water on it. "Stop, Ben, stop!" Mum calls. She comes to the rescue. She sees the leak and gets angry. She calls for Ben. "Lily, Ben, stop! You are being foolish. You should not play with water. You are making a big mess!" Ben stops and looks at Lily. She is sorry. She says sorry to Ben. They hug and go inside. They are happy.

[4] Words: point, stadium, fast
  SFT   (r=1.00): One day, a boy named Tom went to the stadium with his dad. They wanted to see a fast car race. Tom was very excited. He loved cars. Tom's dad said, "Tom, let's go fast to the stadium." They walked fast to the stadium. Tom saw many cars. He pointed at one car and said, "Look, Dad! A car!" His dad smiled and said, "Yes, Tom. That's a fast car. Let's watch it." They sat down and watched the fast car race. Tom was happy. He loved the stadium.
  Base  (r=0.00): Yarni Little Tom: Yarni Little Tom: Yarni Little Tom: Yarni Little Tom: Yarni Little Tom!
  DPO   (r=1.00): Once upon a time, there was a little girl named Lily. She loved to play with her toy cars and go fast around the house. One day, her dad took her to the stadium to watch a game. Lily was so excited to see all the cars go fast. As they were walking, Lily's dad pointed to the big stadium. "Look, Lily! That's a big stadium!" he said. "Yes, it's a big place where people go to watch cars." Lily looked up at her dad and said, "I want to go fast too!" Her dad smiled and said, "You can't go fast, but you can still have fun." They continued to watch the cars race and Lily pointed to the one that was going fast. "Look, Daddy! The one that goes fast!" she said. They both laughed and had a great time at the stadium.
  GRPO  (r=1.00): Once upon a time, there was a little girl named Lily. She loved to play with her toy cars and go fast around the house. One day, her dad took her to the stadium to watch a game. Lily was so excited to see all the cars go fast. As they were watching, Lily's dad pointed to a big red car that was going very fast. "Look, Lily! That car is going so fast!" he said. Lily looked up and saw the big red car. "Wow, that's so cool!" she said. "I want to go fast like that one." "I can't," said her dad. "But we can still have fun and watch the cars together." Lily smiled and pointed to the big red car. "I want to go fast like that one!" she said. They continued to watch the cars and had a great time.
  PPO   (r=1.00): Once upon a time, there was a little girl named Lily. She loved to play with her toy cars and watch them go fast. One day, Lily's dad took her to the stadium to watch a game. They were going to watch a big game of cars. Lily was so excited! As they were watching the game, Lily's dad pointed to the big red car that was going very fast. "Look, Lily! That car is going so fast!" he said. Lily smiled and said, "I want to go fast too!" They both laughed and continued to watch the game. After the game, Lily's dad said, "Let's go home now, Lily." Lily nodded and said, "Yes, let's go fast!" They walked home together, happy and excited to tell their friends about the fun day they had at the stadium.

[5] Words: slide, switch, expensive
  SFT   (r=0.67): Lila and Ben are friends. They like to play in the park. One day, they find a big slide. They want to go on it. They climb up the ladder and slide down. They laugh and have fun. But then, they see a switch on the slide. It is shiny and red. They want to touch it. They take turns to turn it. But then, they hear a loud noise. It is the car. It is broken. The car is angry. It does not want to be fixed. It does not want to be fixed. It does not want to be fixed. It does not want to be fixed. It does not like to be fixed. It does not like to be fixed. It does not like to be fixed. It does not like to be fixed. It does not like to be fixed. It does not like to be fixed.
  Base  (r=0.00): 
  DPO   (r=1.00): Lila and Ben are friends. They like to play in the park. They see a big slide. They want to go on the slide. They run to the slide. They go to the switch. They turn the switch. They go down the slide. They go fast. They laugh and clap. But then, they see a big car. The car is red and shiny. It has a switch. It is expensive. It is not for them. It is for the car. Lila and Ben are angry. They cry. They say sorry to Tom. They hug and hug. They say, "We are sorry. We will help you fix the car." They go to their room. They find a switch. They turn it on. They turn it on. They turn it on. They say, "Now we can play again." They go to their room. They are happy.
  GRPO  (r=1.00): Lila and Ben are friends. They like to play in the park. One day, they see a big slide. They want to go on it. They run to the slide. They climb up and sit on the slide. They go down fast. They laugh and have fun. But then, they see a switch on the slide. It is a big, expensive car. It is shiny and new. Lila and Ben are curious. They want to see the switch. They run to the switch. They turn the switch. But the switch is not good. It breaks. The car is broken. Lila and Ben are sad. They cry. They say sorry to Tom. They hug and hug. They say they are sorry. They promise to help Tom fix the switch. They promise to never use a switch again.
  PPO   (r=0.67): Lila and Ben are friends. They like to play in the park. One day, they see a big slide. They want to go on the slide. They run to the slide. They climb up the ladder. They go down the slide. They go so fast. They laugh and shout. But then, they see a switch on the slide. They have never seen a switch before. They want to see what it does. They go to the switch. They turn the switch. But the switch is not good. It breaks. The car is broken. Lila and Ben are angry. They cry. They say sorry to Tom. They hug and hug. They say they are sorry. They promise to help Tom fix the switch. They go to the slide. They have fun. They are happy.

[6] Words: call, tank, jolly
  SFT   (r=1.00): One day, a jolly girl named Lily went to play outside. She saw her friend Tom playing with a big tank. Lily wanted to play with the tank too. They decided to race their toys down the hill. As they raced, a big dog came and took Lily's tank. Lily was sad and cried. Tom saw Lily crying and said, "Don't worry, I will help you get your tank back." Tom called his dog and said, "Please give us my tank back." The dog came back and gave the tank back to Lily. Lily was happy and said, "Thank you, Tom!" They played with the jolly dog and had lots of fun.
  Base  (r=0.00): Lily and Tom: The dog is friendly and friendly. They make a new friend and share their toys. They are happy and grateful.
  DPO   (r=1.00): Once upon a time, there was a jolly little girl named Lily. She loved to play with her toy tank. One day, Lily's friend Tom came over to play. They decided to have a race to see who could get to the top of the hill first. As they were running, they saw a big dog coming towards them. Lily got scared and called out, "Tom, come back!" Tom stopped running and looked at Lily. He said, "Don't worry, I'll call for help." Tom called out, "Don't worry, I'll come with you." They ran as fast as they could and finally reached the top of the hill. They were so happy and jolly. They hugged each other and said, "Thank you for calling for help."
  GRPO  (r=1.00): Once upon a time, there was a jolly little girl named Lily. She loved to play with her toy tank. One day, Lily's friend Tom came over to play. They decided to have a race to see who could get to the top of the hill first. Lily said, "I'll call you, Tom!" Tom replied, "No way, I'll get to the top!" They started to race, but then a big dog came running towards them. The dog was very fast and he took Lily's tank! Lily was sad and said, "That's not fair! You should have called me." Tom felt bad and said, "I'm sorry, Lily. I will buy you a new tank." Lily forgave Tom and they continued to play with the jolly dog. They had a fun day together.
  PPO   (r=1.00): Once upon a time, there was a jolly little girl named Lily. She loved to play with her toys and her friends. One day, Lily and her friend Tom were playing with their toy cars. They were having so much fun that they didn't notice the big hill in their town. Suddenly, a big dog came running towards them. Lily and Tom were scared and didn't know what to do. But then, a kind man came and called for help. The man came and said, "Don't worry, I'll get your tank back." The man went to the store and bought a new tank for Lily and Tom. They were so happy and thanked the man. They played with the jolly dog and had a great time.

  
--- win-rate: DPO vs SFT (500 held-out prompts, K=4) ---
pass-rate   SFT 0.834   DPO 0.981
win-rate    0.744  (372 win / 9 loss / 119 tie)  |  >= SFT on 0.982
KL(DPO||SFT) 0.036 nats/tok (7.01 nats/seq)

--- regression: validation perplexity ---
SFT 8.838   DPO 9.093   delta +0.255 (+2.89%)  [ok]

--- qualitative: 6 greedy SFT vs DPO generations ---
downloading TinyStories-Instruct-train.txt ...
downloading TinyStories-Instruct-valid.txt ...

[1] Words: escape, war, tall
  SFT (r=0.67): Tom and Lily are friends. They like to play in the forest. They see a big house. It is a house with a tall door. They want to go inside. But the door is closed. They are scared. They want to escape. They look for a way out. They see a big tree. The tree is green and has many leaves. They think it is a good place to live. They go inside the house. They are happy. They are safe. They say, "Hello, tree house. We are your friend." They hug the tree. They are happy. They are not scared anymore. They are friends. They like the tree house. They like the tree house. They like the tree house. They are happy. They are not alone. They are friends.
  DPO (r=1.00): Tom and Lily are friends. They like to play in the forest. They see a big house in the distance. They want to go inside and see what is there. "Let's go in and see!" Tom says. "OK, but be careful. The house is very tall and scary. We have to escape," Lily says. They run to the house and open the door. They see a big war. The war is loud and scary. Tom and Lily are scared. They want to go back to the house. They see a big tree in the forest. The tree is very tall and has many branches. They want to climb it. They think they are brave. "Let's go in and see what is there," Tom says. "Maybe we can find a way to escape." They run to the tree and climb it. They see a hole in the tree. They see a hole and a hole. They are happy. They are safe. They are brave.

[2] Words: meet, waffle, new
  SFT (r=1.00): Once upon a time, there was a little girl named Lily. She loved waffles more than anything in the world. One day, she went to the park with her mom and dad. While they were playing, Lily saw a new friend. It was a little puppy! Lily was so happy to meet the new friend. She played with the puppy all day long. They ran and jumped and had so much fun. When it was time to go home, Lily said goodbye to her new friend and promised to come back and play again. Summary: Lily meets a new friend at the park and they have fun playing together before saying goodbye.
  DPO (r=1.00): Once upon a time, there was a new waffle. It was big and red. The new waffle was very happy. It wanted to meet new friends. One day, the new waffle went to a new place. It was a big park with lots of trees and grass. The new waffle was excited to meet new friends. It walked and walked until it found a big tree. The new waffle met a little bird. The bird said, "Hi, I am new here. Do you want to be my friend?" The new waffle was happy and said, "Yes, I want to be your friend!" They played together in the park. The new waffle was very happy to meet a new friend. They had a lot of fun together. And they became best friends forever. Summary: A new waffle goes to a park and meets a little bird who becomes its friend, and they become best friends forever.

[3] Words: call, leak, foolish
  SFT (r=0.67): Lily and Ben are playing in the garden. They have a bucket and a shovel. They want to make a big splash. They fill the bucket with water from the hose. They are having fun. But then they see a leak in the bucket. The water is coming out of the bucket. It is a foolish leak. It makes a big mess. Lily and Ben are scared. They run to their Mum. She is not happy. She says, "What are you doing? You are foolish! You should not play with water in the bucket. It is not safe." Lily and Ben are sad. They cry. They say, "We are sorry, Mum. We will not play with water in the bucket again." They hug their Mum. They are sorry. They go to their room. They are still sad. They want to play with water. They want to make a new splash.
  DPO (r=1.00): Lily and Ben are playing in the garden. They have a bucket and a bucket. They fill the bucket with water and splash it on the flowers. They are having fun. But then they see a leak in the bucket. The water is coming out of the bucket. It is a foolish leak. It makes a big mess. Lily and Ben are scared. They call their mom. "Mom, what is the leak?" Lily asks. "I don't know, sweetie. It is a leak. It makes water. It makes water for the flowers and the birds. But we don't want water. It is foolish. We should not play with water. It is not safe. We should call for help. We need to be careful. We should not play with water. We should call for help. We should have called for help. We should have called for help. We are in trouble. We are in trouble.

[4] Words: point, stadium, fast
  SFT (r=1.00): One day, a little girl named Lily went to the stadium with her dad. They were going to watch a game. Lily was very excited because she loved to watch the cars go fast. She saw a man point at a big red car. "Look, Dad! That car is so fast!" said Lily. "Yes, it is. Let's go see it," said her dad. They walked to the big red car. The man in the car was there. He was pointing at the cars. "Look, Lily! That car is so fast!" said the man. Lily and her dad watched the cars go fast. They laughed and clapped their hands. "That was so much fun!" said Lily. "Yes, it was," said her dad. They had a great time at the stadium.
  DPO (r=1.00): Once upon a time, there was a little girl named Lily. She loved to go to the stadium with her dad. They would watch the cars go fast and then they would point at each other to see who was the fastest. One day, Lily's dad took her to the stadium to watch a game. They were going to watch a game of tag. Lily was so excited! She pointed at the other cars and said, "Look, Daddy! That's so cool!" As they were watching the game, Lily's dad noticed that the stadium was getting dark. He said, "Lily, it's time to go home now. We have to go now." Lily was sad to leave the stadium, but she knew they had to go home. She said, "Daddy, can we come back to the stadium soon?" Her dad smiled and said, "Of course, Lily. We can come back anytime you want." And they went home, happy and tired from their fun day at the stadium

[5] Words: slide, switch, expensive
  SFT (r=0.33): Lila and Ben are friends. They like to play in the park. One day, they see a big slide. They want to go on the slide. They run to the slide. They go down fast. They go so fast that they hit the wall. They hit the wall. They hit the wall. They were sad. They cry. They say sorry to each other. They hug and say sorry. They promise to help Tom. They go back to the slide. They go down the slide. They have fun. They are happy. They learn that it is not good to play with things that don't belong to them. They are friends.
  DPO (r=1.00): Lila and Ben are friends. They like to play in the park. One day, they see a big slide. They want to go on the slide. They run to the slide. They go to the switch. They go to the switch. They turn the switch. The slide is expensive. They do not know what is happening. They do not know. They do not know what to do. They are scared. They do not want to go on the slide. They do not want to be alone. They do not want to be friends. They do not want to be friends. They do not know what to do. They do not know how to fix the switch. They do not know how to fix it. They do not know how. They are angry. They do not like the switch. They do not like the slide. They are sad. They do not like the expensive slide.

[6] Words: call, tank, jolly
  SFT (r=1.00): Once upon a time, there was a jolly little girl named Lily. She loved to play with her toys and her friends. One day, Lily and her friend Tom were playing with their toy cars. They were having so much fun that they didn't notice the big hill in their town. Suddenly, a big dog came running towards them. Lily and Tom were scared and didn't know what to do. But then, a kind man came and called for help. The man came and said, "Don't worry, I'll get your tank back." The man went to the store and bought a new tank for Lily and Tom. They were so happy and thanked the man. They played with the jolly dog and had a great time.
  DPO (r=1.00): Once upon a time, there was a jolly little girl named Lily. She loved to play with her toy tank. One day, Lily's friend Tom came over to play. Tom wanted to play with Lily's tank, but Lily didn't want to share. Tom said, "Can I play with your tank, please?" Lily replied, "No, it's mine. I don't want to share." Tom got upset and called out, "Why won't you let me play with your tank?" Lily didn't want to fight, so she said, "Okay, I'll let you play with my tank." Tom was happy and they played together. They had so much fun that they forgot about the tank. Suddenly, a big dog came running towards them. The dog was very mean and wanted to take Lily's
