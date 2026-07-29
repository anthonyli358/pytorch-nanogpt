# pytorch-nanogpt
Pytorch implementation of a GPT style decoder

## Usage

To use CUDA 12.6 for torch, we add the pytorch index as [described in the documentation](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) to [pyproject.toml](pyproject.toml). 

# TinyStories SLM — Project Roadmap

Building a small language model end-to-end from the HF TinyStories corpus:
pretraining a decoder-only GPT, then exploring the post-training stack
(SFT → DPO → GRPO/PPO).

**Guiding constraint:** on a ~10M-param model the quality ceiling is low, so
post-training is about learning the *machinery*, not chasing big gains.
TinyStories earns its place here because the instruction variant gives
**verifiable rewards for free** — an ideal sandbox for GRPO.

Steps are ordered by dependency. Two numbers ripple through everything from
step 3 onward and should be pinned in config early: **vocab size (8k)** and
**context length (256)**.

---

## Part I — Pretraining

### 1. Data download ✓
Raw `.txt` per split from the HF Hub, `<|endoftext|>` separators intact.
Cache-backed, so re-runs are free. Deliverable: `data/raw/*.txt`.

Output: `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt`

### 2. Tokenizer ✓
Train a SentencePiece BPE on the raw train text; wrap load/encode/decode in a
thin `Tokenizer` class.
- vocab **8k**, `model_type=bpe`, `byte_fallback=True` (never emit `<unk>`)
- `<|endoftext|>` as a whole `user_defined_symbol`, doubling as BOS/EOS
- subsample lines for training (no need for all ~2GB)

A SentencePiece BPE tokenizer trained on the TinyStories train split. 

- `vocab_size=8000` subword pieces is sufficient for the corpus.
- **Sampling:** merges are learned from 2M lines randomly sampled
  (`shuffle_input_sentence=True`) out of the full 14.6M. BPE merge frequencies
  saturate quickly on a corpus this small and repetitive, so using more lines
  doesn't meaningfully change the vocabulary.
- **`max_sentence_length=8192`:** raised from the 4096-byte default so no full
  stories are dropped during training. Only affects which lines contribute to
  learning merges — `encode()` is never length-limited — and the extra memory
  cost is trivial.
- **`byte_fallback=True`:** unseen characters fall back to bytes, so the
  tokenizer never emits `<unk>`.
- **Special tokens:** `<|endoftext|>` is registered as a single user-defined
  symbol (never split by BPE) and serves as the document / EOS boundary. Native
  BOS/EOS are disabled — the model uses the `<|endoftext|>` id instead.

Output: `spm.model` + `spm.vocab`.

### 3. Preprocess / pack *(current)*
Encode the whole corpus once to token IDs; write a flat `uint16` memmap per
split (8k vocab fits `uint16`), inserting the EOS id between stories. nanoGPT
pattern: pre-tokenize once, then sample random windows at train time.
- **Also check the token-length distribution here** — settles whether 256 is
  right or should be 192 / 384.

Deliverable: `train.bin`, `val.bin`, `meta` (vocab size, EOS id).

### 4. Model
Decoder-only GPT. Prune the seq2seq transformer: keep `MultiHeadAttention`
(with the causal mask from the old decoder self-attn), **drop cross-attention
and the entire encoder**.
- token embedding + positional (learned absolute is fine; RoPE optional)
- N pre-norm causal blocks (attn + MLP)
- final norm + LM head, **weight-tied** to the embedding

Target size (~10M params): `d_model≈384`, `n_layer≈6`, `n_head≈6`, `ctx≈256`.

### 5. Training loop
Sample random windows from the memmap.
- AdamW (betas 0.9/0.95, wd 0.1), cosine decay + linear warmup, grad clip 1.0
- grad accumulation for effective batch; bf16 autocast on Ampere+
- resumable checkpoints (save scheduler `state_dict` too), periodic val eval

### 6. Sampling
Autoregressive decode with temperature + top-k/top-p. Sampling, not beam
search — this is open-ended generation. Prompt in → story out.

### 7. Evaluation
- **Val perplexity** — day-to-day workhorse metric.
- **TinyStories rubric** (grammar / consistency / creativity, graded by a
  larger model) — the gold qualitative eval; wire up once samples are worth
  grading. Defer; perplexity + eyeballing gets most of the early signal.

---

## Part II — Post-training

Ordered offline/simple → online/hard, each stage independently useful.

**Acronym map (read once):** RLHF is not a separate method — it's the umbrella
for SFT → reward model → PPO. DPO and PPO/GRPO are alternative routes from the
*same* preference data (offline-and-simple vs online-and-stronger). GRPO is
just PPO minus the critic. Learning-optimal path:
**SFT → DPO → GRPO(verifiable)**, adding the reward model + PPO last only for
the full classic stack.

### 8. SFT (instruction format)
Teach the base model a lightweight instruction schema. Use
`TinyStories-Instruct` — stories prefaced with constraints (required words,
summary, feature flags: dialogue / bad ending / moral / plot twist).
Fine-tune the base checkpoint on `(instruction → story)` pairs; same
next-token loss, masked to the response span.

Does two jobs: a prompt-following model, **and** the reference policy every
later RL/DPO stage regularizes against. Deliverable: `sft.pt`.

### 9. Reward / preference design
The pivot everything downstream depends on. Three sources, cheapest first:
- **Verifiable / programmatic** — did the story contain the required words?
  match the summary length? include the feature? Zero models, zero labels.
  Cleanest fit for GRPO; the recommended starting point.
- **LLM-as-judge (RLAIF)** — score the rubric with a bigger model. Noisier,
  slower, captures quality the checks can't.
- **Trained reward model** — only for the classic PPO/RLHF path (step 11).

### 10. DPO
Do this **before** any online RL. Needs preference pairs `(chosen, rejected)`:
generate two completions per prompt, rank by verifiable score or judge.
No reward model, no sampling loop, no critic — a classification-style loss
against the frozen SFT reference, KL baked into the objective. Most stable,
easiest to debug. Deliverable: `dpo.pt`.

### 11. Reward model *(only for the RLHF/PPO path)*
A scalar reward head on the preference pairs via the Bradley-Terry loss.
**Skip entirely** for verifiable-reward GRPO or DPO — both bypass it. Build
only if you want the classic three-stage RLHF stack.

### 12. PPO / GRPO
Online RL — the hardest stage.
- **PPO** — classic RLHF workhorse but heavy: policy + value/critic + reward
  model + frozen reference all resident, and finicky to stabilize.
- **GRPO** — drops the value network; samples a *group* of completions per
  prompt and normalizes each reward against the group mean/std for the
  advantage. Lighter (no critic), pairs perfectly with verifiable rewards.

**Recommended:** GRPO with verifiable rewards. Treat PPO as optional "build the
full classic stack for the education."

### 13. Post-training eval
Three things, not one number:
- **Win-rate** vs the SFT baseline (verifiable pass-rate or judge preference)
- **KL from the reference** — catch over-optimization
- **Regression check** — base capability (perplexity, grammar) didn't collapse

Watch for **reward hacking**: an "include these words" reward will teach the
model to cram words in ungrammatically — pass-rate climbs while stories get
worse. Exactly why a quality metric rides alongside the reward.

---

## Config decisions to pin early
| Decision | Value | Ripples into |
|---|---|---|
| Vocab size | 8k | `uint16` packing, embedding budget |
| Context length | 256 | packing, positional embedding size |
| Reward source | verifiable vs judge | whether step 11 is ever built |
| Instruction schema | (set in step 8) | every later generate/score stage |
