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

We use decoder only because story generation isn't sequence to sequence generation, 
its continuation. There's no clean input -> output, they're the same stream of text.

---

# TODO

1. find a home for make_preferences and reward.py
2. README.MD

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

### 3. Preprocess / pack ✓
Encode the whole corpus once to token IDs; write a flat `uint16` memmap per
split (8k vocab fits `uint16`), inserting the EOS id between stories. nanoGPT
pattern: pre-tokenize once, then sample random windows at train time.
- **Also check the token-length distribution here** — settles whether 256 is
  right or should be 192 / 384.

Deliverable: `train.bin`, `val.bin`, `meta` (vocab size, EOS id).

train: 530,386,257 tokens, 2,717,700 stories -> data\packed\train.bin
valid: 5,355,994 tokens, 27,631 stories -> data\packed\val.bin

=== train story token-length distribution ===
p50:    173
p90:    263
p95:    364
p99:    560
max:   1646   mean:  194.2
fraction of stories <= 256 tokens: 0.893

256 is a good number for context length.


### 4. Model ✓
Decoder-only GPT. Prune the seq2seq transformer: keep `MultiHeadAttention`
(with the causal mask from the old decoder self-attn), **drop cross-attention
and the entire encoder**.
- token embedding + positional (learned absolute is fine; RoPE optional)
- N pre-norm causal blocks (attn + MLP)
- final norm + LM head, **weight-tied** to the embedding
- lots of data vs training size, so overfitting is unlikely
- 384/6 = 64, 64 is the universal choice for the attention heads

Target size (~10M params): `d_model≈384`, `n_layer≈6`, `n_head≈6`, `ctx≈256`.

### 5. Training loop ✓
Sample random windows from the memmap.
- AdamW (betas 0.9/0.95, wd 0.1), cosine decay + linear warmup, grad clip 1.0
- grad accumulation for effective batch; bf16 autocast on Ampere+
- resumable checkpoints (save scheduler `state_dict` too), periodic val eval

At batch 64 x accum 8 (gradient accumulation, it simiulates a large batch size when the GPU can't fit one, 
run several smaller microbatches and sum the gradients and do optimizer step after accum) 
x ctx (context length) 256 = 131 tokens/step, one epoch over 530M tokens is 4046 steps.
The 10,000 MAX_STEPS is about 2.5 epochs

### 6. Sampling ✓
Autoregressive decode with temperature + top-k/top-p. Sampling, not beam
search — this is open-ended generation. Prompt in → story out.

Dont need epochs because we sample by concatenating every story into only long 
array and drawing a batch of random windows (sampling with replacement) from it, 
so we only need MAX_STEPS. Since there's replacement, there's no epochs.

We follow Kaparthy's minGPT https://github.com/karpathy/mingpt. 

This makes it stateless and resuming is trivial and decoupled from dataset size.
This makes for a simpler loader and doesn't matter for our use case.

This also helps with some diversity since tokens get seen at differen positions.
Where coverage matters and we can't afford to miss any data, we might want coverage guarantee.

### 7. Evaluation ✓
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

### 8. SFT (instruction format) ✓
Teach the base model a lightweight instruction schema. Use
`TinyStories-Instruct` — stories prefaced with constraints (required words,
summary, feature flags: dialogue / bad ending / moral / plot twist).
Fine-tune the base checkpoint on `(instruction → story)` pairs; same
next-token loss, masked to the response span.

Does two jobs: a prompt-following model, **and** the reference policy every
later RL/DPO stage regularizes against. Deliverable: `sft.pt`.

### SFT — teaching the instruction schema
- Goal: base LM → follows TinyStories-Instruct format (Summary/Words/
  Sentence/Features → Story). Does double duty: the deliverable model AND
  the frozen reference every later stage's KL is measured against.
- Method: same next-token loss as pretraining, but mask the prompt
  (labels = -1 up to "Story:") so only the response + EOS trains. Padded
  DataLoader over the finite instruct set; init from pretrained best.pt.
- Data: filter examples > context (drop, never truncate — model only sees
  complete stories). prompt = text incl. "Story:", response = story + EOS.
- Training: 1 epoch (~50k steps), LR 3e-4 (< pretraining 6e-4), cosine +
  warmup, early-stop on masked val loss (best 1.176).
- Results — the payoff is in the eval, not the loss number:
    · word-inclusion pass-rate: Base 0.08 → SFT 0.85  (SFT is what makes
      instruction-following possible at all)
    · LM trade-off: plain-story ppl 3.67 → 8.84 (specializing costs some
      general LM quality — a real, quantified cost, not a footnote)
    · before/after samples: base ignores "Words:", SFT includes them
- Key insight: the Base→SFT jump is the single biggest capability gain in
  the whole post-training stack; DPO/GRPO/PPO are refinements on top.


### 9. Reward / preference design ✓
The pivot everything downstream depends on. Three sources, cheapest first:
- **Verifiable / programmatic** — did the story contain the required words?
  match the summary length? include the feature? Zero models, zero labels.
  Cleanest fit for GRPO; the recommended starting point.
- **LLM-as-judge (RLAIF)** — score the rubric with a bigger model. Noisier,
  slower, captures quality the checks can't. Skip this to save costs - have done
  a lot of LLM judge work in evals.
- **Trained reward model** — only for the classic PPO/RLHF path (step 11).

Remove the incentive to repeat (set) and other reward hackable methods.

### 10. DPO
Do this **before** any online RL. Needs preference pairs `(chosen, rejected)`:
generate two completions per prompt, rank by verifiable score or judge.
No reward model, no sampling loop, no critic — a classification-style loss
against the frozen SFT reference, KL baked into the objective. Most stable,
easiest to debug. Deliverable: `dpo.pt`.

**Preference pairs** (`make_preferences.py`, finishes step 9): load the SFT
checkpoint and, for each instruct prompt with a `Words:` field, sample K
completions at temperature > 0, score each with `verifiable_reward`, and emit
`(prompt, chosen, rejected)` on a reward spread (ties skipped — no signal).
Saved to `data/dpo/pairs.jsonl`.

**Training** (`dpo_train.py`): policy init from SFT, a frozen reference clone of
SFT, a `sequence_logprob` helper (summed response-token log-probs, reusing the
SFT prompt masking), the `-log σ(β·[...])` objective, and checkpoints to
`dpo_checkpoints/`. Reuses the optimizer / scheduler / checkpoint machinery.
Diagnostics: reward accuracy (policy prefers chosen) and the reward margin.

Make sure we can view the output sentences (like in generate_story.py) and the overall eval of it.

Need to reduce gameability, and include a verifiable reward metric for fluency.


The objective is
    L = -log sigmoid( beta * [ (logp_pi(y_w) - logp_ref(y_w))
                             - (logp_pi(y_l) - logp_ref(y_l)) ] )
where ``logp(y)`` is the summed log-prob of the response tokens (``sequence_logprob``),


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

### 11. Reward model *(only for the RLHF/PPO path)*
A scalar reward head on the preference pairs via the Bradley-Terry loss.
**Skip entirely** for verifiable-reward GRPO or DPO — both bypass it. Build
only if you want the classic three-stage RLHF stack.

Is this the reward model for RLHF + PPO e.g. an LLM judge which outputs a score?

### 12. PPO / GRPO
Online RL — the hardest stage.

We can't RL our way to behaviours the policy has zero proablity of producing.
It only sharpens the existing distribution (which is why we SFT first - in this case on instruct).

- **PPO** — classic RLHF workhorse but heavy: policy + value/critic + reward
  model + frozen reference all resident, and finicky to stabilize.
- **GRPO** — drops the value network; samples a *group* of completions per
  prompt and normalizes each reward against the group mean/std for the
  advantage. Lighter (no critic), pairs perfectly with verifiable rewards.

Apply a KL penalty on the objective, and GAE for the advantage. 

Advantages come from GAE over the valuebaseline, then are normalized across the batch's response tokens.
    The reward model produces one scalar for the whole response and applies 
    that scalar to the very last response token, whilst the KL penalty applies 
    to every generated token.GAE reshapes how we estimate the advantage, normalized over the response tokens.

**Recommended:** GRPO with verifiable rewards. Treat PPO as optional "build the
full classic stack for the education."

GRPO vs PPO, side by side
              GRPO	                    PPO
baseline	    group mean/std	          learned critic (value head)
extra model	  none	                    critic backbone (trained)
advantage	    (r−mean)/std	            GAE over value fn
loss	        clipped surrogate + KL	  clipped actor + clipped value
Both consume the same shaped reward and frozen SFT reference, so you can grade them head-to-head.

That reframes your repetition question usefully: the base sets the diversity ceiling (0.92), and the real target for DPO/GRPO/PPO is getting distinct2 back up toward it while keeping pass-rate high. 

Instruction following decreases the diversity from free-flow story genereation.

PPO
Everything GRPO drops, added back. Each step:

1. **Rollout.** Sample prompts, one completion each from the policy; record the
   old per-token log-probs, the reference log-probs, and the critic's per-token
   values.
2. **Reward.** A per-token KL penalty `-beta*(logp_policy - logp_ref)` at every
   response token, plus the scalar shaped reward added at the final (EOS) token --
   the standard RLHF token-reward shaping.
3. **GAE.** Generalized Advantage Estimation over the response tokens using the
   critic's value baseline (this is what GRPO replaces with a group mean).
4. **Update.** `inner_epochs` passes of a clipped actor surrogate + a clipped
   value loss, actor and critic on separate optimizers/LRs.


GRPO
1. **Rollout.** Sample a batch of instruct prompts; for each, sample a *group* of
   `G` completions from the current policy.
2. **Reward.** Score every completion with the shaped reward (verifiable word
   inclusion minus a repetition penalty -- `src.reward.shaped_reward`).
3. **Advantage.** Normalize each reward against its group: `A = (r - mean) / std`.
   No value network -- the group mean is the baseline. A group whose completions
   all score the same has zero advantage and contributes no gradient.
4. **Update.** A clipped PPO surrogate on the response tokens, with a per-token
   KL leash to the frozen reference (the SFT model), so the policy improves reward
   without drifting into the reward-hacking degeneracy step 13 catches.


### 13. Post-training eval
Three things, not one number:
- **Win-rate** vs the SFT baseline (verifiable pass-rate or judge preference)
- **KL from the reference** — catch over-optimization
- **Regression check** — base capability (perplexity, grammar) didn't collapse
- **Spot checks** - eyeball outputs from Base, SFT, DPO, PPO, GRPO
- **Speculative Decoding** - implement and try, its how all the labs are speeding up token throughput today 

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

