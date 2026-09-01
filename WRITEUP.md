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
coin flip vs SFT), KL 0.002 (it barely moved).

**The training curve shows exactly why.** The critic's value loss collapses from
1.13 to ~0 within the first ~30 steps and flatlines — it trivially learned to
predict the near-constant reward (~0.8 on almost every rollout, because the
verifiable reward is low-variance). Once the value baseline matches the reward,
the **GAE advantages vanish**: `advantage = return − value ≈ 0`. With no advantage
signal the actor gets essentially no gradient and stalls — actor loss hovers at
~−0.002 and reward stays flat and noisy for the remaining 370 steps. That's the
mechanism behind the roadmap's "PPO is heavy and finicky" warning, made concrete:
a critic that fits a flat reward starves its own actor. GRPO sidesteps it — its
group-relative baseline normalizes *within* each prompt's samples, so even a
low-variance reward yields usable advantages. This is the empirical case for
preferring GRPO on verifiable rewards.

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
  no movement (KL 0.002). The heaviest method delivered the least in this budget,
  and its training curve shows why: the critic collapsed to a constant baseline,
  starving the actor of advantage signal (mechanism in Step 5 above).

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

target 13,719,552 params | draft 1,420,800 params | gamma=4 on cuda

--- prompt: 'Once upon a time,' ---
speculative: Once upon a time, there was a little girl named Lily. She was very hungry and wanted to eat some apples. But her mom said no, because she was already out of food. Lily was very sad and cried a lot. She didn't know what to do. Suddenly, she saw a big tree with apples on it. She tried to climb it, but it was too hard. She started to cry even more. Her mom heard her crying and came to see what was wrong. She saw the little girl crying and asked what was wrong. Lily told her about the apples she was hungry and wanted to eat some apples. Her mom told her to close her eyes and wait for them to come out of the tree. Lily did as her mom said and waited. When they came out, she found a big pile of apples on the ground. She was so happy and ate a lot of apples. From that day on, she knew that if she was in trouble, she would ask for help and try to
  vanilla 875 ms (200 tok) | speculative 1219 ms (200 tok) | 2.00 tok/target-call | accept-rate 0.25 | speedup 0.72x

--- prompt: 'One day, a little girl named Lily' ---
speculative: One day, a little girl named Lily was excited to see a new toy. She asked her mom for a new toy. Her mom said yes, and Lily was happy. She went to bed with a smile on her new toy. She dreamed of all the fun things she would do the next day. Summary: Lily was excited to see a new toy and asked her mom for a new one.<|endoftext|>
  vanilla 647 ms (200 tok) | speculative 537 ms (75 tok) | 1.63 tok/target-call | accept-rate 0.16 | speedup 1.21x

--- prompt: 'Tom and Sara went to the park and' ---
speculative: Tom and Sara went to the park and saw a big dog with a scarf around. The boy who lost his owner's scarf and ran away. The boy was sad to see his owner, but he knew he was a good boy and that he would never forget this time. The boy went to the park with his family and the scarf on his scarf. He saw a lady who was looking for her. He knew he would never forget this time. He put the scarf around his neck and went to the lady. She was very happy to see him. She gave him a big hug and said he was a good boy. The boy went to the park with his family and the scarf on his neck. He saw a lady who was looking for her lost scarf. The boy gave her the scarf and she hugged him. The lady was very grateful. She said she was sorry for calling the boy for his owner. The boy was happy to see his owner and felt good inside. He knew he would never forget this time. Summ
  vanilla 646 ms (200 tok) | speculative 1362 ms (200 tok) | 1.69 tok/target-call | accept-rate 0.17 | speedup 0.47x

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

stage	curves.png shows
GRPO	reward 0.80→0.91 ↑, KL 0→0.016 ↑ (controlled), groups active
PPO	reward flat/noisy, value-loss collapse (the underperformance smoking gun), tiny actor loss
DPO	loss ↓, margin ↑ to +4.4, accuracy ↑ to ~0.8


| 29.9 min, ~0.0 left

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