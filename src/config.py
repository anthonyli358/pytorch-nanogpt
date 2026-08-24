from dataclasses import dataclass

# ----- Data Download ----
REPO_ID = "roneneldan/TinyStories"
DATA_DIR = "data/raw"
EOS_MARKER = "<|endoftext|>"  # separate for stories in the dataset
FILE_SETS = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}

# ----- Tokenizer ----
TOKENIZER_DIR = "data/tokenizer"
MODEL_PREFIX = "spm"  # -> data/tokenizer/spm.model + spm.vocab
VOCAB_SIZE = 8000
CHARACTER_COVERAGE = 1.0  # clean English; byte_fallback covers the rest
INPUT_SENTENCE_SIZE = 2_000_000  # subsample lines for training (no need for all ~2GB)
MAX_SENTENCE_LENGTH = 8192  # bytes; default 8192 skips the longest stories (4929)

# ----- Packing ----
PACKED_DIR = "data/packed"
PACKED_FILES = {"train": "train.bin", "valid": "val.bin"}
META_FILE = "meta.json"
PACK_BATCH_STORIES = 1024  # stories per batch-encode call
PACK_LOG_EVERY = 200_000  # print packing progress every N stories
CONTEXT_LEN = 256  # planned model context; used to report coverage


# ----- Model ----
@dataclass
class GPTConfig:
    """Model architecture."""

    vocab_size: int = VOCAB_SIZE
    block_size: int = CONTEXT_LEN
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0  # data-rich (~530M tokens / ~14M params); low overfit risk


# ----- Training ----
SEED = 1337
DEVICE = None  # None -> auto (cuda / cpu)
DTYPE = "bfloat16"  # bfloat16 | float16 | float32 (auto-downgraded if unsupported)
COMPILE = False  # torch.compile the model (big speedup on recent GPUs)

BATCH_SIZE = 64  # sequences per micro-step
GRAD_ACCUM_STEPS = 8  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
MAX_STEPS = 10_000  # optimizer steps (~131k tokens/step; 1 epoch ~= 4k steps)
WARMUP_STEPS = 200
LR = 6e-4  # peak LR
MIN_LR = 6e-5  # cosine floor (~ LR / 10)
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

EVAL_INTERVAL = 500  # steps between val evals + checkpoints
EVAL_ITERS = 100  # batches averaged per eval
LOG_INTERVAL = 20  # steps between train-loss logs

CKPT_DIR = "checkpoints/base"  # pretrained (base) runs; post-training stages sit beside it
RESUME = True                # resume training from a prior run's last.pt
RESUME_FROM = None            # None -> latest run dir; or a run dir / .pt path
CKPT_RUN = None               # sample/evaluate: None -> latest run's best.pt; or a run dir / .pt path

# ----- Sampling ----
SAMPLE_PROMPTS = [
    "Once upon a time,",
    "One day, a little girl named Lily",
    "Tom and Sara went to the park and",
]
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 200
TOP_P = 0.95

# ----- Speculative decoding (Part I, step 8) -----
# A small/fast DRAFT proposes SPEC_GAMMA tokens; the TARGET verifies them in a
# single forward pass and accepts the longest prefix consistent with its own
# distribution (rejection-sampling correction on the first mismatch). Output is
# distributed exactly as sampling from the target alone, but several tokens land
# per target forward -- the throughput win, gated by the draft's acceptance rate.
SPEC_TARGET_DIR = "checkpoints/sft"  # target = the model whose distribution we want to preserve
SPEC_TARGET_RUN = None               # None -> latest run under SPEC_TARGET_DIR
SPEC_DRAFT_DIR = "checkpoints/draft"  # a smaller GPT trained on the same corpus (cheaper per token)
SPEC_DRAFT_RUN = None
SPEC_GAMMA = 4                       # draft tokens proposed per verification round
SPEC_MAX_NEW_TOKENS = 200

# Draft-model training: a small GPT on the same corpus/objective, written to
# SPEC_DRAFT_DIR. It only needs to propose tokens the target usually accepts, so
# it can be crude and cheap -- a better draft raises the acceptance rate (speedup).
DRAFT_N_LAYER = 2
DRAFT_N_HEAD = 2                     # d_model 128 / 2 = 64 head dim
DRAFT_D_MODEL = 128
DRAFT_MAX_STEPS = 3000               # ~0.75 epoch; enough for a usable draft
DRAFT_LR = 6e-4
DRAFT_MIN_LR = 6e-5
DRAFT_WARMUP_STEPS = 100

# ----- Evaluation ----
# ----- Evaluation ----
EVAL_BATCH_SIZE = 64          # batch size for the deterministic perplexity sweep
EVAL_MAX_BATCHES = 200        # cap batches per split (None = full sweep); spread across the file
 
# ----- SFT ----
INSTRUCT_REPO_ID = "roneneldan/TinyStoriesInstruct"   # separate repo from the base data
INSTRUCT_FILE_SETS = {
    "train": "TinyStories-Instruct-train.txt",
    "valid": "TinyStories-Instruct-valid.txt",
}
STORY_MARKER = "Story:"       # up to & incl. this = prompt (masked); after = response
SFT_MAX_LEN = None            # cap example length in tokens; None -> model block_size
MAX_SFT_EXAMPLES = None       # cap examples per split (None = all)
SFT_BUILD_LOG_EVERY = 500_000 # progress print every N records while tokenizing the split

 
SFT_INIT_RUN = None           # pretrained run to fine-tune from (None = latest)
SFT_CKPT_DIR = "checkpoints/sft"   # under the shared checkpoints/ parent
SFT_PATIENCE = 2              # early-stop after this many epochs without val improvement
SFT_BATCH_SIZE = 32
SFT_EPOCHS = 1
SFT_LR = 3e-4                 # lower than pretraining peak (fine-tuning)
SFT_MIN_LR = 3e-5
SFT_WARMUP_STEPS = 100
SFT_WEIGHT_DECAY = 0.1
SFT_GRAD_CLIP = 1.0
SFT_EVAL_INTERVAL = 500
SFT_LOG_INTERVAL = 50

# ----- Reward shaping (repetition penalty; used by preference selection + GRPO) -----
REP_NGRAM = 2                       # n-gram size for the distinct-n-gram diversity metric
REP_WEIGHT = 0.5                    # weight of the repetition penalty subtracted from the verifiable reward

# ----- Preference generation (step 9 -> DPO pairs) -----
# Sample K completions per instruct prompt, score each with the verifiable
# reward, and emit (prompt, chosen, rejected) when there is a reward spread.
DPO_DATA_DIR = "data/dpo"
PAIRS_FILE = "pairs.jsonl"          # -> data/dpo/pairs.jsonl
PREF_INIT_RUN = None                # SFT run to sample from (None = latest under SFT_CKPT_DIR)
PREF_NUM_PROMPTS = 5000             # instruct-train prompts (with a Words: field) to sample from
PREF_SAMPLES_PER_PROMPT = 4         # K completions per prompt
PREF_TEMPERATURE = 1.0              # > 0 for diversity across the K samples
PREF_TOP_K = 200
PREF_TOP_P = 0.95
PREF_MAX_NEW_TOKENS = 256           # upper cap on new tokens; actual = min(this, block_size - prompt_len)
PREF_MIN_NEW_TOKENS = 48            # skip a prompt if the remaining context leaves less room than this
PREF_LOG_EVERY = 200               # progress print every N prompts processed
PREF_SEED = 1337

# ----- DPO (step 10) -----
DPO_CKPT_DIR = "checkpoints/dpo"    # under the shared checkpoints/ parent
DPO_INIT_RUN = None                 # SFT run to init policy + frozen reference from (None = latest)
DPO_MAX_LEN = None                  # cap example length in tokens; None -> model block_size
DPO_VAL_FRACTION = 0.05             # fraction of pairs held out for val loss / reward accuracy
DPO_BETA = 0.3                      # KL strength in the DPO objective (higher = stay closer to ref)
DPO_BATCH_SIZE = 16                 # pairs per step (2x sequences forwarded: chosen + rejected)
DPO_EPOCHS = 1                      # 2+ epochs over-optimized (+42% val ppl, reward hacking); 1 is the safe default
DPO_LR = 1e-5                       # DPO is sensitive; well below the SFT peak
DPO_MIN_LR = 1e-6
DPO_WARMUP_STEPS = 50
DPO_WEIGHT_DECAY = 0.0
DPO_GRAD_CLIP = 1.0
DPO_LOG_INTERVAL = 20
DPO_PATIENCE = 2                    # early-stop after this many epochs without val improvement

# ----- Post-training eval (step 13: win-rate vs SFT + KL from reference) -----
# On HELD-OUT instruct-valid prompts (not the train prompts pairs were made from),
# sample from SFT and DPO, score with the verifiable reward, and compare.
WINRATE_NUM_PROMPTS = 500           # held-out valid prompts (with a Words: field) to evaluate
WINRATE_SAMPLES_PER_PROMPT = 4      # completions per prompt per model; mean reward is the per-prompt score
WINRATE_LOG_EVERY = 50             # progress print every N prompts
WINRATE_SEED = 1234                # eval-only seed (held-out prompts, distinct from PREF_SEED)
WINRATE_RESULTS_FILE = "winrate.json"  # written into the DPO run dir
WINRATE_SAMPLE_DUMP = 6            # side-by-side SFT vs DPO generations (greedy) to print for eyeballing

# ----- GRPO (step 12: online RL with the verifiable + shaped reward) -----
# Sample a GROUP of completions per prompt, normalize each reward against the
# group mean/std for the advantage (no critic), and take a clipped PPO step with
# a KL leash to the frozen reference. Lighter than PPO; pairs with verifiable rewards.
GRPO_CKPT_DIR = "checkpoints/grpo"
GRPO_INIT_DIR = SFT_CKPT_DIR        # dir to resolve the init from (SFT for canonical GRPO; point at DPO to continue)
GRPO_INIT_RUN = None                # policy + frozen reference init (None = latest run under GRPO_INIT_DIR)
GRPO_PROMPT_POOL = 20_000           # instruct-train prompts (with Words:) pre-loaded to sample rollouts from
GRPO_GROUP_SIZE = 8                 # completions per prompt (the group the advantage normalizes within)
GRPO_PROMPTS_PER_STEP = 16          # prompts per rollout; rollout batch = this * GRPO_GROUP_SIZE sequences
GRPO_STEPS = 400                    # optimizer steps (rollouts)
GRPO_INNER_EPOCHS = 1               # PPO update passes per rollout (1 = pure on-policy, ratio ~ 1)
GRPO_LR = 1e-6                      # RL is delicate; well below SFT/DPO
GRPO_WARMUP_STEPS = 20
GRPO_WEIGHT_DECAY = 0.0
GRPO_GRAD_CLIP = 1.0
GRPO_BETA = 0.04                    # KL(policy || reference) penalty weight
GRPO_CLIP_EPS = 0.2                 # PPO ratio clip range (1 +/- eps)
GRPO_TEMPERATURE = 1.0              # rollout sampling temperature (> 0 for group diversity)
GRPO_TOP_K = 200
GRPO_TOP_P = 0.95
GRPO_MAX_NEW_TOKENS = 200           # cap on rollout completion length (also bounded by remaining context)
GRPO_REP_WEIGHT = REP_WEIGHT        # repetition-penalty weight in the rollout reward (0 = verifiable only)
GRPO_LOG_INTERVAL = 5
GRPO_CKPT_INTERVAL = 50             # save last.pt every N steps; best.pt on best smoothed reward
GRPO_SEED = 1337

# ----- PPO (step 12, full classic actor-critic stack) -----
# The "build the whole thing for the education" path: policy + a value head
# (critic) + frozen reference, per-token KL-penalty reward, GAE advantages, and a
# clipped actor + clipped value loss. Heavier than GRPO (a critic to train), and
# it still uses the verifiable/shaped reward directly (reward model, step 11, skipped).
PPO_CKPT_DIR = "checkpoints/ppo"
PPO_INIT_DIR = SFT_CKPT_DIR         # policy + critic backbone + frozen reference init
PPO_INIT_RUN = None                 # None = latest run under PPO_INIT_DIR
PPO_PROMPT_POOL = 20_000            # instruct-train prompts (with Words:) to sample rollouts from
PPO_PROMPTS_PER_STEP = 32           # completions per rollout (one per prompt; no groups -- critic is the baseline)
PPO_STEPS = 400                     # optimizer steps (rollouts)
PPO_INNER_EPOCHS = 4                # PPO reuses each rollout for several clipped update passes
PPO_LR = 1e-6                       # actor LR
PPO_VALUE_LR = 1e-5                 # critic can learn faster than the actor
PPO_WARMUP_STEPS = 20
PPO_WEIGHT_DECAY = 0.0
PPO_GRAD_CLIP = 1.0
PPO_CLIP_EPS = 0.2                  # PPO ratio + value clip range
PPO_VF_COEF = 0.5                   # weight of the value loss in the total objective
PPO_KL_BETA = 0.02                  # per-token KL(policy||ref) penalty folded into the reward
PPO_GAMMA = 1.0                     # discount (1.0: short episodes, no far-future discounting)
PPO_LAM = 0.95                      # GAE lambda (bias/variance trade-off)
PPO_TEMPERATURE = 1.0
PPO_TOP_K = 200
PPO_TOP_P = 0.95
PPO_MAX_NEW_TOKENS = 200
PPO_REP_WEIGHT = REP_WEIGHT         # repetition-penalty weight in the terminal reward
PPO_LOG_INTERVAL = 5
PPO_CKPT_INTERVAL = 50
PPO_SEED = 1337

# ----- Post-training comparison set (used by eval/winrate.py) -----
# Everything is graded against the baseline; each candidate stage is resolved to
# the latest run under its dir and silently skipped if that dir has no runs yet.
WINRATE_BASELINE = ("SFT", SFT_CKPT_DIR, None)   # (label, checkpoint dir, run or None=latest)
WINRATE_MODELS = [
    ("Base", CKPT_DIR, None),                    # pretrained floor: shows what SFT/post-training bought
    ("DPO", DPO_CKPT_DIR, None),
    ("GRPO", GRPO_CKPT_DIR, None),
    ("PPO", PPO_CKPT_DIR, None),
]
