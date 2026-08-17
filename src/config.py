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

CKPT_DIR = "checkpoints"
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
SFT_CKPT_DIR = "sft_checkpoints"   # separate from pretraining checkpoints/
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
DPO_CKPT_DIR = "dpo_checkpoints"    # separate from SFT / pretraining checkpoints
DPO_INIT_RUN = None                 # SFT run to init policy + frozen reference from (None = latest)
DPO_MAX_LEN = None                  # cap example length in tokens; None -> model block_size
DPO_VAL_FRACTION = 0.05             # fraction of pairs held out for val loss / reward accuracy
DPO_BETA = 0.1                      # KL strength in the DPO objective (higher = stay closer to ref)
DPO_BATCH_SIZE = 16                 # pairs per step (2x sequences forwarded: chosen + rejected)
DPO_EPOCHS = 1
DPO_LR = 1e-5                       # DPO is sensitive; well below the SFT peak
DPO_MIN_LR = 1e-6
DPO_WARMUP_STEPS = 50
DPO_WEIGHT_DECAY = 0.0
DPO_GRAD_CLIP = 1.0
DPO_LOG_INTERVAL = 20
DPO_PATIENCE = 2                    # early-stop after this many epochs without val improvement
