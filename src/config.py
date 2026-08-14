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
MODEL_PREFIX = "spm"               # -> data/tokenizer/spm.model + spm.vocab
VOCAB_SIZE = 8000
CHARACTER_COVERAGE = 1.0           # clean English; byte_fallback covers the rest
INPUT_SENTENCE_SIZE = 2_000_000    # subsample lines for training (no need for all ~2GB)
MAX_SENTENCE_LENGTH = 8192        # bytes; default 8192 skips the longest stories (4929)

# ----- Packing ----
PACKED_DIR = "data/packed"
PACKED_FILES = {"train": "train.bin", "valid": "val.bin"}
META_FILE = "meta.json"
PACK_BATCH_STORIES = 1024          # stories per batch-encode call
PACK_LOG_EVERY = 200_000           # print packing progress every N stories
CONTEXT_LEN = 256                  # planned model context; used to report coverage
 
 # ----- Model ----
class GPTConfig:
    """Model architecture."""
 
    vocab_size: int = VOCAB_SIZE
    block_size: int = CONTEXT_LEN
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0   # data-rich (~530M tokens / ~14M params); low overfit risk

 # ----- Training ----
SEED = 1337
DEVICE = None                 # None -> auto (cuda / cpu)
DTYPE = "bfloat16"            # bfloat16 | float16 | float32 (auto-downgraded if unsupported)
COMPILE = False               # torch.compile the model (big speedup on recent GPUs)
 
BATCH_SIZE = 64               # sequences per micro-step
GRAD_ACCUM_STEPS = 8          # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
MAX_STEPS = 10_000            # optimizer steps (~131k tokens/step; 1 epoch ~= 4k steps)
WARMUP_STEPS = 200
LR = 6e-4                     # peak LR
MIN_LR = 6e-5                 # cosine floor (~ LR / 10)
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0
 
EVAL_INTERVAL = 500           # steps between val evals + checkpoints
EVAL_ITERS = 100              # batches averaged per eval
LOG_INTERVAL = 20             # steps between train-loss logs
 
CKPT_DIR = "checkpoints"
RESUME = False                # resume from checkpoints/last.pt if present
 
 
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
EVAL_BATCH_SIZE = 64          # batch size for the deterministic perplexity sweep
