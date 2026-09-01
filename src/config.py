from dataclasses import dataclass

# ----- Download Data ----
DATA_DIR = "data/raw"
EOS_MARKER = "<|endoftext|>"  # separates stories in the dataset
FILE_SETS = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}

# ----- Prepare Tokenizer ----
VOCAB_SIZE = 8000

# ----- Pack Data ----
PACKED_DIR = "data/packed"
PACKED_FILES = {"train": "train.bin", "valid": "val.bin"}
META_FILE = "meta.json"
CONTEXT_LEN = 256  # model context length


# ----- Model (identical across every layer) ----
@dataclass
class GPTConfig:
    """Model architecture."""

    vocab_size: int = VOCAB_SIZE
    block_size: int = CONTEXT_LEN
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0  # data-rich (~530M tokens / ~14M params); low overfit risk

# ----- Global reproducibility ----
SEED = 1337  # eval uses its own separate seed
# Pretraining resume wiring (the schedule itself lives on PretrainConfig)
RESUME = False       # resume pretraining from a prior run's last.pt
RESUME_FROM = None   # None -> latest run dir; or a run dir / .pt path

# ----- Checkpoint layout + data paths (contracts between stages) ----
CKPT_DIR = "checkpoints/base"        # pretrained (base) runs
SFT_CKPT_DIR = "checkpoints/sft"     # Also the DPO/RL init
DPO_CKPT_DIR = "checkpoints/dpo"
GRPO_CKPT_DIR = "checkpoints/grpo"
PPO_CKPT_DIR = "checkpoints/ppo"
SPEC_TARGET_DIR = "checkpoints/sft"  # speculative: model whose distribution we preserve
SPEC_DRAFT_DIR = "checkpoints/draft"  # draft model for speculative
DPO_DATA_DIR = "data/dpo"
PAIRS_FILE = "pairs.jsonl"           # -> data/dpo/pairs.jsonl

# ----- Reward / metric definition (must match across training + eval) ----
REP_NGRAM = 2     # n-gram size for the repetition penalty
REP_WEIGHT = 0.5  # weight of the repetition penalty subtracted from the verifiable reward
