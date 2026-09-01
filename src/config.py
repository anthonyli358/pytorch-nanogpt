"""Shared, cross-cutting configuration.

Only things that must be consistent across stages live here: the model
architecture (every stage trains/loads the same GPT), filesystem paths (the
contracts by which stages hand off checkpoints and data), the reward/metric
definition (must mean the same thing in training and eval), and the global seed.

Per-stage hyperparameters live on each trainer's ``*Config`` dataclass
(``src/training/*.py``); inference sampling defaults live in
``src/inference/generate.py``; the eval protocol lives in ``src/eval/winrate.py``;
device + autocast dtype are auto-detected in ``training/common.py``.
"""

from dataclasses import dataclass

# ----- Data ----
DATA_DIR = "data/raw"
EOS_MARKER = "<|endoftext|>"  # separates stories in the dataset
FILE_SETS = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}

# ----- Tokenizer / packing ----
VOCAB_SIZE = 8000
PACKED_DIR = "data/packed"
PACKED_FILES = {"train": "train.bin", "valid": "val.bin"}
META_FILE = "meta.json"
CONTEXT_LEN = 256  # model context length


# ----- Model (identical across every stage) ----
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
SEED = 1337  # every training/data-gen stage; eval uses its own WINRATE_SEED

# ----- Reward / metric definition (must match across training + eval) ----
REP_NGRAM = 2     # n-gram size for the distinct-n-gram diversity metric / repetition penalty
REP_WEIGHT = 0.5  # weight of the repetition penalty subtracted from the verifiable reward

# ----- Checkpoint layout + data paths (contracts between stages) ----
CKPT_DIR = "checkpoints/base"        # pretrained (base) runs; post-training stages sit beside it
SFT_CKPT_DIR = "checkpoints/sft"     # SFT; also the speculative target + the DPO/RL init
DPO_CKPT_DIR = "checkpoints/dpo"
GRPO_CKPT_DIR = "checkpoints/grpo"
PPO_CKPT_DIR = "checkpoints/ppo"
SPEC_TARGET_DIR = "checkpoints/sft"  # speculative: model whose distribution we preserve
SPEC_DRAFT_DIR = "checkpoints/draft"  # speculative: a smaller/cheaper draft (draft.py writes here)
DPO_DATA_DIR = "data/dpo"
PAIRS_FILE = "pairs.jsonl"           # -> data/dpo/pairs.jsonl

# ----- Pretraining resume wiring (the schedule itself lives on PretrainConfig) ----
RESUME = False       # resume pretraining from a prior run's last.pt
RESUME_FROM = None   # None -> latest run dir; or a run dir / .pt path

# Example-length cap shared by DPO training and eval/winrate's KL computation.
DPO_MAX_LEN = None   # cap example length in tokens; None -> model block_size
