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
