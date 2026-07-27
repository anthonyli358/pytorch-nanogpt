# ----- Data Download ----
REPO_ID = "roneneldan/TinyStories"
OUT_DIR = "data/raw"
EOS_MARKER = "<|endoftext|>"  # separate for stories in the dataset
FILE_SETS = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}

