from collections.abc import Iterator
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from src.config import DATA_DIR, EOS_MARKER
from src.data.common import build_example
from src.models.tokenizer import Tokenizer

INSTRUCT_REPO_ID = "roneneldan/TinyStoriesInstruct"   # separate repo from the base data
INSTRUCT_FILE_SETS = {
    "train": "TinyStories-Instruct-train.txt",
    "valid": "TinyStories-Instruct-valid.txt",
}
STORY_MARKER = "Story:"       # up to & incl. this = prompt (masked); after = response
MAX_SFT_EXAMPLES = None       # cap examples per split (None = all); also read by training/sft.py
SFT_BUILD_LOG_EVERY = 500_000 # progress print every N records while tokenizing the split


def download_instruct(data_dir: str = DATA_DIR) -> dict[str, Path]:
    """Download the raw TinyStories-Instruct splits; return {split: path}."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, fname in INSTRUCT_FILE_SETS.items():
        print(f"downloading {fname} ...")
        local = hf_hub_download(
            repo_id=INSTRUCT_REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=data_dir,
        )
        paths[split] = Path(local)
    return paths


def parse_records(
    path: Path, marker: str = EOS_MARKER
) -> Iterator[tuple[str, str]]:
    """Yield (prompt, response) pairs from a raw instruct split.

    The prompt includes the Story: marker and the response is the story text.
    Records without a Story: marker are skipped.

    Args:
        path: Raw instruct split to read.
        marker: End-of-record separator between blocks.

    Yields:
        (prompt, response) for each record containing a Story: marker.
    """
    block: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == marker:
                yield from _emit(block)
                block = []
            else:
                block.append(line.rstrip("\n"))
    yield from _emit(block)


def _emit(block: list[str]) -> Iterator[tuple[str, str]]:
    """
    Split one record block at Story: into masked (prompt, response) pairs.

    Yields:
        A single (prompt, response) pair, or nothing if the block has no
        Story: marker, or an empty response.
    """
    text = "\n".join(block).strip()
    if STORY_MARKER not in text:
        return
    i = text.index(STORY_MARKER) + len(STORY_MARKER)
    prompt, response = text[:i], text[i:].strip()
    if response:
        yield prompt, response


class SFTDataset(Dataset):
    """
    Pre-tokenized (input_ids, labels) examples with the prompt masked.

    Each record is an instruction block (a random subset of Summary / Words /
    Sentence / Features) followed by a Story` marker and the story, terminated
    by <|endoftext|>. Story: is the deliminiter.

    Only story (+EOS) contributes to the loss, via cross_entropy(ignore_index=-1).
    """

    def __init__(
        self,
        path: Path,
        tok: Tokenizer,
        max_len: int,
        max_examples: int | None = MAX_SFT_EXAMPLES,
    ):
        self.examples: list[tuple[list[int], list[int]]] = []
        kept = skipped = seen = 0
        print(
            f"building SFT dataset from {Path(path).name} (tokenizing full split) ..."
        )
        for prompt, response in parse_records(path):
            ex = build_example(tok, prompt, response, max_len)
            seen += 1
            if ex is None:
                skipped += 1
            else:
                self.examples.append(ex)
                kept += 1
            if seen % SFT_BUILD_LOG_EVERY == 0:
                print(
                    f"  ...{seen:,} records processed ({kept:,} kept, {skipped:,} dropped)"
                )
            if max_examples and kept >= max_examples:
                break
        print(
            f"{Path(path).name}: {kept:,} examples kept, {skipped:,} dropped (> {max_len} tokens)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.examples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )
