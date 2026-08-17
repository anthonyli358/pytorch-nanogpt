"""SFT data for TinyStories-Instruct.

Each record is an instruction block (a random subset of Summary / Words /
Sentence / Features) followed by a ``Story:`` marker and the story, terminated
by ``<|endoftext|>``. The split point is ``Story:``: everything up to and
including it is the prompt; the story after it is the response.

For SFT we mask the prompt -- ``labels`` are ``-1`` over the prompt span so only
the story (+EOS) contributes to the loss, via ``cross_entropy(ignore_index=-1)``.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from src.config import (
    REPO_ID,
    OUT_DIR,
    EOS_MARKER,
    INSTRUCT_FILE_SETS,
    STORY_MARKER,
    MAX_SFT_EXAMPLES,
)
from src.models.tokenizer import Tokenizer


def download_instruct(out_dir: str = OUT_DIR) -> dict[str, Path]:
    """Download the raw TinyStories-Instruct splits; return {split: path}."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, fname in INSTRUCT_FILE_SETS.items():
        print(f"downloading {fname} ...")
        local = hf_hub_download(
            repo_id=REPO_ID, filename=fname, repo_type="dataset", local_dir=out_dir
        )
        paths[split] = Path(local)
    return paths


def parse_records(path: Path, marker: str = EOS_MARKER):
    """Yield ``(prompt, response)`` pairs from a raw instruct split.

    The prompt includes the ``Story:`` marker; the response is the story text.
    Records without a ``Story:`` marker are skipped.
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


def _emit(block: list[str]):
    text = "\n".join(block).strip()
    if STORY_MARKER not in text:
        return
    i = text.index(STORY_MARKER) + len(STORY_MARKER)
    prompt, response = text[:i], text[i:].strip()
    if response:
        yield prompt, response


def build_example(tok: Tokenizer, prompt: str, response: str, max_len: int):
    """Tokenize one pair into ``(input_ids, labels)`` with the prompt masked.

    Returns None if the example exceeds ``max_len`` (dropped rather than
    truncated, so the model only ever sees complete stories).
    """
    prompt_ids = tok.encode(prompt)
    response_ids = tok.encode(response) + [tok.eos_id]
    if len(prompt_ids) + len(response_ids) > max_len:
        return None
    input_ids = prompt_ids + response_ids
    labels = [-1] * len(prompt_ids) + response_ids  # mask the prompt span
    return input_ids, labels


class SFTDataset(Dataset):
    """Pre-tokenized (input_ids, labels) examples with the prompt masked."""

    def __init__(
        self,
        path: Path,
        tok: Tokenizer,
        max_len: int,
        max_examples: int | None = MAX_SFT_EXAMPLES,
    ):
        self.examples: list[tuple[list[int], list[int]]] = []
        kept = skipped = 0
        for prompt, response in parse_records(path):
            ex = build_example(tok, prompt, response, max_len)
            if ex is None:
                skipped += 1
                continue
            self.examples.append(ex)
            kept += 1
            if max_examples and kept >= max_examples:
                break
        print(
            f"{Path(path).name}: {kept:,} examples kept, {skipped:,} dropped (> {max_len} tokens)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        input_ids, labels = self.examples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )


def pad_batch(batch, pad_id: int):
    """Right-pad a batch to its max length: inputs with ``pad_id``, labels with ``-1``.

    Right-padding is safe under causal attention -- real tokens never attend into
    padding, and padded positions carry label -1 so they add no loss.
    """
    max_len = max(x.size(0) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - x.size(0)
        xs.append(torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)]))
        ys.append(torch.cat([y, torch.full((pad,), -1, dtype=torch.long)]))
    return torch.stack(xs), torch.stack(ys)
