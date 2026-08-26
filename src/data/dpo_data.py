import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.common import build_example, pad_batch
from src.models.tokenizer import Tokenizer


def read_pairs(path: Path) -> Iterator[tuple[str, str, str]]:
    """
    Yield (prompt, chosen, rejected) triples from a pairs.jsonl file.

    Created by make_preferences.py
    """
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec["prompt"], rec["chosen"], rec["rejected"]


class DPODataset(Dataset):
    """
    Preference pairs pre-tokenized into masked (chosen, rejected) examples.

    The reuse of build_example means the prompt is masked to -1

    A pair is dropped if eitherprompt or response exceeds max_len.
    """

    def __init__(self, path: Path, tok: Tokenizer, max_len: int):
        self.pairs: list[tuple[tuple, tuple]] = []
        kept = skipped = 0
        for prompt, chosen, rejected in read_pairs(path):
            c = build_example(tok, prompt, chosen, max_len)
            r = build_example(tok, prompt, rejected, max_len)
            if c is None or r is None:
                skipped += 1
                continue
            self.pairs.append((c, r))
            kept += 1
        print(
            f"{Path(path).name}: {kept:,} pairs kept, {skipped:,} dropped (> {max_len} tokens)"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        (cx, cy), (rx, ry) = self.pairs[idx]
        chosen = (
            torch.tensor(cx, dtype=torch.long),
            torch.tensor(cy, dtype=torch.long),
        )
        rejected = (
            torch.tensor(rx, dtype=torch.long),
            torch.tensor(ry, dtype=torch.long),
        )
        return chosen, rejected


def dpo_collate(
    batch: list[
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    ],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a batch of pairs into ``(2B, T)`` inputs/labels: chosen then rejected.

    Both halves share one padded length so the caller can run a single forward
    over the concatenation and split the per-sequence log-probs at ``B``.
    """
    chosen = [c for c, _ in batch]
    rejected = [r for _, r in batch]
    return pad_batch(chosen + rejected, pad_id)  # (2B, T), (2B, T)
