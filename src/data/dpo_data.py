"""Preference-pair data for DPO.

Each line of ``data/dpo/pairs.jsonl`` is ``{prompt, chosen, rejected, ...}``,
produced by ``make_preferences.py``. A pair becomes two SFT-style examples --
``(prompt, chosen)`` and ``(prompt, rejected)`` -- reusing ``build_example`` so
the prompt is masked to ``-1`` exactly as in SFT: only the response tokens (+EOS)
carry a label, which is precisely the span the DPO log-probs sum over.

The collate packs both halves of every pair into one ``(2B, T)`` batch (chosen
first, then rejected) so a single forward pass scores both. ``pad_batch`` pads
inputs with ``pad_id`` and labels with ``-1``, and the shared max length lets the
two halves stack for the concatenated forward.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.sft_data import build_example, pad_batch
from src.models.tokenizer import Tokenizer


def read_pairs(path: Path):
    """Yield ``(prompt, chosen, rejected)`` triples from a pairs.jsonl file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec["prompt"], rec["chosen"], rec["rejected"]


class DPODataset(Dataset):
    """Preference pairs pre-tokenized into masked (chosen, rejected) examples.

    A pair is dropped only if *either* completion (prompt + response) exceeds
    ``max_len`` -- the two halves must both survive to form a valid contrast.
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
        print(f"{Path(path).name}: {kept:,} pairs kept, {skipped:,} dropped (> {max_len} tokens)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        (cx, cy), (rx, ry) = self.pairs[idx]
        chosen = (torch.tensor(cx, dtype=torch.long), torch.tensor(cy, dtype=torch.long))
        rejected = (torch.tensor(rx, dtype=torch.long), torch.tensor(ry, dtype=torch.long))
        return chosen, rejected


def dpo_collate(batch, pad_id: int):
    """Pack a batch of pairs into ``(2B, T)`` inputs/labels: chosen then rejected.

    Both halves share one padded length so the caller can run a single forward
    over the concatenation and split the per-sequence log-probs at ``B``.
    """
    chosen = [c for c, _ in batch]
    rejected = [r for _, r in batch]
    return pad_batch(chosen + rejected, pad_id)  # (2B, T), (2B, T)
