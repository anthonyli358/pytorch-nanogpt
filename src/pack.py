import json
from pathlib import Path

import numpy as np

from src.config import (
    DATA_DIR,
    FILE_SETS,
    PACKED_DIR,
    PACKED_FILES,
    META_FILE,
    PACK_BATCH_STORIES,
    PACK_LOG_EVERY,
    CONTEXT_LEN,
)
from src.data import read_stories
from src.tokenizer import Tokenizer

DTYPE = np.uint16


def pack_split(tokenizer: Tokenizer, raw_path: Path, out_path: Path,
               batch_stories: int = PACK_BATCH_STORIES,
               log_every: int = PACK_LOG_EVERY) -> tuple[int, int, np.ndarray]:
    """Encode one split and write it as a flat uint16 .bin.

    Stories are batch-encoded for speed and appended to the file incrementally,
    so peak memory stays bounded regardless of corpus size.

    Args:
        tokenizer: A loaded :class:`Tokenizer`.
        raw_path: Raw .txt split to read.
        out_path: Destination ``.bin`` path.
        batch_stories: Number of stories per batch-encode call.
        log_every: Print running progress every this many stories.

    Returns:
        ``(n_tokens, n_stories, lengths)`` where ``lengths`` is the per-story
        token count (excluding the appended eos), as a uint32 array.
    """
    sp = tokenizer.sp
    eos_id = tokenizer.eos_id
    n_tokens = n_stories = 0
    length_chunks: list[np.ndarray] = []
    buf: list[str] = []

    with out_path.open("wb") as out:
        def flush(stories: list[str]) -> None:
            nonlocal n_tokens, n_stories
            if not stories:
                return
            encoded = sp.encode(stories, out_type=int)   # list[list[int]]
            flat: list[int] = []
            lens = np.empty(len(encoded), dtype=np.uint32)
            for i, ids in enumerate(encoded):
                lens[i] = len(ids)
                flat.extend(ids)
                flat.append(eos_id)
            np.asarray(flat, dtype=DTYPE).tofile(out)
            length_chunks.append(lens)
            n_tokens += len(flat)
            n_stories += len(stories)

        next_log = log_every
        for story in read_stories(raw_path):
            buf.append(story)
            if len(buf) >= batch_stories:
                flush(buf)
                buf = []
                if n_stories >= next_log:
                    print(f"  {out_path.name}: {n_stories:,} stories, {n_tokens:,} tokens ...")
                    next_log += log_every
        flush(buf)

    lengths = np.concatenate(length_chunks) if length_chunks else np.empty(0, np.uint32)
    return n_tokens, n_stories, lengths


def pack_data(tokenizer: Tokenizer | None = None, overwrite: bool = False) -> dict:
    """Pack all splits to uint16 memmaps and write meta.json.

    At train time we read tiny random slices out of a ~1GB file thousands of times per epoch, 
    and memmap is the mechanism that makes that cheap.

    Skips packing (and just loads the existing meta) when the outputs already
    exist and ``overwrite`` is False, so this is safe to call in a pipeline.

    Args:
        tokenizer: A loaded tokenizer; loaded from the default model if None.
        overwrite: If True, repack even when outputs already exist.

    Returns:
        The meta dict (vocab_size, eos_id, per-split token/story counts).
    """
    tokenizer = tokenizer or Tokenizer()
    assert tokenizer.vocab_size <= np.iinfo(DTYPE).max + 1, "vocab too large for uint16"

    packed_dir = Path(PACKED_DIR)
    meta_path = packed_dir / META_FILE
    out_paths = {s: packed_dir / PACKED_FILES[s] for s in FILE_SETS}

    if not overwrite and meta_path.exists() and all(p.exists() for p in out_paths.values()):
        print(f"already packed: {meta_path}")
        return json.loads(meta_path.read_text())

    packed_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    train_lengths = None

    for split in FILE_SETS:
        raw_path = Path(DATA_DIR) / FILE_SETS[split]
        out_path = out_paths[split]
        n_tokens, n_stories, lengths = pack_split(tokenizer, raw_path, out_path)
        counts[split] = {"tokens": int(n_tokens), "stories": int(n_stories)}
        print(f"{split}: {n_tokens:,} tokens, {n_stories:,} stories -> {out_path}")
        if split == "train":
            train_lengths = lengths

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "dtype": np.dtype(DTYPE).name,
        "splits": counts,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    if train_lengths is not None and train_lengths.size:
        _report_lengths(train_lengths)
    return meta


def _report_lengths(lengths: np.ndarray, context_len: int = CONTEXT_LEN) -> None:
    """Print the train-split story token-length distribution vs the context length."""
    print("\n=== train story token-length distribution ===")
    for p in (50, 90, 95, 99):
        print(f"p{p:<2}: {np.percentile(lengths, p):6.0f}")
    print(f"max: {int(lengths.max()):6d}   mean: {lengths.mean():6.1f}")
    frac = float((lengths <= context_len).mean())
    print(f"fraction of stories <= {context_len} tokens: {frac:.3f}")
