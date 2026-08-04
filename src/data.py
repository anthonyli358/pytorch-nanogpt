from pathlib import Path
from huggingface_hub import hf_hub_download

from src.config import REPO_ID, DATA_DIR, EOS_MARKER, FILE_SETS


def download(data_dir: str = DATA_DIR) -> dict[str, Path]:
    """Download the raw train/valid .txt files and return their local paths."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for split, fname in FILE_SETS.items():
        print(f"downloading {fname} ...")
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=data_dir,  # copies the file straight into data_dir
        )
        paths[split] = Path(local)
    return paths


def summarise(path: Path) -> tuple[int, int]:
    """
    Stream the file once; return (num_stories, num_bytes).

    Avoid loading the whole file into memory, so memory-safe on ~2GB.
    """
    num_stories = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOS_MARKER:
                num_stories += 1
    return num_stories, path.stat().st_size


def download_data(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """
    Download the raw TinyStories splits and report their locations.

    Files are fetched via the Hugging Face Hub cache, so repeated calls
    return the cached paths without re-downloading.
    """
    paths = download(data_dir)
    for split, path in paths.items():
        n, nbytes = summarise(path)
        print(f"{split}: {path}   ({n:,} stories, {nbytes / 1e6:.1f} MB)")
    return paths


def read_stories(path: Path, marker: str = EOS_MARKER, limit: int | None = None):
    """Stream stories from a raw split, splitting on the marker line.
 
    Memory-safe: yields one story at a time rather than loading the file.
 
    Args:
        path: Raw .txt split.
        marker: Document separator (a line equal to this ends a story).
        limit: Stop after yielding this many stories, if given.
 
    Yields:
        Each story as a single string (internal newlines preserved).
    """
    story: list[str] = []
    n = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == marker:
                if story:
                    yield "\n".join(story)
                    story = []
                    n += 1
                    if limit and n >= limit:
                        return
            else:
                story.append(line.rstrip("\n"))
    if story:
        yield "\n".join(story)
