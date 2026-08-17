"""Verifiable rewards for TinyStories-Instruct completions.

The instruction fields give programmatically checkable constraints. The cleanest
is ``Words:`` -- a story either contains the required words or it doesn't, with
no model or labels needed. This score turns sampled completions into preference
pairs for DPO (and later a reward signal for GRPO).

Matching is lenient prefix-based (``jump`` matches ``jumped``, ``jumping``) to
credit inflected forms, at the cost of occasional false positives on short words.
"""

import re

_FIELD_RE = {
    "words": re.compile(r"^Words:\s*(.+)$", re.M),
    "features": re.compile(r"^Features:\s*(.+)$", re.M),
    "sentence": re.compile(r"^Sentence:\s*(.+)$", re.M),
    "summary": re.compile(r"^Summary:\s*(.+)$", re.M),
}


def parse_instruction(prompt: str) -> dict:
    """Extract instruction fields from a prompt (the text up to ``Story:``).

    Returns:
        Dict with any of ``words`` (list[str]), ``features`` (list[str]),
        ``sentence`` (str), ``summary`` (str) that are present.
    """
    fields: dict = {}
    for key, rx in _FIELD_RE.items():
        m = rx.search(prompt)
        if m:
            fields[key] = m.group(1).strip()
    if "words" in fields:
        fields["words"] = [w.strip().lower() for w in re.split(r"[,\s]+", fields["words"]) if w.strip()]
    if "features" in fields:
        fields["features"] = [f.strip() for f in fields["features"].split(",") if f.strip()]
    return fields


def word_inclusion(story: str, words: list[str]) -> float:
    """Fraction of required words present in the story (lenient prefix match).

    Each distinct word counts once (set semantics), so neither a duplicate entry
    in the ``Words:`` list nor repeating a word in the story changes the score --
    presence is boolean per required word.
    """
    unique = set(words)
    if not unique:
        return 0.0
    text = story.lower()
    hits = sum(1 for w in unique if re.search(rf"\b{re.escape(w)}", text))
    return hits / len(unique)


def verifiable_reward(prompt: str, story: str) -> float | None:
    """Score a completion in [0, 1] from the verifiable instruction fields.

    Currently the word-inclusion fraction. Returns None when the prompt carries
    no verifiable signal (no ``Words:`` field), so callers can skip it.

    Args:
        prompt: The instruction (text up to and including ``Story:``).
        story: The generated story text.

    Returns:
        Reward in [0, 1], or None if there is nothing to verify.
    """
    fields = parse_instruction(prompt)
    if "words" not in fields:
        return None
    return word_inclusion(story, fields["words"])