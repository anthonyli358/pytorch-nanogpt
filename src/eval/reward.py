"""
Verifiable rewards for TinyStories-Instruct completions.
"""

import re

_FIELD_RE = {
    "words": re.compile(r"^Words:\s*(.+)$", re.M),
    "features": re.compile(r"^Features:\s*(.+)$", re.M),
    "sentence": re.compile(r"^Sentence:\s*(.+)$", re.M),
    "summary": re.compile(r"^Summary:\s*(.+)$", re.M),
}


def parse_instruction(prompt: str) -> dict:
    """
    Extract instruction fields from a prompt (the text up to `Story:`).

    Returns:
        Dict with any of `words` (list[str]), `features` (list[str]),
        `sentence` (str), and `summary` (str) that are present.
    """
    fields: dict = {}
    for key, rx in _FIELD_RE.items():
        m = rx.search(prompt)
        if m:
            fields[key] = m.group(1).strip()
    if "words" in fields:
        fields["words"] = [
            w.strip().lower() for w in re.split(r"[,\s]+", fields["words"]) if w.strip()
        ]
    if "features" in fields:
        fields["features"] = [
            f.strip() for f in fields["features"].split(",") if f.strip()
        ]
    return fields


def word_inclusion(story: str, words: list[str]) -> float:
    """
    Fraction of required words present in the story (lenient prefix match).

    Each distinct word counts once (set semantics), so neither a duplicate entry
    in the `Words:` list nor repeating a word in the story changes the score.
    """
    unique = set(words)
    if not unique:
        return 0.0
    text = story.lower()
    hits = sum(1 for w in unique if re.search(rf"\b{re.escape(w)}", text))
    return hits / len(unique)


def verifiable_reward(prompt: str, story: str) -> float | None:
    """
    Score a completion in [0, 1] from the verifiable instruction fields.

    Currently the word-inclusion fraction. Returns None when the prompt has
    no `Words:` field (so nothing verifiable).

    Args:
        prompt: The instruction (text up to and including `Story:`).
        story: The generated story text.

    Returns:
        Reward in [0, 1], or None if there is nothing to verify.
    """
    fields = parse_instruction(prompt)
    if "words" not in fields:
        return None
    return word_inclusion(story, fields["words"])


def distinct_ngram_ratio(story: str, n: int = 2) -> float:
    """
    Fraction of word n-grams that are distinct (1.0 = none repeat, low = looping).

    A fluency proxy that the word-inclusion reward is blind to.
    Helps to prevent greedy decode loops which repeat phrases and satify the
    word inclusion reward.
    """
    tokens = story.lower().split()
    if len(tokens) < n + 1:
        return 1.0  # too short to repeat
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_penalty(story: str, n: int = 2) -> float:
    """Repetition amount in [0, 1]: 0 = all n-grams distinct, higher = more looping."""
    return 1.0 - distinct_ngram_ratio(story, n)


def shaped_reward(
    prompt: str, story: str, rep_weight: float = 0.5, n: int = 2
) -> float | None:
    """
    The overall training signal - verifiable reward minus a repetition penalty.

    Prevents reward hacking with two objectives, both include the words and stay diverse.
    At rep_weight=0 it is exactly the verifiable reward.

    Args:
        prompt: The instruction (text up to and including `Story:`).
        story: The generated story text.
        rep_weight: Weight of the repetition penalty subtracted from the reward.
        n: n-gram size for the diversity metric.

    Returns:
        Shaped reward, or None if there is nothing to verify.
    """
    base = verifiable_reward(prompt, story)
    if base is None:
        return None
    return base - rep_weight * repetition_penalty(story, n)
