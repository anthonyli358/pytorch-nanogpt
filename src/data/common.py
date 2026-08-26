import torch

from src.models.tokenizer import Tokenizer


def build_example(
    tok: Tokenizer, prompt: str, response: str, max_len: int
) -> tuple[list[int], list[int]] | None:
    """
    Tokenize one pair from TinyStories into (input_ids, labels) for next-token prediction.

    Targets are shifted because the model predicts the next token
    """
    prompt_ids = tok.encode(prompt)
    response_ids = tok.encode(response) + [tok.eos_id]
    if len(prompt_ids) + len(response_ids) > max_len:  # incomplete story
        return None
    full = prompt_ids + response_ids
    input_ids = full[:-1]
    labels = full[1:]
    for j in range(len(prompt_ids) - 1):  # mask targets that predict a prompt token
        labels[j] = -1
    return input_ids, labels


def pad_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Right-pad a batch to its max length: 
    - inputs with pad_id
    - labels with`-1

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
