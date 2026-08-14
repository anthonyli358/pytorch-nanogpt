import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.config import GPTConfig
from src.models.nn_block import Block


class GPT(nn.Module):
    """
    Decoder-only GPT language model, assembled from transformer blocks.
    """

    def __init__(self, cfg: GPTConfig | None = None):
        """Build the model.

        Args:
            cfg: Model hyperparameters; a default :class:`GPTConfig` if None.
        """
        super().__init__()
        cfg = cfg or GPTConfig()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: the token embedding and the output projection share weights.
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2): std shrinks with depth.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Count parameters.

        Args:
            non_embedding: If True, exclude the position embedding table. (The
                token embedding is tied to the LM head, so it is always counted.)

        Returns:
            Parameter count.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Run the model.

        Args:
            idx: Token ids, shape (B, T).
            targets: Next-token targets, shape (B, T). Use`-1 to ignore
                a position in the loss.

        Returns:
            (logits, loss).
                With targets, logits is (B, T, vocab) and loss is the mean cross-entropy.
                Without targets (inference), only the last position is computed.
        """
        B, T = idx.shape
        assert (
            T <= self.cfg.block_size
        ), f"sequence length {T} > block size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss

        logits = self.lm_head(x[:, [-1], :])  # only the last position is needed
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample continuations (basic; top-p added in step 6).

        Args:
            idx: Prompt token ids, shape (B, T).
            max_new_tokens: Number of tokens to generate.
            temperature: Softmax temperature; lower is greedier.
            top_k: If set, sample only from the top-k logits.

        Returns:
            idx extended by max_new_tokens columns.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0.0:  # greedy
                idx_next = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                continue

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()  # keep first token past p
                remove[..., 0] = False
                remove = remove.scatter(-1, sorted_idx, remove)
                logits = logits.masked_fill(remove, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
