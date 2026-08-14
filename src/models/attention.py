import torch
import torch.nn as nn
from torch.nn import functional as F

from src.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    This is the decoder self-attention from the seq2seq transformer.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # fused Q, K, V
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)  # output projection
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
