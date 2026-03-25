import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

from sympy.codegen.ast import float32


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super(LearnedPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = x + self.pe(positions)

        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super(SinusoidalPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len, device=d_model).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len, :]

        return x

@dataclass
class RoPEConfig:
    head_dim: int = 64
    max_seq_len: int = 512
    base: float = 10000.0


class RoPE(nn.Module):
    def __init__(self, config: RoPEConfig):
        super(RoPE, self).__init__()

        self.config = config
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.base = config.base

        theta = 1.0 / (self.base ** torch.arange(0, self.head_dim, 2).float())
        m = torch.arange(0, self.max_seq_len, dtype=float32)

        freqs = torch.outer(m, theta)

        self.register_buffer("cos_cached", freqs.cos())  # [max_len, head_dim / 2]
        self.register_buffer("sin_cached", freqs.sin())  # [max_len, head_dim / 2]

    def apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [bs, num_heads, seq_len, head_dim]
        bs, num_heads, seq_len, head_dim = x.shape

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return torch.stack([x1, x2], dim=-1).flatten(-2)

    def apply_rotary_emb_paged(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x shape [num_tokens, num_heads, head_dim]
        # positions shape [num_tokens, ]

        cos = self.cos_cached[positions].unsqueeze(1)  # [num_tokens, 1, head_dim / 2]
        sin = self.sin_cached[positions].unsqueeze(1)  # [num_tokens, 1, head_dim / 2]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return torch.stack([y1, y2], dim=-1).flatten(-2)