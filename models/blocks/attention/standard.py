import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from ..positional_encoding.positional_encoding import RoPE

class StandardAttention(nn.Module):
    def __init__(self, args):
        super(StandardAttention, self).__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.d_model = self.args.d_model

        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads

        assert self.n_heads // self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads

        self.w_q = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.w_kv = nn.Linear(self.d_model, 2 * self.n__kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.n_heads * self.head_dim, args.d_model, bias=False)

        mask = torch.ones((1, 1, args.max_seq_len, args.max_seq_len), dtype=torch.bool)
        mask = torch.tril(mask)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs) -> torch.Tensor:
        pass