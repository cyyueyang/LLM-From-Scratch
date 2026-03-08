import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import math

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, d_model = x.size() # [bs, seq_len, d_model]

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0) # [1, seq_len]

        x = x + self.pe(positions)
        x = self.dropout(x)

        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super(SinusoidalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

@dataclass
class RoPEConfig:
    head_dim: int = 64
    max_seq_len: int = 512
    base: float = 10000.0

class RoPE(nn.Module):
    def __init__(self, config: RoPEConfig):
        super(RoPE, self).__init__()
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.base = config.base

        theta = 1.0 / (self.base ** torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        t = torch.arange(0, self.max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, theta)

        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def _rotate_half(self, t: torch.Tensor) -> torch.Tensor:
        """后半部分取负数放到前面，前半部分放后面
            也可以相邻元素之间这么操作  那这样就应该修改 cos_cached 和 sin_cached 改为插值重复
        """

        t1 = t[..., :self.head_dim // 2]
        t2 = t[..., self.head_dim // 2:]
        return torch.cat((-t2, t1), dim=-1)

    def apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        # x size [bs, num_heads, seq_len, head_dim]
        _, _, seq_len, _ = x.size()

        cos = self.cos_cached.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, head_dim / 2]
        sin = self.sin_cached.unsqueeze(0).unsqueeze(0) # [1, 1, max_seq_len, head_dim / 2]

        cos = cos.repeat(1, 1, 1, 2)
        sin = sin.repeat(1, 1, 1, 2)

        cos = cos[:, :, :seq_len, :].to(x.device)
        sin = sin[:, :, :seq_len, :].to(x.device)

        x_rotated = x * cos + self._rotate_half(x) * sin

        return x_rotated.type_as(x)

    def apply_rotary_emb_paged(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        针对 PagedAttention 使用的 RoPE 因为 不同序列的tokens物理上不连续
        :param x: 输入张量 [num_tokens, num_heads, head_dim]
        :param positions: [num_tokens, ]  dim = 1
        :return: 旋转后的张量
        """

        cos = self.cos_cached[positions].to(x.device) # [num_tokens, head_dim / 2]
        sin = self.sin_cached[positions].to(x.device) # [num_tokens, head_dim / 2]

        cos = cos.unsqueeze(1) # [num_tokens, 1, head_dim / 2]
        sin = sin .unsqueeze(1) # [num_tokens, 1, head_dim / 2]

        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)

        x_rotated = x * cos + self._rotate_half(x) * sin

        return x_rotated.type_as(x)

def get_alibi_slopes(num_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:

    # m_i = power(2, -8i / n)
    slopes = 2.0 ** (-torch.arange(1, num_heads + 1).float() * 8.0 / num_heads)

    relative_positions = torch.arange(0, seq_len, device=device).unsqueeze(0) - torch.arange(0, seq_len, device=device).unsqueeze(1)

    alibi = -torch.abs(relative_positions)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)

    return alibi.unsqueeze(0)


if __name__ == "__main__":
    model = LearnedPositionalEncoding(d_model=512, max_seq_len=512)
    x = torch.randn(4, 512, 512)
    y = model(x)
    print(y.size())

    model = SinusoidalPositionalEncoding(d_model=512, max_seq_len=512)
    x = torch.randn(4, 512, 512)
    y = model(x)
    print(y.size())

    model = RoPE(RoPEConfig())
    x = torch.randn(4, 16, 128, 64)
    y = model.apply_rotary_emb(x)
    print(y.size())

    y = get_alibi_slopes(num_heads=16, seq_len=16, device=torch.device('cpu'))
    print(y.size())








