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

if __name__ == "__main__":
    model = LearnedPositionalEncoding(d_model=512, max_seq_len=512)
    x = torch.randn(4, 512, 512)
    y = model(x)
    print(y.size())

    model = SinusoidalPositionalEncoding(d_model=512, max_seq_len=512)
    x = torch.randn(4, 512, 512)
    y = model(x)
    print(y.size())





