import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    # hidden_dim 通常 为 8/3 dim
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden_dim = (hidden_dim + multiple_of - 1) // multiple_of * multiple_of
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

if __name__ == "__main__":
    x = torch.randn(16, 128, 512)

    ffn = FeedForward(512, 512 * 4)

    print(ffn(x).size())
