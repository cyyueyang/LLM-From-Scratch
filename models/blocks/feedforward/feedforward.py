import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple: int = 256
    ):
        super(FeedForward, self).__init__()

        hidden_dim = (hidden_dim + multiple - 1) // multiple * multiple
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.w_gate(x)
        gate_output = F.silu(gate_output)
        up_output = self.w_up(x)

        fused_output = gate_output * up_output

        output = self.w_down(fused_output)

        return output