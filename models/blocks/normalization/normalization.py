import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        x_normed = (x - mean) / (torch.sqrt(var + self.eps))

        return self.gamma * x_normed + self.beta

class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * x_normed

class Qwen2RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(Qwen2RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (1 + self.gamma) * x_normed


if __name__ == "__main__":
    x = torch.randn(4, 12, 128)

    layernorm = LayerNorm(128)
    rmsnorm = RMSNorm(128)
    qwen2rmsnorm = Qwen2RMSNorm(128)

    x_1 = layernorm(x)
    x_2 = rmsnorm(x)
    x_3 = qwen2rmsnorm(x)

    print(x_1.size())
    print(x_2.size())
    print(x_3.size())