import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.feedforward.feedforward import FeedForward

class MoELayer(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            num_experts: int,
            num_experts_per_tok: int,
            multiple_of: int = 256,
            num_shared_experts: int = 2,
            use_aux_free_lb: bool = True,
    ):
        super(MoELayer, self).__init__()
        hidden_dim = multiple_of * (hidden_dim + multiple_of - 1) // multiple_of

        self.dim = dim
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.router = nn.Linear(dim, num_experts, bias=False)
        if use_aux_free_lb:
            self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(dim, hidden_dim, multiple_of) for _ in range(num_shared_experts)
            ])

        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of) for _ in range(num_experts)
        ])

        self.register_buffer("last_expert_counts", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, d_model = x.shape

        x_flat = x.view(-1, d_model)

        shared_output = 0
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                shared_output += expert(x_flat)

        router_logits = self.router(x_flat)

        if self.use_aux_free_lb:
            router_logits = router_logits + self.expert_bias

        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(x.dtype) # [total_tokens, topk]

        final_output = torch.zeros_like(x_flat)
        # [total_tokens, topk, num_experts] -> [num_experts, total_tokens, topk]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 0, 1)

        if self.training and self.use_aux_free_lb:
            current_counts = expert_mask.sum(dim=(1, 2)).detach().float()
            self.last_experts_counts = current_counts

        for expert_idx in range(self.num_experts):
            idx_in_topk = torch.where(expert_mask[expert_idx] > 0)

            # idx_in_topk[0] token的全局索引
            # idx_in_topk[1] 在topk列表中的位置
            if idx_in_topk[0].numel() == 0:
                continue

            token_indices = idx_in_topk[0]

            expert_output = self.experts[expert_idx](x_flat[token_indices])
            weights = routing_weights[token_indices, idx_in_topk[1]].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_output * weights)

        if self.num_shared_experts > 0:
            final_output = final_output + shared_output

        return final_output.view(bs, seq_len, -1)

    def update_bias(self, lr: float = 0.05):
        if not self.use_aux_free_lb:
            return

        counts = self.last_expert_counts
        mean_count = counts.mean() + 1e-6

        error = counts - mean_count
        update_step = lr * torch.sign(error)

        self.expert_bias.data -= update_step

        self.expert_bias.data.clamp_(min=-10, max=10)

        self.last_expert_counts_zero_()


