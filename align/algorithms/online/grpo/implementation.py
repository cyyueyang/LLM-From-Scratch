import torch

def grpo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_epsilon: float,
        beta: float,
) -> torch.Tensor:
    """
    R1 论文中， 是序列级别的奖励，序列级别的重要性采样
    """
    seq_log_ratios = ((log_probs - old_log_probs) * attention_mask).sum(dim=-1)
    seq_ratios = torch.exp(seq_log_ratios)

    ref_seq_log_ratios = ((ref_log_probs - log_probs) * attention_mask).sum(dim=-1)
    ref_seq_ratios = torch.exp(ref_seq_log_ratios)

    surr1 = seq_ratios * advantages
    surr2 = torch.clamp(seq_ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    policy_loss = (-torch.min(surr1, surr2) + beta * (ref_seq_ratios - torch.log(ref_seq_ratios) - 1)).mean()
    
    return policy_loss
