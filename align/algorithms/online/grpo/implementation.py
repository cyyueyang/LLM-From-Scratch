import torch

def grpo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float,
) -> torch.Tensor:
    ratios = torch.exp(log_probs - old_log_probs)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    policy_loss = -torch.min(surr1, surr2)
    
    return policy_loss
