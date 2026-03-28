import torch
from typing import Tuple

def compute_advantages(
        reward: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        lambda_gae: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(reward)

    last_advantage = 0

    for t in reversed(range(reward.size(1))):
        next_value = values[:, t + 1] if t < reward.size(1) - 1 else 0.0

        delta = reward[: t] + gamma * next_value - values[t]

        last_advantage = delta + lambda_gae * gamma * last_advantage * mask[:, t]

        advantages[:, t] = last_advantage

    returns = advantages + values

    return advantages, returns

def ppo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        clip_epsilon: float,
) -> torch.Tensor:
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    loss_unmasked = -torch.min(surr1, surr2)
    loss_masked = loss_unmasked * mask

    mask_sum = mask.sum()

    if mask_sum < 1.0:
        return loss_masked.sum() * 0.0

    return loss_masked.sum / mask_sum

