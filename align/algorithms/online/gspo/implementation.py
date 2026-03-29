import torch


def gspo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    attention_mask [group_size, seq_len] 1表示有效token 0表示pad
    """
    actual_lengths = attention_mask.sum(dim=-1)
    log_ratios = (log_probs - old_log_probs) * attention_mask
    log_seq_ratios = log_ratios.sum(dim=-1) / actual_lengths
    seq_ratios = torch.exp(log_seq_ratios)

    surr1 = seq_ratios * advantages
    surr2 = torch.clamp(seq_ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss
