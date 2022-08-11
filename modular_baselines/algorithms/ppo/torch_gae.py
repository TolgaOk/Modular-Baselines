"Differentiable GAE"
from typing import Tuple
import torch


def calculate_diff_gae(rewards: torch.Tensor,
                       terminations: torch.Tensor,
                       values: torch.Tensor,
                       last_value: torch.Tensor,
                       gamma: float,
                       gae_lambda: float,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (rewards.shape == terminations.shape and rewards.shape == values.shape):
        raise ValueError(f"Shape mismatch! rewards: {rewards.shape}, terminations: {terminations.shape}, values: {values.shape}")
    batch_size, rollout_len = values.shape[:2]
    advantages = []
    advantage = torch.zeros_like(values[:, 0])
    assert last_value.shape == advantage.shape, "Shape mismatch"
    for index in reversed(range(rollout_len)):
        reward = rewards[:, index]
        termination = terminations[:, index].float()
        value = values[:, index]

        td_error = last_value * (1 - termination) * gamma + reward - value
        last_value = value
        advantage = advantage * (1 - termination) * gamma * gae_lambda + td_error
        advantages.insert(0, advantage)
    
    advantages = torch.stack(advantages, dim=1)
    returns = advantages + values
    return advantages, returns