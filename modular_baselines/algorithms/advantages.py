from typing import Tuple
import numpy as np


def calculate_gae(rewards: np.ndarray,
                  terminations: np.ndarray,
                  values: np.ndarray,
                  last_value: np.ndarray,
                  gamma: float,
                  gae_lambda: float,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate Genearlized Advantages Estimation

    Args:
        rewards (np.ndarray): [description]
        terminations (np.ndarray): [description]
        values (np.ndarray): [description]
        last_value (np.ndarray): [description]
        gamma (float): [description]
        gae_lambda (float): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """
    rewards, terminations, values, last_value = map(
        lambda array: array.astype(np.float32),
        (rewards, terminations, values, last_value)
    )
    _, rollout_len, _ = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    advantage = np.zeros_like(advantages[:, 0])
    assert last_value.shape == advantage.shape, "Shape mismatch"
    for index in reversed(range(rollout_len)):
        reward = rewards[:, index].astype(np.float32)
        termination = terminations[:, index].astype(np.float32)
        value = values[:, index].astype(np.float32)
        td_error = last_value * (1 - termination) * gamma + reward - value
        last_value = value
        advantage = advantage * (1 - termination) * gamma * gae_lambda + td_error
        advantages[:, index] = advantage
    returns = advantages + values
    return advantages, returns
