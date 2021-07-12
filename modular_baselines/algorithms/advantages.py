from os import access
import numpy as np
from typing import Tuple


def calculate_gae(rewards: np.ndarray,
                  terminations: np.ndarray,
                  values: np.ndarray,
                  last_value: np.ndarray,
                  gamma: float,
                  gae_lambda: float,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]

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
    _, rollout_len, _ = rewards.shape
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    advantage = np.zeros_like(returns[:, 0])
    assert last_value.shape == advantage.shape, "Shape mismatch"
    for index in reversed(range(rollout_len)):
        reward = rewards[:, index]
        termination = terminations[:, index]
        value = values[:, index]
        td_error = last_value * (1 - termination) * gamma + reward - value
        last_value = value
        advantage = advantage * (1 - termination) * gamma * gae_lambda + td_error
        advantages[:, index] = advantage
        returns[:, index] = advantage + value
    return advantages, returns
