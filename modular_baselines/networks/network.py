from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import torch
import numpy as np
from gym.spaces import Space, Box, Discrete


class SharedFeatureNetwork(torch.nn.Module):

    def __init__(self, observation_space: Space, action_space: Space, hidden_size: int = 128):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.in_size = observation_space.shape[0]
        self.out_size = action_space.shape[0]
        if isinstance(action_space, Box):
            self.out_size = self.out_size * 2

        self.hidden_size = hidden_size
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.RReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.RReLU(),
        )

        self.policy_head = torch.nn.Linear(hidden_size, self.out_size)

        torch.nn.init.xavier_normal_(self.policy_head.weight, 0.01)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, *args
                ) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """ Return policy distribution and value
        Args:
            state (torch.Tensor): State tensor
        Returns:
            Tuple[torch.distributions.Normal, torch.Tensor]: 
                policy distribution, value
        """
        feature = self.feature(state)
        logits = self.policy_head(feature)
        value = self.value_head(feature)

        dist = get_dist(logits, self.action_space)
        return dist, value


class SeparateFeatureNetwork(torch.nn.Module):

    def __init__(self, observation_space: Space, action_space: Space, policy_hidden_size: int = 64, value_hidden_size: int = 64):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.in_size = observation_space.shape[0]
        self.out_size = action_space.shape[0]
        if isinstance(action_space, Box):
            self.out_size = self.out_size * 2

        self.policy_hidden_size = policy_hidden_size
        self.value_hidden_size = value_hidden_size

        self.policy_net = torch.nn.Sequential(
            layer_init(torch.nn.Linear(self.in_size, policy_hidden_size)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(policy_hidden_size, policy_hidden_size)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(policy_hidden_size, self.out_size), std=0.01)
        )
        self.value_net = torch.nn.Sequential(
            layer_init(torch.nn.Linear(self.in_size, value_hidden_size)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(value_hidden_size, value_hidden_size)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(value_hidden_size, 1), std=1.0)
        )

    def forward(self, state: torch.Tensor, *args
                ) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """ Return policy distribution and value
        Args:
            state (torch.Tensor): State tensor
        Returns:
            Tuple[torch.distributions.Normal, torch.Tensor]: 
                policy distribution, None, value
        """
        logits = self.policy_net(state)
        value = self.value_net(state)

        dist = get_dist(logits, self.action_space)
        return dist, value


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_dist(logits: torch.Tensor, action_space: Union[Box, Discrete]
             ) -> Union[torch.distributions.Normal, torch.distributions.Categorical]:
    if isinstance(action_space, Box):
        mean, std_logit = logits.split(logits.shape[-1] // 2, dim=1)
        std = torch.nn.functional.softplus(std_logit)
        dist = torch.distributions.Normal(loc=mean, scale=std + 0.05)
        dist = torch.distributions.independent.Independent(dist, 1)
        return dist

    if isinstance(action_space, Discrete):
        dist = torch.distributions.Categorical(logits=logits)
        return dist

    raise ValueError(f"Action space: {action_space.__class__.__name__} is not supported!")
