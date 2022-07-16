from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import torch
import numpy as np
from gym.spaces import Space, Box, Discrete


class BaseNetwork(torch.nn.Module):

    def __init__(self, observation_space: Space, action_space: Union[Box, Discrete]):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.in_size = observation_space.shape[0]
        self.out_size = action_space.shape[0]
        if isinstance(action_space, Box):
            self.out_size = self.out_size * 2

    def dist(self, parameters: torch.Tensor) -> Union[torch.distributions.Normal, torch.distributions.Categorical]:
        if isinstance(self.action_space, Box):
            mean, std_logit = parameters.split(parameters.shape[-1] // 2, dim=1)
            std = torch.nn.functional.softplus(std_logit)
            dist = torch.distributions.Normal(loc=mean, scale=std + 0.05)
            dist = torch.distributions.independent.Independent(dist, 1)
            return dist

        if isinstance(self.action_space, Discrete):
            dist = torch.distributions.Categorical(logits=parameters)
            return dist

        raise ValueError(f"Action space: {self.action_space.__class__.__name__} is not supported!")


class SharedFeatureNetwork(BaseNetwork):

    def __init__(self, observation_space: Space, action_space: Space, hidden_size: int = 128):
        super().__init__(observation_space, action_space)

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
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return policy distribution and value
        Args:
            state (torch.Tensor): State tensor
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                policy distribution, value
        """
        feature = self.feature(state)
        logits = self.policy_head(feature)
        value = self.value_head(feature)

        return logits, value


class SeparateFeatureNetwork(BaseNetwork):

    def __init__(self, observation_space: Space, action_space: Space, policy_hidden_size: int = 64, value_hidden_size: int = 64):
        super().__init__(observation_space, action_space)

        self.policy_hidden_size = policy_hidden_size
        self.value_hidden_size = value_hidden_size

        self.policy_net = torch.nn.Sequential(
            linear_layer_init(torch.nn.Linear(self.in_size, policy_hidden_size)),
            torch.nn.Tanh(),
            linear_layer_init(torch.nn.Linear(policy_hidden_size, policy_hidden_size)),
            torch.nn.Tanh(),
            linear_layer_init(torch.nn.Linear(policy_hidden_size, self.out_size), std=0.01)
        )
        self.value_net = torch.nn.Sequential(
            linear_layer_init(torch.nn.Linear(self.in_size, value_hidden_size)),
            torch.nn.Tanh(),
            linear_layer_init(torch.nn.Linear(value_hidden_size, value_hidden_size)),
            torch.nn.Tanh(),
            linear_layer_init(torch.nn.Linear(value_hidden_size, 1), std=1.0)
        )

    def forward(self, state: torch.Tensor, *args
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return policy distribution and value
        Args:
            state (torch.Tensor): State tensor
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                policy distribution, None, value
        """
        logits = self.policy_net(state)
        value = self.value_net(state)
        return logits, value


class LSTMSeparateNetwork(BaseNetwork):

    def __init__(self, observation_space: Space, action_space: Space, policy_hidden_size: int = 64, value_hidden_size: int = 64):
        super().__init__(observation_space, action_space)

        self.policy_hidden_size = policy_hidden_size
        self.value_hidden_size = value_hidden_size

        self.policy_in_layer = linear_layer_init(torch.nn.Linear(self.in_size, policy_hidden_size))
        self.policy_lstm = lstm_init(torch.nn.LSTMCell(policy_hidden_size, policy_hidden_size))
        self.policy_out = linear_layer_init(torch.nn.Linear(
            policy_hidden_size, self.out_size), std=0.01)

        self.value_in_layer = linear_layer_init(torch.nn.Linear(self.in_size, value_hidden_size))
        self.value_lstm = lstm_init(torch.nn.LSTMCell(value_hidden_size, value_hidden_size))
        self.value_out = linear_layer_init(torch.nn.Linear(
            value_hidden_size, 1), std=1.0)

    @property
    def hidden_state_info(self) -> Dict[str, int]:
        return dict(value_hx=self.value_hidden_size,
                    value_cx=self.value_hidden_size,
                    policy_hx=self.policy_hidden_size,
                    policy_cx=self.policy_hidden_size)

    def forward(self,
                state: torch.Tensor,
                hidden_states: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        policy_hx, policy_cx = hidden_states["policy_hx"], hidden_states["policy_cx"]
        value_hx, value_cx = hidden_states["value_hx"], hidden_states["value_cx"]

        policy_features = torch.tanh(self.policy_in_layer(state))
        policy_hx, policy_cx = self.policy_lstm(policy_features, (policy_hx, policy_cx))
        policy_logit = self.policy_out(policy_hx)

        value_features = torch.tanh(self.value_in_layer(state))
        value_hx, value_cx = self.value_lstm(value_features, (value_hx, value_cx))
        value = self.value_out(value_hx)

        hidden_states = dict(value_hx=value_hx, value_cx=value_cx,
                             policy_hx=policy_hx, policy_cx=policy_cx)
        return policy_logit, value, hidden_states


def linear_layer_init(layer: torch.nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def lstm_init(layer: torch.nn.LSTMCell, std: float = 1, bias_const: float = 0.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            torch.nn.init.constant_(param, bias_const)
        elif "weight" in name:
            torch.nn.init.orthogonal_(param, std)
    return layer
