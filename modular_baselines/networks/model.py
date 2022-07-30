from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import torch
import numpy as np


class ModelNetwork(torch.nn.Module):

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.state_size=state_size
        self.action_size=action_size

        self.layer_state_input = torch.nn.Linear(state_size, 64)
        self.layer_action_input = torch.nn.Linear(action_size, 64)
        self.hidden_layer = torch.nn.Linear(128, 128)
        self.output_layer = torch.nn.Linear(128, state_size * 2)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_feature = self.layer_state_input(state)
        action_feature = self.layer_action_input(action)
        feature = torch.cat([state_feature, action_feature], dim=-1)
        feature = self.hidden_layer(feature)
        parameters = self.output_layer(feature)
    
        return parameters

    def dist(self, parameters: torch.Tensor) -> torch.distributions.Distribution:
        mean, std_logit = parameters.split(parameters.shape[-1] // 2, dim=-1)
        std = torch.nn.functional.softplus(std_logit)
        dist = torch.distributions.Normal(loc=mean, scale=std + 0.01)
        dist = torch.distributions.independent.Independent(dist, 1)
        return dist
