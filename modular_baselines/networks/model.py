from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import torch
from abc import abstractmethod
import geotorch
import numpy as np


class BaseModel(torch.nn.Module):

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

    @abstractmethod
    def immersion(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def submersion(self, embedding: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dist(parameters: torch.Tensor) -> torch.distributions.Distribution:
        pass


class ModelNetwork(BaseModel):

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__(state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size

        self.layer_state_input = torch.nn.Linear(state_size, 64)
        self.layer_action_input = torch.nn.Linear(action_size, 64)
        self.hidden_layer = torch.nn.Linear(128, 128)
        self.output_layer = torch.nn.Linear(128, state_size * 2)

    def immersion(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def submersion(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding

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


class ModRelu(torch.nn.Module):

    def __init__(self, features: int):
        super().__init__()
        self.features = features
        self.b = torch.nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # return torch.abs(inputs) + self.b
        # return inputs
        return self._forward(inputs, self.b)

    @staticmethod
    def _forward(inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        norm = torch.abs(inputs)
        biased_norm = norm + bias
        magnitude = torch.nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class StiefelNetwork(BaseModel):

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, n_layers: int = 2) -> None:
        super().__init__(state_size, action_size)
        if action_size > hidden_size:
            raise ValueError(
                f"Action size: {action_size} can not be higher than hidden size: {hidden_size}")
        if state_size > hidden_size:
            raise ValueError(
                f"State size: {state_size} can not be higher than hidden size: {hidden_size}")

        self.hidden_size = hidden_size

        self.layer_state_input = torch.nn.Linear(state_size, hidden_size, bias=False)
        geotorch.orthogonal(self.layer_state_input, "weight")

        self.layer_action_input = torch.nn.Linear(action_size, hidden_size, bias=True)
        geotorch.orthogonal(self.layer_action_input, "weight")

        self.std_parameters = torch.nn.Parameter(torch.zeros(1, hidden_size))

        self.state_activation = ModRelu(hidden_size)

        layers = []
        for index in range(n_layers):
            hidden_layer = torch.nn.Linear(hidden_size, hidden_size, bias=True)
            geotorch.orthogonal(hidden_layer, "weight")
            layers.append(hidden_layer)
            if index != n_layers - 1:
                layers.append(ModRelu(hidden_size))

        self.hidden_layers = torch.nn.Sequential(*layers)

    def immersion(self, state: torch.Tensor) -> torch.Tensor:
        return self.layer_state_input(state)

    def submersion(self, state_embed: torch.Tensor) -> torch.Tensor:
        return state_embed @ self.layer_state_input.weight

    def forward(self, state_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feature = self.layer_action_input(torch.tanh(action)) + self.state_activation(state_embed)
        hidden = self.hidden_layers(feature)

        std_parameters = torch.exp(self.std_parameters.expand_as(hidden))
        return torch.cat([hidden, std_parameters], dim=-1)

    def dist(self, parameters: torch.Tensor) -> torch.distributions.Distribution:
        mean, std = parameters.split(parameters.shape[-1] // 2, dim=-1)
        dist = torch.distributions.Normal(loc=mean, scale=std + 0.01)
        dist = torch.distributions.independent.Independent(dist, 1)
        return dist

