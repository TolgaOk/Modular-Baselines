from typing import List, Any, Dict, Union, Optional, Tuple, Callable, Type
import torch
from abc import abstractmethod
import geotorch
import numpy as np


class AggregateState():

    def __init__(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
        super().__init__()
        self.state = state
        self.next_state = next_state

    def merge(self) -> torch.Tensor:
        batch_size, rollout_length, *state_dims = self.state.shape
        return torch.cat([
            self.state.reshape(batch_size * rollout_length, *state_dims),
            self.next_state.reshape(batch_size * rollout_length, *state_dims)
        ], dim=0)

    def split(self, merged_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, rollout_length, *_ = self.state.shape
        embed_dims = merged_tensor.shape[1:]
        state, next_state = merged_tensor.split(batch_size * rollout_length, dim=0)
        return (
            state.reshape(batch_size, rollout_length, *embed_dims),
            next_state.reshape(batch_size, rollout_length, *embed_dims)
        )


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

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 200, n_layers: int = 2) -> None:
        super().__init__(state_size, action_size)
        if action_size > hidden_size:
            raise ValueError(
                f"Action size: {action_size} can not be higher than hidden size: {hidden_size}")
        if state_size > hidden_size:
            raise ValueError(
                f"State size: {state_size} can not be higher than hidden size: {hidden_size}")

        self.hidden_size = hidden_size

        self.encoding = self._make_network(
            in_size=state_size,
            out_size=hidden_size,
            n_hiddens=1,
            hidden_size=hidden_size,
            non_linearity=lambda: torch.nn.LeakyReLU(negative_slope=0.05),
            use_final_nonlinearity=True,
            use_orthogonal_weights=True)
        self.steifel_hidden = self._make_network(
            in_size=hidden_size,
            out_size=hidden_size,
            n_hiddens=n_layers,
            hidden_size=hidden_size,
            non_linearity=lambda: ModRelu(hidden_size),
            use_final_nonlinearity=False,
            use_orthogonal_weights=True
        )

        self.layer_action_input = torch.nn.Linear(action_size, hidden_size, bias=True)
        geotorch.orthogonal(self.layer_action_input, "weight")

        self.std_parameters = torch.nn.Parameter(torch.zeros(1, hidden_size))

    def immersion(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoding(state)

    def submersion(self, state_embed: torch.Tensor) -> torch.Tensor:
        return self.inverse(state_embed, self.encoding)

    def inverse(self, output: torch.Tensor, layers: torch.nn.Sequential) -> torch.Tensor:
        for layer in reversed(layers):
            if (layer.__class__.__name__ == "ParametrizedLinear"):
                if layer.bias is not None:
                    output = output - layer.bias
                output = output @ layer.weight
            elif isinstance(layer, torch.nn.Tanh):
                output = torch.atanh(output)
            elif isinstance(layer, torch.nn.LeakyReLU):
                output = torch.nn.functional.leaky_relu(output, negative_slope=1/layer.negative_slope)
            else:
                raise RuntimeError(f"Unknown layer: {layer.__class__.__name__}")
        return output

    def forward(self, state_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feature = torch.tanh(self.layer_action_input(action)) + state_embed #self.state_activation(state_embed)
        hidden = self.steifel_hidden(feature)

        std_parameters = torch.exp(self.std_parameters.expand_as(hidden))
        return torch.cat([hidden, std_parameters], dim=-1)

    def dist(self, parameters: torch.Tensor) -> torch.distributions.Distribution:
        mean, std = parameters.split(parameters.shape[-1] // 2, dim=-1)
        dist = torch.distributions.Normal(loc=mean, scale=std + 0.05)
        dist = torch.distributions.independent.Independent(dist, 1)
        return dist

    def _make_network(self,
                      in_size: int,
                      out_size: int,
                      n_hiddens: int,
                      hidden_size: int,
                      non_linearity: Type[torch.nn.Module],
                      use_final_nonlinearity: bool,
                      use_orthogonal_weights: bool) -> torch.nn.Sequential:
        layers = []
        hidden_sizes = ([hidden_size] * n_hiddens)
        for (in_s, out_s) in zip([in_size, *hidden_sizes], [*hidden_sizes, out_size]):
            layer = torch.nn.Linear(in_s, out_s)
            if use_orthogonal_weights:
                geotorch.orthogonal(layer, "weight")
            layers.append(layer)
            layers.append(non_linearity())
        if not use_final_nonlinearity:
            layers.pop()
        return torch.nn.Sequential(*layers)


