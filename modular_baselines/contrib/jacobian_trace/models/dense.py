import torch
from typing import Union, Tuple, Optional


class DenseModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: Tuple[int],
                 layers: int,
                 hidden_size: int,
                 output_shape: Optional[tuple] = None,
                 activation=torch.nn.ELU):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.hidden_size = hidden_size
        self.output_shape = output_shape
        self.activation = activation
        self.model = self.build_model()

    def build_model(self) -> torch.nn.Module:
        model = [torch.nn.Linear(self.input_size, self.hidden_size)]
        model += [self.activation()]
        for _ in range(self.layers - 1):
            model += [torch.nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        model += [torch.nn.Linear(self.hidden_size, self.output_size)]
        return torch.nn.Sequential(*model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.model(features)
        if self.output_shape:
            return output.reshape(output.shape[0], *self.output_shape)
        return output


class GruModel(DenseModel):

    def build_model(self) -> torch.nn.Module:
        self.pre_gru = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            self.activation())
        self.gru = torch.nn.GRUCell(self.hidden_size, self.hidden_size)
        self.output = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        pre_hidden = self.pre_gru(features)
        hidden = self.gru(pre_hidden, hidden)
        if self.output_shape:
            return self.output(hidden).reshape(hidden.shape[0], *self.output_shape), hidden
        return self.output(hidden), hidden

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device=self.output.weight.device)


class NormalDistributionModel(DenseModel):

    def __init__(self,
                 input_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 output_shape: Optional[tuple] = None,
                 activation=torch.nn.ELU,
                 min_std=0.1):
        if output_shape:
            output_shape = (*output_shape[:-1], output_shape[-1] * 2)
        super().__init__(input_size=input_size,
                         output_size=output_size * 2,  # mean and std
                         layers=layers,
                         hidden_size=hidden_size,
                         output_shape=output_shape,
                         activation=activation)
        self.min_std = min_std

    def forward(self, features: torch.Tensor) -> torch.distributions.Normal:
        logits = super().forward(features)
        mean, std_logit = torch.chunk(logits, 2, dim=-1)
        std = torch.torch.nn.functional.softplus(std_logit + self.min_std)
        return torch.distributions.Normal(mean, std)

    @staticmethod
    def chunk_dist(dist: torch.distributions.Normal, chunks: int, dim: int) -> Tuple[
            torch.distributions.Normal, torch.distributions.Normal]:
        means = torch.chunk(dist.loc, chunks, dim=dim)
        stds = torch.chunk(dist.scale, chunks, dim=dim)
        return tuple(torch.distributions.Normal(mean, std) for mean, std in zip(means, stds))

    @staticmethod
    def reparametrize(dist: torch.distributions.Normal,
                      sample: torch.Tensor,
                      epsilon: float = 1e-7):
        mean, std = dist.loc, dist.scale
        return mean + std * ((sample - mean) / (std + epsilon)).detach()


class CategoricalDistributionModel(DenseModel):

    def forward(self, features: torch.Tensor, onehot: bool = False
                ) -> Union[torch.distributions.Categorical,
                           torch.distributions.OneHotCategorical]:
        logit = super().forward(features)
        if onehot:
            return torch.distributions.OneHotCategorical(logits=logit)
        return torch.distributions.Categorical(logits=logit)

    @staticmethod
    def reparametrize(dist: Union[torch.distributions.Categorical,
                                  torch.distributions.OneHotCategorical],
                      sample: torch.Tensor):
        assert sample.shape == dist.probs.shape
        return dist.probs + (sample - dist.probs).detach()

    @staticmethod
    def make_onehot(features: torch.Tensor, size: int):
        onehot = (features.unsqueeze(-1) == torch.arange(
            size,
            device=features.device
        ).reshape(*([1] * len(features.shape)), -1))
        return onehot
