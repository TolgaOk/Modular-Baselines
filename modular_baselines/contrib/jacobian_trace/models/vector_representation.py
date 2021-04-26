import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class DenseModel(nn.Module):
    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 activation=nn.ELU):
        super().__init__()
        self._output_size = output_size
        self._layers = layers
        self._hidden_size = hidden_size
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for _ in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, features):
        return self.model(features)


class DistributionModel(DenseModel):

    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 layers: int,
                 hidden_size: int,
                 activation=nn.ReLU):
        super().__init__(feature_size,
                         output_size * 2,
                         layers,
                         hidden_size,
                         activation)

    def forward(self, features):
        mean, std = torch.chunk(super().forward(features), 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = torch.distributions.Normal(mean, std)
        return dist

    def reparam(self, dist, sample):
        return dist.loc + dist.scale * ((sample - dist.loc)/dist.scale).detach()


class ObservationEncoder(DenseModel):
    pass


class ObservationDecoder(DistributionModel):
    pass


class StateDistribution(DistributionModel):
    pass


class TransitionDistribution(DenseModel):

    def __init__(self,
                 feature_size: int,
                 output_size: tuple,
                 action_size: int,
                 layers: int,
                 hidden_size: int,
                 activation=nn.ReLU):
        self._action_size = action_size
        super().__init__(feature_size,
                         output_size * 2 * action_size,
                         layers,
                         hidden_size,
                         activation)

    def forward(self, features, actions):
        features = super().forward(features).reshape(features.shape[0], self._action_size, -1)

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)
        action_indexes = actions.unsqueeze(-1).repeat_interleave(features.shape[-1], dim=-1)
        selected_features = features.gather(dim=1, index=action_indexes).squeeze(1)

        mean, std = torch.chunk(selected_features, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1
        dist = torch.distributions.Normal(mean, std)
        return dist


class Actor(DenseModel):
    pass


class Critic(DenseModel):
    pass


class JTACModel(torch.nn.Module):

    def __init__(self,
                 input_size,
                 action_size,
                 lr=4e-4,
                 state_size=32,
                 latent_size=200,
                 encoder_layers=2,
                 encoder_hidden_size=200,
                 decoder_layers=2,
                 decoder_hidden_size=200,
                 state_layers=2,
                 state_hidden_size=200,
                 trans_layers=2,
                 trans_hidden_size=200,
                 actor_layers=1,
                 actor_hidden_size=200,
                 critic_layers=1,
                 critic_hidden_size=200):
        super().__init__()
        self.encoder = ObservationEncoder(input_size,
                                          latent_size,
                                          encoder_layers,
                                          encoder_hidden_size)
        self.decoder = ObservationDecoder(latent_size + state_size,
                                          input_size,
                                          decoder_layers,
                                          decoder_hidden_size)
        self.state_dist = StateDistribution(latent_size,
                                            state_size,
                                            state_layers,
                                            state_hidden_size)
        self.transition_dist = TransitionDistribution(latent_size + state_size,
                                                 state_size,
                                                 action_size,
                                                 trans_layers,
                                                 trans_hidden_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.actor = Actor(state_size + latent_size,
                           action_size,
                           actor_layers,
                           actor_hidden_size)
        self.critic = Critic(state_size + latent_size,
                             1,
                             critic_layers,
                             critic_hidden_size)

    def encode(self, observation):
        return self.encoder(observation)

    def make_state_dist(self, logits):
        return self.state_dist(logits)

    def make_transition_dist(self, logits, rstate, actions):
        return self.transition_dist(torch.cat([logits, rstate], dim=-1), actions)

    def make_feature(self, logits, rstate):
        return torch.cat([logits, rstate], dim=-1)

    def forward(self, observation):
        logits = self.encoder(observation)
        state_dist = self.state_dist(logits)

        rstate = state_dist.rsample()
        return logits, state_dist, rstate

    @staticmethod
    def batch_chunk_distribution(distribution):
        prev_loc, next_loc = torch.chunk(distribution.loc, 2, dim=0)
        prev_scale, next_scale = torch.chunk(distribution.scale, 2, dim=0)
        return (torch.distributions.Normal(prev_loc, prev_scale),
                torch.distributions.Normal(next_loc, next_scale))

    @staticmethod
    def make_normal_prior(distribution):
        return torch.distributions.Normal(
            torch.zeros_like(distribution.loc),
            torch.ones_like(distribution.scale))
