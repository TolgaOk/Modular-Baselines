from itertools import chain
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from typing import Union

from modular_baselines.contrib.jacobian_trace.type_aliases import DiscreteLatentTuple
from modular_baselines.contrib.jacobian_trace.models.vqvae import VectorQuantizer
from modular_baselines.contrib.jacobian_trace.models.dense import DenseModel, CategoricalDistributionModel


class EmbeddingModule(torch.nn.Module):

    def __init__(self, n_embeddings: int, embedding_size: int):
        super().__init__()
        self.embedding = VectorQuantizer(n_embeddings, embedding_size)


class ObservationDecoder(torch.nn.Module):

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, latent_tuple: DiscreteLatentTuple):
        features = self.make_feature(latent_tuple)
        if isinstance(self.model, DenseModel):
            features = features.reshape(features.shape[0], -1)
        return self.model(features)

    @staticmethod
    def make_feature(latent_tuple: DiscreteLatentTuple):
        return latent_tuple.quantized


class ObservationEncoder(EmbeddingModule):

    def __init__(self,
                 model: torch.nn.Module,
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__(n_embeddings, embedding_size)
        self.model = model
        if isinstance(self.model, DenseModel):
            assert self.model.output_shape is not None

    def forward(self, observation: torch.Tensor):
        features = self.model(observation)
        return self.embedding(features)


class DiscreteTransitionDistribution(EmbeddingModule):

    def __init__(self,
                 model: Union[DenseModel],
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__(n_embeddings, embedding_size)
        self.model = model
        if isinstance(self.model, DenseModel):
            assert self.model.output_shape is not None
        self.forget_gate = DenseModel(model.input_size,
                                      np.product(model.output_shape[:-1]),
                                      1,
                                      model.hidden_size,
                                      output_shape=model.output_shape[:-1])

    def forward(self, latent_tuple: DiscreteLatentTuple,
                actions: torch.Tensor,
                prev_logits: torch.Tensor):
        if len(actions.shape) == 1:
            actions = CategoricalDistributionModel.make_onehot(
                actions, size=self.model.output_shape[-1]).float()
        features = self.make_feature(latent_tuple)

        attention = torch.sigmoid(self.forget_gate(features))
        logits = self.model(features)

        logits = torch.einsum("b...a,ba->b...", logits, actions)
        dist_logits = logits * attention + (1 - attention) * prev_logits

        dist = torch.distributions.OneHotCategorical(logits=dist_logits)
        return dist

    @staticmethod
    def reparam(dist: torch.distributions.OneHotCategorical, sample: torch.Tensor):
        return CategoricalDistributionModel.reparametrize(dist, sample)

    @staticmethod
    def log_prob(dist: torch.distributions.OneHotCategorical, sample: torch.Tensor):
        return dist.log_prob(sample).reshape(sample.shape[0], -1).sum(-1)

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        return latent_tuple.encoding.reshape(latent_tuple.encoding.shape[0], -1)


class DiscreteActor(EmbeddingModule):

    def __init__(self,
                 model: Union[DenseModel],
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__(n_embeddings, embedding_size)
        self.model = model

    def forward(self, latent_tuple: DiscreteLatentTuple, onehot: bool = False):
        features = self.make_feature(latent_tuple)
        act_logits = self.model(features)
        if onehot:
            return torch.distributions.one_hot_categorical.OneHotCategorical(logits=act_logits)
        return torch.distributions.categorical.Categorical(logits=act_logits)

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        encoding = latent_tuple.encoding
        return self.embedding.quantize(encoding).reshape(encoding.shape[0], -1)

    def make_onehot(self, features: torch.Tensor):
        onehot_size = self.model.output_shape[-1] if self.model.output_shape else self.model.output_size
        return CategoricalDistributionModel.make_onehot(features, onehot_size)

    @staticmethod
    def reparam(dist: torch.distributions.OneHotCategorical, sample: torch.Tensor):
        return CategoricalDistributionModel.reparametrize(dist, sample)


class Critic(EmbeddingModule):

    def __init__(self,
                 model: Union[DenseModel],
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__(n_embeddings, embedding_size)
        self.model = model

    def forward(self, latent_tuple: DiscreteLatentTuple):
        features = self.make_feature(latent_tuple)
        value = self.model(features)
        return value

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        encoding = latent_tuple.encoding
        return self.embedding.quantize(encoding).reshape(encoding.shape[0], -1)


class DiscreteJTACModel(torch.nn.Module):

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 transition_model,
                 actor_model,
                 critic_model,
                 model_lr: float = 4e-4,
                 actor_lr: float = 4e-4,
                 critic_lr: float = 4e-4,
                 n_embeddings: int = 128,
                 embedding_size: int = 32):
        super().__init__()
        self.encoder = ObservationEncoder(
            model=encoder_model,
            n_embeddings=n_embeddings,
            embedding_size=embedding_size)
        self.decoder = ObservationDecoder(
            model=decoder_model)
        self.transition_dist = DiscreteTransitionDistribution(
            model=transition_model,
            n_embeddings=n_embeddings,
            embedding_size=embedding_size)
        self.actor = DiscreteActor(
            model=actor_model,
            n_embeddings=n_embeddings,
            embedding_size=embedding_size)
        self.critic = Critic(
            model=critic_model,
            n_embeddings=n_embeddings,
            embedding_size=embedding_size)

        self.model_optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(),
                  self.decoder.parameters(),
                  self.transition_dist.parameters()),
            lr=model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_latent(self, observation: torch.Tensor):
        observation = self._preprocess(observation)
        return self.encoder(observation)

    def _preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def _forward(self, observation: torch.Tensor):
        latent_tuple, *_ = self.get_latent(observation)

        values = self.critic(latent_tuple)
        actor_dist = self.actor(latent_tuple, onehot=False)
        return actor_dist, values, latent_tuple

    def forward(self, observation: torch.Tensor):
        actor_dist, values, latent_tuple = self._forward(observation)

        action = actor_dist.sample()
        log_prob = actor_dist.log_prob(action)
        return action, values, log_prob

    # Must have
    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor):
        actor_dist, values, latent_tuple = self._forward(observation)

        entropy = actor_dist.entropy()
        log_prob = actor_dist.log_prob(action)
        return values, log_prob, entropy

    # Must have
    def loss(self,
             observations: torch.Tensor,
             next_observations: torch.Tensor,
             actions: torch.Tensor):
        combined_obs = torch.cat([observations, next_observations], dim=0)
        combined_latent, q_latent_loss, e_latent_loss, perplexity = self.get_latent(combined_obs)
        pred_obs_dist = self.decoder(combined_latent)

        latent_tuple, next_latent_tuple = [DiscreteLatentTuple(*half_tensor) for half_tensor in zip(
            *[torch.chunk(tensor, 2, dim=0) for tensor in combined_latent])]
        trans_dist = self.transition_dist(latent_tuple, actions)
        transition_loss = -self.transition_dist.log_prob(trans_dist, next_latent_tuple.encoding)
        transition_loss = transition_loss.reshape(observations.shape[0], -1).sum(1).mean(0)

        recon_loss = -pred_obs_dist.log_prob(combined_obs).sum(1).mean(0)

        return recon_loss, transition_loss, e_latent_loss, q_latent_loss, perplexity