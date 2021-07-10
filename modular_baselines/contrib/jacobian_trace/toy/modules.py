from itertools import chain
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from typing import Union, Tuple
from functools import partial
from copy import deepcopy

from modular_baselines.contrib.jacobian_trace.type_aliases import DiscreteLatentTuple
from modular_baselines.contrib.jacobian_trace.models.vqvae import VectorQuantizer
from modular_baselines.contrib.jacobian_trace.models.dense import DenseModel, CategoricalDistributionModel
from modular_baselines.contrib.jacobian_trace.models.discrete_latent import EmbeddingModule
from modular_baselines.contrib.jacobian_trace.models.discrete_latent import Critic


class DiscreteTransitionDistribution(EmbeddingModule):

    def __init__(self,
                 model: Union[DenseModel],
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__(n_embeddings, embedding_size)
        self.model = model
        if isinstance(self.model, DenseModel):
            assert self.model.output_shape is not None
        # self.forget_gate = deepcopy(self.model)

    def forward(self,
                latent_tuple: DiscreteLatentTuple,
                actions: Tuple[torch.Tensor],
                # hidden: torch.Tensor
                ):
        onehot_actions = []
        for index, action in enumerate(actions):
            if len(action.shape) == 1:
                action = CategoricalDistributionModel.make_onehot(
                    action, size=self.model.output_shape[-len(actions) + index]).float()
            onehot_actions.append(action)

        features = self.make_feature(latent_tuple)

        output_logits = self.model(features)
        # forget_logit = self.forget_gate(features)

        for action in reversed(onehot_actions):
            output_logits = torch.einsum("b...a,ba->b...", output_logits, action)
            # forget_logit = torch.einsum("b...a,ba->b...", forget_logit, action)
    
        # forget_gate = torch.sigmoid(forget_logit)
        # output_logits = output_logits * forget_gate + (1 - forget_gate) * prev_logits
        
        dist = torch.distributions.OneHotCategorical(logits=output_logits)
        return dist

    @ staticmethod
    def reparam(dist: torch.distributions.OneHotCategorical, sample: torch.Tensor):
        return CategoricalDistributionModel.reparametrize(dist, sample)

    @ staticmethod
    def log_prob(dist: torch.distributions.OneHotCategorical, sample: torch.Tensor):
        return dist.log_prob(sample).reshape(sample.shape[0], -1).sum(-1)

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        encoding = latent_tuple.encoding
        return encoding.reshape(encoding.shape[0], -1)
        


class DiscreteActor(EmbeddingModule):

    def __init__(self,
                 model: Union[DenseModel],
                 nvec: Tuple[int],
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__(n_embeddings, embedding_size)
        self.model = model
        self.nvec = nvec

    def forward(self, latent_tuple: DiscreteLatentTuple, onehot: bool = False):
        features = self.make_feature(latent_tuple)
        logits = self.model(features)
        act_logits = torch.split(logits, self.nvec, dim=-1)
        if onehot:
            return tuple(torch.distributions.one_hot_categorical.OneHotCategorical(logits=logit)
                         for logit in act_logits)
        return tuple(torch.distributions.categorical.Categorical(logits=logit)
                     for logit in act_logits)

    def make_feature(self, latent_tuple: DiscreteLatentTuple):
        encoding = latent_tuple.encoding
        return self.embedding.quantize(encoding).reshape(encoding.shape[0], -1)

    def make_onehot(self, actions: Tuple[torch.Tensor]):
        return tuple(CategoricalDistributionModel.make_onehot(action, size)
                     for action, size in zip(actions, self.nvec))

    @ staticmethod
    def reparam(dists: Tuple[torch.distributions.OneHotCategorical], samples: Tuple[torch.Tensor]):
        return tuple(CategoricalDistributionModel.reparametrize(dist, sample)
                     for dist, sample in zip(dists, samples))


class EncodedJTACModel(torch.nn.Module):

    def __init__(self,
                 transition_model,
                 actor_model,
                 critic_model,
                 model_lr: float = 4e-4,
                 actor_lr: float = 4e-4,
                 critic_lr: float = 4e-4,
                 n_embeddings: int = 128,
                 embedding_size: int = 32):
        super().__init__()
        self.transition_dist = DiscreteTransitionDistribution(
            model=transition_model,
            n_embeddings=n_embeddings,
            embedding_size=embedding_size)
        self.actor = DiscreteActor(
            model=actor_model,
            nvec=transition_model.output_shape[-2:],
            n_embeddings=n_embeddings,
            embedding_size=embedding_size)
        self.critic = Critic(
            model=critic_model,
            n_embeddings=n_embeddings,
            embedding_size=embedding_size)

        self._init_weights()

        self.model_optimizer = torch.optim.Adam(self.transition_dist.parameters(), lr=model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.optimizer = torch.optim.Adam(chain(self.actor.parameters(), self.critic.parameters()), lr=actor_lr)

    def _init_weights(self, common_gain=np.sqrt(2)):
        module_gains = {
            self.transition_dist: 1,
            self.actor: 0.01,
            self.critic: 1,
        }
        for module, gain in module_gains.items():
            module.apply(lambda _module: self._apply_weight(_module, gain, common_gain))

    def _apply_weight(self, module, last_gain, common_gain):
        if isinstance(module, DenseModel):
            layers = tuple(module.children())[0]
            for index, layer in enumerate(layers):
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    torch.nn.init.xavier_uniform_(
                        layer.weight,
                        gain=common_gain if index < (len(layers) - 1 ) else last_gain)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)

    def _preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def _forward(self, observation: torch.Tensor):
        latent_tuple = DiscreteLatentTuple(None, None, observation)

        values = self.critic(latent_tuple)
        actor_dists = self.actor(latent_tuple, onehot=False)
        return actor_dists, values

    def forward(self, observation: torch.Tensor):
        actor_dists, values = self._forward(observation)

        actions = [dist.sample() for dist in actor_dists]
        log_prob = sum(dist.log_prob(action) for dist, action in zip(actor_dists, actions))
        actions = torch.stack(actions, dim=-1)
        return actions, values, log_prob

    def evaluate_actions(self, observation: torch.Tensor, actions: torch.Tensor):
        actor_dists, values = self._forward(observation)
        actions = torch.unbind(actions, dim=-1)
        log_prob = sum(dist.log_prob(action) for dist, action in zip(actor_dists, actions))
        entropy = sum(dist.entropy() for dist in actor_dists)
        return values, log_prob, entropy
