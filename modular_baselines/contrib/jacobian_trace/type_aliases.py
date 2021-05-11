from enum import Enum
import torch
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

class LatentTuple(NamedTuple):
    embedding: torch.Tensor
    dist: torch.distributions.Distribution
    rsample: torch.Tensor

class DiscreteLatentTuple(NamedTuple):
    logit: torch.Tensor
    quantized: torch.Tensor
    encoding: torch.Tensor

class TransitionTuple(NamedTuple):
    latent_tuple: LatentTuple
    action: torch.Tensor
    action_dist: torch.distributions.Distribution

class SequentialRolloutSamples(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminations: torch.Tensor
    rewards: torch.Tensor