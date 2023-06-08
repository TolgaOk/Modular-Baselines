from typing import Optional, Any, Dict, Tuple, Union, List, Callable
from abc import ABC, abstractmethod
import flax.linen as nn
import jax
import distrax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Space, Discrete, Box
from flax.training import train_state

from modular_baselines.loggers.logger import MBLogger
from modular_baselines.utils.utils import to_jax


# TODO: Replace Base class with Protocol
class BaseAgent(ABC):

    def __init__(self, observation_space: Space, action_space: Space, logger: MBLogger) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.logger = logger

    @abstractmethod
    def sample_action(self,
                      observation: np.ndarray,
                      policy_state: Union[np.ndarray, None],
                      ) -> Tuple[np.ndarray, Union[np.ndarray, None], Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass


class JaxAgent(BaseAgent):
    # TODO: Add discrete action space support

    def __init__(self,
                 network: nn.Module,
                 observation_space: Space,
                 action_space: Space,
                 logger: MBLogger,
                 rng_seed: Optional[int] = None,
                 ) -> None:
        self.network = network
        self.rng = jax.random.PRNGKey(rng_seed)
        super().__init__(observation_space, action_space, logger)

    def forward(self,
                observation: jnp.ndarray,
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        policy_params, value = self.network.apply_fn({'params': self.network.params}, observation)
        return policy_params, value

    def dist(self,
             parameters: jnp.ndarray
             ) -> Union[distrax.Normal, distrax.Categorical]:
        if isinstance(self.action_space, Box):
            mean, std_logit = parameters.split(indices_or_sections=2, axis=1)
            std = jax.nn.softplus(std_logit)
            dist = distrax.Normal(loc=mean, scale=std + 0.05)
            dist = distrax.Independent(dist, 1)
            return dist

        if isinstance(self.action_space, Discrete):
            dist = distrax.Categorical(logits=parameters)
            return dist

        raise ValueError(f"Action space: {self.action_space.__class__.__name__} is not supported!")

    def sample_action(self,
                      observation: np.ndarray,
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        self.rng, _rng = jax.random.split(self.rng)

        jx_observation = to_jax(observation)

        policy_params, _ = self.forward(jx_observation)
        policy_dist = self.dist(policy_params)
        jx_action = policy_dist.sample(seed=_rng)
        log_prob = policy_dist.log_prob(jx_action).unsqueeze(-1)
        if isinstance(self.action_space, Discrete):
            # th_action = th_action.unsqueeze(-1)
            raise NotImplementedError(
                f"Unsupported action space distribution {self.action_space.__class__.__name__}!")
        return jx_action.numpy(), {"old_log_prob": log_prob.numpy()}
    

    def save(self, path: str) -> None:
        # TODO: UPDATE THIS PART.
        jax.numpy.save({
            "agent_state_dict": self.network.state_dict(),
        }, path)
