from typing import List, Any, Dict, Union, Optional, Tuple, Callable
import numpy as np
from gymnasium.spaces import Space, Box, Discrete

import flax
import optax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class TrainState(flax.struct.PyTreeNode):
  step: int
  apply_fn: Callable = flax.struct.field(pytree_node=False)
  params: flax.core.FrozenDict[str, Any]
  tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
  opt_state: optax.OptState

  def apply_gradients(self, *, grads, **kwargs):
    # print(type(grads), type(self.opt_state), type(self.params))
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )


class PolicyNet(nn.Module):
    in_size: int
    output_size: int
    policy_hidden_size: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        h1 = nn.Dense(self.policy_hidden_size, kernel_init = orthogonal(np.sqrt(2)))(state)
        h1 = nn.tanh(h1)
        h2 = nn.Dense(self.policy_hidden_size, kernel_init = orthogonal(np.sqrt(2)))(h1)
        h2 = nn.tanh(h2)
        output = nn.Dense(self.output_size, kernel_init = orthogonal(np.sqrt(2)), bias_init = constant(1.0))(h2)

        return output


class ValueNet(nn.Module):
    in_size: int
    output_size: int
    value_hidden_size: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        h1 = nn.Dense(self.value_hidden_size, kernel_init = orthogonal(np.sqrt(2)))(state)
        h1 = nn.tanh(h1)
        h2 = nn.Dense(self.value_hidden_size, kernel_init = orthogonal(np.sqrt(2)))(h1)
        h2 = nn.tanh(h2)
        output = nn.Dense(1, kernel_init = orthogonal(np.sqrt(2)), bias_init = constant(1.0))(h2)

        return output
        

class SeparateFeatureNetwork(nn.Module):
    in_size: int
    out_size: int
    policy_hidden_size: int
    value_hidden_size: int
    observation_space: Space
    action_space: Space

    def setup(self):
        self.policy_net = PolicyNet(self.in_size, self.out_size, self.policy_hidden_size)
        self.value_net = ValueNet(self.in_size, self.out_size, self.value_hidden_size)

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        policy_output = self.policy_net(state)
        value_output = self.value_net(state)

        return policy_output, value_output

        