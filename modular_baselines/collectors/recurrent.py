raise DeprecationWarning()
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np

from stable_baselines3.common.vec_env import VecEnv

from modular_baselines.collectors.collector import RolloutCollector, BaseCollectorCallback
from modular_baselines.buffers.buffer import BaseBuffer
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.loggers.data_logger import DataLogger


class ResetHiddenCallback(BaseCollectorCallback):
    """ Reset hidden states of the agent at terminations 
    """

    def on_rollout_start(self, *args) -> None:
        pass

    def on_rollout_end(self, *args) -> None:
        pass

    def on_rollout_step(self, locals_: Dict[str, Any]) -> bool:
        dones = locals_["dones"].reshape(-1, 1).astype(np.float32)
        hidden_states = locals_["self"]._last_hidden_state
        agent = locals_["self"].agent
        n_env = locals_["self"].env.num_envs

        if np.sum(dones) > 0:
            reset_states = agent.init_hidden_state(n_env)
            for name, hidden_state in hidden_states.items():
                hidden_states[name] = hidden_state * (1 - dones) + reset_states[name] * dones


class RecurrentRolloutCollector(RolloutCollector):
    """ Recurrent enabled version of Rollout Collector

    Args:
        env (VecEnv): Vectorized environment
        buffer (BaseBuffer): Buffer to push rollout experiences
        agent (BaseAgent): Action sampling agent
        logger (DataLogger): Data logger to log environment reward and lengths at termination
        callbacks (Optional[Union[List[BaseCollectorCallback], BaseCollectorCallback]], optional): Collector Callback. Defaults to None.
    """

    def __init__(self,
                 env: VecEnv,
                 buffer: BaseBuffer,
                 agent: BaseAgent,
                 logger: DataLogger,
                 store_normalizer_stats: bool = False,
                 callbacks: Optional[Union[List[BaseCollectorCallback], BaseCollectorCallback]] = None):
        # Add Hidden resetting callback for handling terminations
        reset_callback = ResetHiddenCallback()
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        callbacks = [reset_callback, *callbacks]

        self._last_hidden_state = agent.init_hidden_state(batch_size=env.num_envs)
        # Extending buffer fields to include hidden state and next hidden state keys
        self._required_buffer_fields = (
            *self._required_buffer_fields,
            *self._last_hidden_state.keys(),
            *[f"next_{name}" for name in self._last_hidden_state.keys()]
        )
        super().__init__(env, buffer, agent, logger, store_normalizer_stats, callbacks)

    def get_actions(self):
        actions, hidden_states, policy_content = self.agent.sample_action(
            self._last_obs, self._last_hidden_state)
        buffer_info = {**self._last_hidden_state,
                       **{f"next_{key}": value for key, value in hidden_states.items()}}
        self._last_hidden_state = hidden_states
        for name in buffer_info.keys():
            if name in policy_content.keys():
                raise RuntimeError("Policy content conflicts with hidden state keys")
        policy_content.update(buffer_info)
        return actions, policy_content
