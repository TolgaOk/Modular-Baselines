import numpy as np

from gym.spaces import (Box,
                        Discrete)
from gym import (ObservationWrapper,
                 ActionWrapper,
                 Wrapper)


class NormalizeObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        self.unnormalized_high = self.observation_space.high
        self.unnormalized_low = self.observation_space.low
        assert np.all(self.unnormalized_high > self.unnormalized_low).item()
        self.observation_space = Box(low=0, high=1,
                                     shape=self.observation_space.shape,
                                     dtype=np.float32)

    def observation(self, observation):

        observation = observation.astype(self.observation_space.dtype)
        observation = (observation - self.unnormalized_low) / \
            (self.unnormalized_high - self.unnormalized_low)
        return observation


class SkipSteps(Wrapper):
    def __init__(self, env, n_skip):
        super().__init__(env)
        self.n_skip = n_skip
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        noops = self.n_skip if self.n_skip > 0 else 0
        for _ in range(noops):
            state, _, done, _ = self.env.step(0)
            if done:
                state = self.env.reset(**kwargs)
        return state


class AggregateObservation(ObservationWrapper):
    def __init__(self, env, aggr_indexes):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 1
        self.aggr_indexes = aggr_indexes
        dtype = self.observation_space.dtype
        low = self.observation_space.low[0]
        high = self.observation_space.high[0]
        length = self.observation_space.shape[0]
        self.observation_space = Box(
            low=low, high=high, shape=((len(aggr_indexes) + length),), dtype=dtype)

    def observation(self, observation):
        new_observation = np.concatenate([observation, self.prev_obs])
        self.prev_obs = observation[self.aggr_indexes]
        return new_observation

    def _reset_prev_obs(self):
        self.prev_obs = np.zeros(shape=len(self.aggr_indexes))

    def reset(self, **kwargs):
        self._reset_prev_obs()
        return super().reset(**kwargs)


class IndexObsevation(ObservationWrapper):
    def __init__(self, env, obs_indexes):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 1
        self.obs_indexes = obs_indexes
        dtype = self.observation_space.dtype
        low = self.observation_space.low[0]
        high = self.observation_space.high[0]
        self.observation_space = Box(
            low=low, high=high, shape=(len(obs_indexes),), dtype=dtype)

    def observation(self, observation):
        return observation[self.obs_indexes]


class IndexAction(ActionWrapper):
    def __init__(self, env, act_indexes):
        super().__init__(env)
        assert isinstance(self.action_space, Discrete)
        self.act_indexes = act_indexes

        self.action_space = Discrete(n=len(act_indexes))

    def action(self, action):
        return self.act_indexes[action]


class ResetWithNonZeroReward(Wrapper):
    def step(self, *args, **kwargs):
        state, reward, done, info = super().step(*args, **kwargs)
        if reward != 0:
            done = True
        return state, reward, done, info
