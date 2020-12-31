import numpy as np
import gym


class WalkEnv(gym.Env):
    def __init__(self, n_state=3):
        assert n_state % 2 == 1, ""
        self.n_state = n_state
        self.state_set = np.arange(n_state) - (self.n_state // 2)
        self.reward_set = np.array([-1, 0, 1])
        self.observation_space = gym.spaces.Discrete(n=n_state)
        self.action_space = gym.spaces.Discrete(n=2)
        self.state = None

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Reset is not called")
        state = self.state
        state += (action * 2 - 1)
        done = np.abs(state) == (self.n_state // 2)
        reward = float(done) * ((state > 0) * 2 - 1)
        info = {}
        self.state = None if done else state

        return state, reward, done, info
