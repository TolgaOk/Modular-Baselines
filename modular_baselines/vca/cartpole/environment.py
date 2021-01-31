import gym
import  numpy as np


class CartPole(gym.Env):

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self):
        return self.wrap_observation(self.env.reset())

    def step(self, act):
        state, reward, done, info = self.env.step(act)
        return self.wrap_observation(state), reward, done, info

    def reward_info(self):
        return {"index": [0, 1]}

    def wrap_observation(self, obs):
        return obs.astype(np.float32)

 