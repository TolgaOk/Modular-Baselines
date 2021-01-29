from gymcolab.envs.donut_world import DonutWorld
from gymcolab.envs.simplemaze import SimpleMaze

import numpy as np
import gym
import time
import random


class MazeEnv(gym.Env):

    world_map = ["##########",
                 "#      @ #",
                 "#        #",
                 "##### ####",
                 "#        #",
                 "# P      #",
                 "##########"]

    # world_map = ["##################################################",
    #              "#   #       #       #         #         #        #",
    #              "#P  #   #   #   #   #    #    #    #    #   #    #",
    #              "#       #       #        #         #        #  @ #",
    #              "##################################################"]

    # world_map = ["#########################",
    #              "#          #        @   #",
    #              "#          #            #",
    #              "#####                ####",
    #              "#     ############      #",
    #              "#     #    P#    #      #",
    #              "#     #          #      #",
    #              "#           #           #",
    #              "#           #           #",
    #              "#########################"]

    def __init__(self):
        self.colab_env = SimpleMaze(worldmap=self.world_map)

        state = self.colab_env.reset()

        state_idx = np.argwhere(state[0] == 0)

        self.state_map = {tuple(cord): ix for ix, cord in enumerate(state_idx)}
        self.state_set = np.arange(len(state_idx))
        self.reward_set = np.array([0, 1], dtype="float32")

        self.observation_space = gym.spaces.Discrete(n=len(state_idx))
        self.action_space = gym.spaces.Discrete(n=4)

        goal_ix = np.argwhere(state[1] == 1)
        self.goal_state = self.state_map[tuple(goal_ix[0])]

    def expected_reward(self):
        reward_arr = np.ones_like(self.state_set) * 0
        reward_arr[self.goal_state] = 1
        return reward_arr

    def reset(self):
        return self.process_observation(self.colab_env.reset())

    def process_observation(self, obs):
        state_idx = np.argwhere(obs[2] == 1)
        if len(state_idx.reshape(-1)) == 0:
            print(obs)
            raise RuntimeError("$!" * 40)
        return self.state_map[tuple(state_idx[0])]

    def step(self, action):
        action = action.item()
        obs, reward, done, info = self.colab_env.step(action)
        state = self.process_observation(obs)
        return state, reward, done, info

    def render(self):
        self.colab_env.render()


if __name__ == "__main__":
    env = MazeEnv()

    env.reset()
    for i in range(200):
        state, reward, done, _ = env.step(env.action_space.sample())
        # env.render()
        # time.sleep(0.1)
        print(state, reward, done)
        if done:
            break
