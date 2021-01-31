import numpy as np
import gym
import time
import random
from copy import deepcopy

from gymcolab.envs.donut_world import DonutWorld
from gymcolab.envs.simplemaze import SimpleMaze


class MazeEnv(gym.Env):

    little_world_map = ["##########",
                        "#      @ #",
                        "#        #",
                        "##### ####",
                        "#        #",
                        "# P      #",
                        "##########"]

    medium_world_map = ["################",
                        "#    @         #",
                        "#              #",
                        "########   #####",
                        "#              #",
                        "#####     ######",
                        "#              #",
                        "# P            #",
                        "################"]

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

    def __init__(self, world_map=None):
        if world_map is None:
            world_map = MazeEnv.little_world_map
        self.colab_env = SimpleMaze(worldmap=world_map)
        self.world_map = world_map
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

    def get_ideal_logits(self):
        ideal_logits = np.zeros((self.observation_space.n,
                                 self.action_space.n,
                                 self.observation_space.n))

        empty_world_map = deepcopy(self.world_map)
        p_pos_y, p_pos_x = np.argwhere(
            np.array([list(row) for row in empty_world_map]) == "P")[0]
        row = list(empty_world_map[p_pos_y])
        row[p_pos_x] = " "
        empty_world_map[p_pos_y] = "".join(row)

        for (pos_y, pos_x), state_ix in self.state_map.items():
            world_map = deepcopy(empty_world_map)
            row = list(world_map[pos_y])
            row[pos_x] = "P"
            world_map[pos_y] = "".join(row)

            for act_ix in range(self.action_space.n):
                maze_env = SimpleMaze(worldmap=world_map)
                maze_env.reset()
                player_pos = tuple(np.argwhere(
                    maze_env.step(act_ix)[0][2] == 1)[0])
                ideal_logits[state_ix][act_ix][self.state_map[player_pos]] = 10
        return ideal_logits


class ChannelMaze(gym.Env):

    def __init__(self, world_map=MazeEnv.little_world_map):
        self.env = SimpleMaze(worldmap=world_map)

        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Discrete(n=4)

    def reset(self):
        return self.env.reset()

    def step(self, act):
        if "item" in dir(act):
            act = act.item()
        return self.env.step(act)

    def expected_reward(self):
        return {"target": self.env.reset()[1],
                "target_index": 2}


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
