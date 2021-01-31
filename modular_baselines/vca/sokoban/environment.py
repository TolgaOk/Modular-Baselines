import numpy as np
import gym
from bqplot import Figure, LinearScale, Axis, ColorScale
from bqplot_image_gl import ImageGL
import random

from gym_sokoban.envs.sokoban_env import SokobanEnv


class Sokoban(gym.Env):

    def __init__(self, dim_room=(10, 10), num_boxes=4):
        self.sokoban_env = FastSokoban(
            dim_room=dim_room, num_boxes=num_boxes)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4, *self.sokoban_env.dim_room))
        self.action_space = self.sokoban_env.action_space
        self.image = None

    def render(self):
        self.image.image = self.sokoban_env.render(mode="rgb_array")

    def make_figure(self, scale=2):
        scale_x = LinearScale(min=0, max=1)
        scale_y = LinearScale(min=1, max=0)
        scales = {"x": scale_x,
                  "y": scale_y}

        figure = Figure(scales=scales, axes=[])
        figure.layout.height = "{}px".format(210 * scale)
        figure.layout.width = "{}px".format(160 * scale)

        scales_image = {"x": scale_x,
                        "y": scale_y,
                        "image": ColorScale(min=0, max=1)}

        image = ImageGL(image=np.zeros(
            (210, 160, 3), dtype=np.uint8), scales=scales_image)

        figure.marks = (image,)
        self.image = image
        return figure

    def step(self, action):
        obs, reward, done, info = self.sokoban_env.step(action)
        reward = float(self.sokoban_env._check_if_all_boxes_on_target())
        return self.wrap_observation(obs), reward, done, info

    def reset(self):
        return self.wrap_observation(self.sokoban_env.reset())

    def expected_reward(self):
        return {"target": self.env.reset()[1],
                "target_index": 2}

    def wrap_observation(self, obs):
        return np.stack(obs).astype(np.float32)


class FastSokoban(SokobanEnv):

    def reset(self):
        self.room_fixed = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 2, 0],
             [0, 1, 1, 2, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]])
        self.room_state = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 4, 1, 1, 2, 0],
             [0, 1, 1, 2, 1, 1, 0],
             [0, 0, 0, 0, 4, 1, 0],
             [0, 0, 0, 0, 5, 1, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]])
        self.box_mapping = {(1, 5): (1, 2), (2, 3): (3, 4)}

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render("raw")
        return starting_observation

    def step(self, action):
        return super().step(action, observation_mode="raw")
