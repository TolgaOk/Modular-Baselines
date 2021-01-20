import numpy as np
import gym
from bqplot import Figure, LinearScale, Axis, ColorScale
from bqplot_image_gl import ImageGL

from modular_baselines.utils.wrappers import (NormalizeObservation,
                                              SkipSteps,
                                              AggregateObservation,
                                              IndexObsevation,
                                              IndexAction,
                                              ResetWithNonZeroReward)


class PongEnv(gym.Env):

    def __init__(self,
                 envname="Pong-ramDeterministic-v4",
                 state_ix=[51, 50, 49, 54],
                 action_ix=[0, 2, 3],
                 aggr_ix=[2, 3],
                 skip_initial_n_steps=16):
        self.pong_env = self.make_pong_env(
            envname,
            state_ix,
            action_ix,
            aggr_ix,
            skip_initial_n_steps)

        self.observation_space = self.pong_env.observation_space
        self.action_space = self.pong_env.action_space
        self.image = None

    def render(self):
        self.image.image = self.pong_env.render(mode="rgb_array")

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
        return self.pong_env.step(action)

    def reset(self):
        return self.pong_env.reset()

    def make_pong_env(self,
                      envname="Pong-ramDeterministic-v4",
                      state_ix=[51, 50, 49, 54],
                      action_ix=[0, 2, 3],
                      aggr_ix=[2, 3],
                      skip_initial_n_steps=16):

        env = gym.make(envname)
        env = IndexObsevation(env, state_ix)
        env = AggregateObservation(env, aggr_ix)
        env = SkipSteps(env, skip_initial_n_steps)
        env = NormalizeObservation(env)
        env = IndexAction(env, action_ix)
        env = ResetWithNonZeroReward(env)

        return env

    def reward_info(self):
        return {"index": 2,
                "lower_threshold": 0.2,
                "upper_threshold": 0.8}
