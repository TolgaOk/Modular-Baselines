import time
import torch
import numpy as np

from modular_baselines.vca.collector import make_onehot
from modular_baselines.vca.modules import (CategoricalPolicyModule,
                                           FullCategoricalTransitionModule)
from maze.environment import MazeEnv


def process_state(state, maxsize):
    state = torch.tensor([[state]])
    return make_onehot(state, maxsize)


def render(policy_param_path, fps=5):

    env = MazeEnv()
    policy_m = CategoricalPolicyModule(
        insize=env.observation_space.n,
        actsize=env.action_space.n,
        hidden_size=64,
        tau=1,
        use_gumbel=False)
    policy_m.load_state_dict(torch.load(policy_param_path))

    state = env.reset()
    done = False
    env.render()
    time.sleep(7)
    while not done:
        act = policy_m(
            process_state(state,
                          env.observation_space.n))
        print(act)
        state, reward, done, info = env.step(act)
        env.render()
        time.sleep(1/fps)


if __name__ == "__main__":
    render("maze/static/policy_m2.b")
