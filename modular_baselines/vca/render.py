import time
import torch
import numpy as np

from modular_baselines.vca.collector import make_onehot
from modular_baselines.vca.modules import CategoricalPolicyModule
from maze.environment import MazeEnv


def process_state(state, maxsize):
    state = torch.tensor([[state]])
    return make_onehot(state, maxsize)


def render(policy_param_path, fps=60):

    env = MazeEnv()
    policy_m = CategoricalPolicyModule(
        insize=env.observation_space.n,
        actsize=env.action_space.n,
        hidden_size=32)
    torch.load(policy_param_path)

    state = env.reset()
    done = False
    env.render()
    time.sleep(1)
    while not done:
        act = policy_m(
            process_state(state,
            env.observation_space.n)).argmax()
        state, reward, done, info = env.step(act)
        env.render()
        time.sleep(1/fps)


if __name__ == "__main__":
    render("maze/static/policy_m.b")
