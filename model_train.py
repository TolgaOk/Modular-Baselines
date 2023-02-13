from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass
from collections import defaultdict
from tqdm.notebook import tqdm
import plotly.graph_objects as go
from IPython.display import display
import torch
import numpy as np
import gym
import json

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env

from modular_baselines.networks.model import StiefelNetwork


@dataclass
class UnitaryLog():
    y: List[float]


class FigureLogger():

    def __init__(self, log_period: int = 100) -> None:
        self.fig = go.FigureWidget()
        self.fig.update_layout(template="plotly_white")
        self.trace_data = defaultdict(self._default_log)
        self.shared_x = 0
        self.log_period = log_period

    def _default_log(self) -> UnitaryLog:
        self.fig.add_trace(go.Scatter(y=[], x=[]))
        return UnitaryLog([])

    def log(self, log_info: Dict[str, float]) -> None:
        for name, value in log_info.items():
            self.trace_data[name].y.append(value)
        self.shared_x += 1
        if (self.shared_x % self.log_period) == 0:
            with self.fig.batch_update():
                for index, (name, unilog) in enumerate(self.trace_data.items()):
                    self.fig.data[index].y = [*self.fig.data[index].y,
                                              np.mean(self.trace_data[name].y)]
                    self.trace_data[name].y = []
                    self.fig.data[index].x = [*self.fig.data[index].x, self.shared_x]
                    self.fig.data[index].name = name

    def render(self) -> None:
        display(self.fig)
        return None


class Trainer():

    def __init__(self, venv: VecEnv):
        self.venv = venv

    def gather(self, n_iterations: int) -> np.ndarray:
        buffer = np.zeros(
            shape=(n_iterations // self.venv.num_envs, self.venv.num_envs),
            dtype=np.dtype(
                [("obs", np.float32, self.venv.observation_space.shape),
                 ("next_obs", np.float32, self.venv.observation_space.shape),
                 ("act", self.venv.action_space.dtype, (self.venv.action_space.shape[0],))]
            )
        )
        observation = self.venv.reset()
        for iteration in tqdm(range(n_iterations // self.venv.num_envs)):
            action = np.random.uniform(0, 1, (self.venv.num_envs, self.venv.action_space.shape[0]))
            action = action * (self.venv.action_space.high - self.venv.action_space.low)
            action = action + self.venv.action_space.low

            new_observation, reward, done, info = self.venv.step(action)

            if done.any():
                next_observation = new_observation.copy()
                for index, env_data in enumerate(info):
                    if done[index]:
                        next_observation[index] = env_data["terminal_observation"]
            else:
                next_observation = new_observation

            buffer[iteration, :]["obs"] = observation
            buffer[iteration, :]["act"] = action
            buffer[iteration, :]["next_obs"] = next_observation

            observation = new_observation
        return buffer

    def train(self,
              model: torch.nn.Module,
              buffer: np.ndarray,
              n_epocs: int,
              batch_size: int,
              lr: float,
              device: str,
              logger: Callable[[Dict[str, float]], None]):
        model.to(device)
        th_buffers = [torch.from_numpy(buffer.reshape(-1)[name]).to(device)
                      for name in buffer.dtype.names]

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

        n_iteration = buffer.size // batch_size
        train_indices = torch.from_numpy(np.concatenate(
            [np.random.permutation(buffer.size) for _ in range(n_epocs)])).long()
        for indices in tqdm(np.split(train_indices, n_iteration * n_epocs)):
            th_obs, th_next_obs, th_act = [tensor[indices] for tensor in th_buffers]

            embed = model.immersion(th_obs)
            next_pred_embed_param = model(embed, th_act)
            dist = model.dist(next_pred_embed_param)
            with torch.no_grad():
                next_embed = model.immersion(th_next_obs)

            embed_loss = -dist.log_prob(next_embed).mean(0)
            recon_loss = torch.nn.functional.mse_loss(model.submersion(dist.rsample()), th_next_obs)
            loss = embed_loss + recon_loss
            logger({"loss": loss.item(), "embed_loss": embed_loss.item(), "recon_loss": recon_loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return


if __name__ == "__main__":
    buffer = np.load("walker-2d-buffer-1M-random.npy")
    buffer["obs"] /= 10
    buffer["next_obs"] /= 10
    venv = make_vec_env(lambda: gym.make(
        "Walker2d-v4", exclude_current_positions_from_observation=False), 16)
    model = StiefelNetwork(
        state_size=venv.observation_space.shape[0],
        action_size=venv.action_space.shape[0],
        # hidden_size=200,
        # n_layers=2,
    )
    trainer = Trainer(venv)
    logger = FigureLogger(500)
    # display(logger.fig)
    trainer.train(model, buffer, 1, 32, 0.0001, "cuda:0", logger=logger.log)

    with open("model_fig_data.json", "w") as obj:
        data = json.loads(logger.fig.to_json())
        json.dump(data, obj)
    torch.save(model.state_dict(), "model_data_1M.b")

