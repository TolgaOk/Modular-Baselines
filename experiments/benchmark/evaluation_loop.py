import torch
import numpy as np
import gym
import time
import argparse
from ctypes import c_int
import cv2
from torch.multiprocessing import Process as ThProcess
from torch.multiprocessing import Value, Lock

from stable_baselines3.common.atari_wrappers import AtariWrapper


class Policy(torch.nn.Module):

    def __init__(self,
                 in_channel,
                 action_size,
                 hidden_size=128):
        super().__init__()
        self.in_channel = in_channel
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 32,
                            kernel_size=8, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 64, 512),
            torch.nn.ReLU()
        )
        self.action_layers = torch.nn.Sequential(
            torch.nn.Linear(512, action_size)
        )

    def forward(self, tensor):
        processed_tensor = tensor.float() / 255
        features = self.cnn(processed_tensor)
        act_logit = self.action_layers(features)
        dist = torch.distributions.categorical.Categorical(logits=act_logit)
        action = dist.sample()
        return action


class EnvWrapper(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 frame_stack: int = 4,
                 frame_skip: int = 4,
                 screen_size: int = 84,
                 noop_max=30):

        assert frame_skip > 1
        assert noop_max > 0
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(frame_skip, 84, 84),
            dtype=env.observation_space.dtype)

        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.noop_max = noop_max

        self.obs_memory = np.zeros(
            (self.observation_space.shape),
            dtype=self.observation_space.dtype)

        self.frame_memory = np.zeros(
            (2, *self.observation_space.shape[1:]),
            dtype=self.observation_space.dtype)

    def gray_scale(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,
                           (self.screen_size, self.screen_size),
                           interpolation=cv2.INTER_AREA)
        return frame

    def step(self, action):
        total_reward = 0.0
        done = None
        for index in range(self.frame_skip):
            frame, reward, done, info = self.env.step(action)
            if index == self.frame_skip - 2:
                self.frame_memory[0] = self.gray_scale(frame)
            if index == self.frame_skip - 1:
                self.frame_memory[1] = self.gray_scale(frame)
            total_reward += reward
            if done:
                break

        self.obs_memory[1:] = self.obs_memory[:-1]
        self.obs_memory[0] = self.frame_memory.max(axis=0)
        return self.obs_memory, total_reward, done, info

    def reset(self):
        frame = self.env.reset()
        noops = np.random.randint(self.noop_max // 2, self.noop_max + 1)

        obs = np.zeros(0)
        for _ in range(noops):
            frame, _, done, _ = self.env.step(0)
            if done:
                frame = self.env.reset()

        self.obs_memory[:] = 0
        self.obs_memory[0] = self.gray_scale(frame)
        return self.obs_memory


class PolicyLoopBenchmark():

    def __init__(self, envname, num_procs, device, maxsize=int(1e6)):
        self.num_procs = num_procs
        self.envname = envname
        self.maxsize = maxsize
        self.device = device

        self.counter = Value(c_int)
        self.counter_lock = Lock()

        self.policy = Policy(in_channel=4, action_size=6).to(device)
        self.policy.share_memory()

        self.processes = []
        for rank in range(self.num_procs):
            process = ThProcess(target=PolicyLoopBenchmark.async_worker,
                                args=(rank, envname, maxsize, self.counter,
                                      self.counter_lock, self.policy, device))
            self.processes.append(process)

    def start(self):

        start_time = time.time()
        for process in self.processes:
            process.start()

        for process in self.processes:
            process.join()
        return self.maxsize / (time.time() - start_time)

    @ staticmethod
    def increment(counter, lock):
        with lock:
            counter.value += 1
            value = counter.value
        return value

    @ staticmethod
    def async_worker(rank, envname, maxsize, counter, lock, policy, device):
        env = gym.make(envname)
        env = EnvWrapper(env,
                         frame_stack=4,
                         noop_max=30,
                         frame_skip=4,
                         screen_size=84,)
        state = env.reset()

        value = 0
        while value < maxsize:
            # state = (torch.from_numpy(state).float() / 255).unsqueeze(0)
            # action = policy(state.to(device)).item()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            value = PolicyLoopBenchmark.increment(counter, lock)


if __name__ == "__main__":
    cv2.ocl.setUseOpenCL(False)
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of parallelized workers")
    parser.add_argument("--maxsteps", type=int, default=int(1e5),
                        help="Maximum number of steps")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Model device")
    args = parser.parse_args()

    fps = PolicyLoopBenchmark("PongNoFrameskip-v4", args.n_procs,
                              args.device, args.maxsteps).start()
    print("FPS: {}".format(fps))
