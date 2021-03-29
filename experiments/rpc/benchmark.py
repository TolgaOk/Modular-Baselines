import torch
import gym
import time
import argparse
from ctypes import c_int
from torch.multiprocessing import Process as ThProcess
from torch.multiprocessing import Queue as ThQueue
from torch.multiprocessing import Value, Lock


class EnvBenchmark():

    def __init__(self, envname, num_procs, sync, maxsize=int(1e6)):
        self.num_procs = num_procs
        self.envname = envname
        self.maxsize = maxsize

        self.counter = Value(c_int)
        self.counter_lock = Lock()

        self.processes = []
        for rank in range(self.num_procs):
            if sync:
                process = ThProcess(target=EnvBenchmark.sync_worker,
                                    args=(rank, envname, maxsize, num_procs))
            else:
                process = ThProcess(target=EnvBenchmark.async_worker,
                                    args=(rank, envname, maxsize, self.counter, self.counter_lock))
            self.processes.append(process)

    def start(self):

        start_time = time.time()
        for process in self.processes:
            process.start()

        for process in self.processes:
            process.join()
        return self.maxsize / (time.time() - start_time)
        print("FPS: {}".format())

    @staticmethod
    def increment(counter, lock):
        with lock:
            counter.value += 1
            value = counter.value
        return value

    @staticmethod
    def async_worker(rank, envname, maxsize, counter, lock):

        env = gym.make(envname)
        state = env.reset()

        value = 0
        while value < maxsize:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            value = EnvBenchmark.increment(counter, lock)

    @staticmethod
    def sync_worker(rank, envname, maxsize, num_workers):

        env = gym.make(envname)
        state = env.reset()

        for _ in range(maxsize // num_workers):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--sync", action="store_true",
                        help="Use sync workers")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of parallelized workers")
    parser.add_argument("--maxsteps", type=int, default=int(1e5),
                        help="Maximum number of steps")
    args = parser.parse_args()

    fps = EnvBenchmark("PongNoFrameskip-v4", args.n_procs, args.sync, args.maxsteps).start()
    print("FPS: {}".format(fps))
