from typing import Tuple, Union, Dict, Generator, List, Protocol
from time import time
from dataclasses import dataclass
from gymnasium.vector import VectorEnv
import jax
import optax
import numpy as np
import jax.numpy as jnp
import distrax
import flax.linen as nn
from flax.training.train_state import TrainState

from modular_baselines.collectors.collector import RolloutCollector
from modular_baselines.buffers.buffer import Buffer
from modular_baselines.algorithms.agent import BaseAgent
from modular_baselines.utils.annealings import Coefficient
from modular_baselines.algorithms.advantages import calculate_gae
from modular_baselines.loggers.logger import MBLogger, LogGroup
from modular_baselines.loggers.writers import StdWriter, JsonWriter
from modular_baselines.loggers.datalog import ListDataLog, SingularDataLog, HistogramDataLog
from modular_baselines.utils.utils import to_jax, flatten_time, get_spaces
from modular_baselines.networks.network_jax import TrainState


@dataclass(frozen=True)
class PPOArgs():
    rollout_len: int
    ent_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    epochs: int
    seed: int
    lr: Coefficient
    clip_value: Coefficient
    batch_size: int
    max_grad_norm: float
    normalize_advantage: bool
    log_interval: int
    total_timesteps: int


class PPOAgent(Protocol):
    network: nn.Module

    def device(self) -> str: ...
    def forward(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]: ...

    def dist(self, obs: jnp.ndarray) -> Union[
        distrax.Normal,
        distrax.Categorical]: ...


class PPOLogger(Protocol):
    value_loss: ListDataLog
    policy_loss: ListDataLog
    entropy_loss: ListDataLog
    learning_rate: ListDataLog
    clip_range: ListDataLog
    approxkl: ListDataLog
    iteration: SingularDataLog
    timesteps: SingularDataLog
    time_elapsed: SingularDataLog
    fps: SingularDataLog
    env_reward: ListDataLog
    env_length: ListDataLog
    actions: HistogramDataLog


class PPO():

    def __init__(self,
                 agent: PPOAgent,
                 collector: RolloutCollector,
                 args: PPOArgs,
                 logger: PPOLogger,
                 rng_seed: int
                 ):

        self.agent = agent
        self.collector = collector
        self.logger = logger
        self.rng = jax.random.PRNGKey(rng_seed)
        
        self.buffer = self.collector.buffer
        self.num_envs = self.collector.env.num_envs

        self.args = args
        
        # self.create_network()
        

    def create_network(self):
        rng, _rng = jax.random.split(self.rng)

        inputs = [_rng, self.collector._last_obs]
        network_params = self.agent.network.init(*inputs)

        tx = optax.chain(optax.clip_by_global_norm(self.args.max_grad_norm), optax.adam(learning_rate=1e-4))

        self.network = TrainState.create(apply_fn=self.agent.network.apply,
                                         params=network_params['params'],
                                         tx=tx,
                                        )

        self.rng = rng


    def learn(self) -> None:
        """ Main loop for running the on-policy algorithm
        """
        
        train_start_time = time()
        num_timesteps = 0
        iteration = 0

        while num_timesteps < self.args.total_timesteps:
            iteration_start_time = time()
            num_timesteps = self.collector.collect(self.args.rollout_len)
            self.update()
            iteration += 1

            self.logger.iteration.push(iteration)
            self.logger.timesteps.push(num_timesteps)
            self.logger.time_elapsed.push(time() - train_start_time)
            self.logger.fps.push(
                (time() - iteration_start_time) / (self.num_envs * self.args.rollout_len))
            if iteration % self.args.log_interval == 0:
                self.logger.write("progress")

        return None

    @jax.jit
    def update(self) -> Dict[str, float]:
        """ One step parameter update. This will be called once per rollout.

        Returns:
            Dict[str, float]: Dictionary of losses to log
        """
        self.agent.network.train(True)
        sample = self.buffer.sample(batch_size=self.num_envs,
                                    rollout_len=self.args.rollout_len,
                                    sampling_length=self.args.rollout_len)

        clip_value = next(self.args.clip_value)
        lr = next(self.args.lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        rollout_data = self.prepare_rollout(sample)

        for epoch in range(self.args.epochs):
            for advantages, returns, action, old_log_prob, *replay_data in self.rollout_loader(*rollout_data):

                if self.args.normalize_advantage:
                    advantages = (advantages - advantages.mean()
                                  ) / (advantages.std() + 1e-8)
                    
                policy_params, values = self.replay_rollout(*replay_data)
                policy_dist = self.agent.dist(policy_params)
                log_probs = policy_dist.log_prob(action).unsqueeze(-1)
                entropies = policy_dist.entropy().unsqueeze(-1)

                def loss_func(self, values, returns, log_probs, old_log_prob, advantages, clip_value,
                    entropies):
                    value_loss = optax.mse_loss(values, returns)
                    ratio = jax.exp(log_probs - jax.lax.stop_gradient(old_log_prob))

                    surrogate_loss_1 = advantages * ratio
                    surrogate_loss_2 = advantages * \
                                jax.lax.clamp(1 - clip_value, ratio, 1 + clip_value)
                    
                    policy_loss = - \
                                jnp.mean(jax.lax.min(surrogate_loss_1, surrogate_loss_2))
                    entropy_loss = jnp.mean(entropies)
                    loss = value_loss * self.args.value_coef + \
                                policy_loss + entropy_loss * self.args.ent_coef
                    
                    return loss, value_loss, policy_loss, entropy_loss, ratio

                (loss, value_loss, policy_loss, entropy_loss, ratio), grads  = \
                jax.value_and_grad(loss_func, has_aux=True, argnums=0)(values,
                                                                            returns,
                                                                            log_probs,
                                                                            old_log_prob,
                                                                            advantages,
                                                                            clip_value,
                                                                            entropies)
                
                updates, new_opt_state = self.agent.network.tx.update(grads, self.opt_state, self.agent.network.params)
                new_rewarder_params = optax.apply_updates(self.agent.network.params, updates)

                self.agent.network = self.agent.network.replace(step=self.agent.network.step + 1,
                        params=new_rewarder_params,
                        opt_state=new_opt_state)

                self.logger.value_loss.push(value_loss)
                self.logger.policy_loss.push(policy_loss)
                self.logger.entropy_loss.push(entropy_loss)
                self.logger.approxkl.push(
                    ((ratio - 1) - ratio.log()).mean().item())

        self.logger.clip_range.push(clip_value)
        self.logger.learning_rate.push(lr)

    


    def rollout_to_jax(self, sample: jnp.ndarray) -> Tuple[jnp.ndarray]:
        th_obs, th_action, th_next_obs, th_old_log_prob = to_jax([
            sample["observation"],
            sample["action"],
            sample["next_observation"][:, -1],
            sample["old_log_prob"]])
        return th_obs, th_action, th_next_obs, th_old_log_prob

    def prepare_rollout(self, sample: jnp.ndarray) -> List[jnp.ndarray]:
        env_size, rollout_size = sample["observation"].shape[:2]
        th_obs, th_action, th_next_obs, th_old_log_prob = self.rollout_to_torch(
            sample)

        _, th_flatten_values = self.agent.network(flatten_time(th_obs))
        th_values = th_flatten_values.reshape(env_size, rollout_size, 1)
        _, th_next_value = self.agent.network(th_next_obs)

        advantages, returns = to_jax(calculate_gae(
            rewards=sample["reward"],
            terminations=sample["termination"],
            values=th_values.detach().cpu().numpy(),
            last_value=th_next_value.detach().cpu().numpy(),
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda)
        )

        return (advantages, returns, th_action, th_old_log_prob, th_obs)

    def rollout_loader(self, *tensors: Tuple[jnp.ndarray]
                       ) -> Generator[Tuple[jnp.ndarray], None, None]:
        if len(tensors) == 0:
            raise ValueError("Empty tensors")

        flatten_tensors = [flatten_time(tensor) for tensor in tensors]
        perm_indices = jax.random.permutation(key = self.rng, x = flatten_tensors[0].shape[0])

        for indices in perm_indices.random.split(self.args.batch_size):
            yield tuple([tensor[indices] for tensor in flatten_tensors])

    def replay_rollout(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        policy_params, values = self.agent.forward(obs)
        return policy_params, values

    @staticmethod
    def initialize_loggers(log_dir: str) -> MBLogger:

        logger = MBLogger()
        logger.value_loss = ListDataLog()
        logger.policy_loss = ListDataLog()
        logger.entropy_loss = ListDataLog()
        logger.learning_rate = ListDataLog()
        logger.clip_range = ListDataLog()
        logger.approxkl = ListDataLog()
        logger.iteration = SingularDataLog()
        logger.timesteps = SingularDataLog()
        logger.time_elapsed = SingularDataLog()
        logger.fps = SingularDataLog()
        logger.env_reward = ListDataLog()
        logger.env_length = ListDataLog()
        logger.actions = HistogramDataLog(n_bins=10)

        stdout_writer = StdWriter()
        json_writer = JsonWriter(log_dir)

        logger.add_group(
            LogGroup(
                name="progress",
                loggers=dict(
                    train=LogGroup(dict(
                        value_loss=(logger.value_loss, np.mean),
                        policy_loss=(logger.policy_loss, np.mean),
                        entropy_loss=(logger.entropy_loss, np.mean),
                        approxkl=(logger.approxkl, np.mean),
                        learning_rate=(logger.learning_rate, np.mean),
                        clip_range=(logger.clip_range, np.mean),
                    )),
                    time=LogGroup(dict(
                        iteration=(logger.iteration, lambda x: x),
                        steps=(logger.timesteps, lambda x: x),
                        elapsed=(logger.time_elapsed, lambda x: x),
                        fps=(logger.fps, lambda x: x)
                    )),
                    collector=LogGroup(dict(
                        env_reward_mean=(logger.env_reward, np.mean),
                        env_reward_median=(logger.env_reward, np.median),
                        env_length=(logger.env_length, np.mean),
                    )),
                ),
                writers=[
                    stdout_writer,
                    json_writer
                ]
            )
        )

        return logger

    @staticmethod
    def setup(env: VectorEnv,
              agent: BaseAgent,
              mb_logger: MBLogger,
              args: PPOArgs,
              rng_seed: int,
              ) -> "PPO":
        observation_space, action_space, action_dim = get_spaces(env)

        struct = np.dtype([
            ("observation", np.float32, observation_space.shape),
            ("next_observation", np.float32, observation_space.shape),
            ("action", action_space.dtype, (action_dim,)),
            ("reward", np.float32, (1,)),
            ("termination", np.float32, (1,)),
            ("truncation", np.float32, (1,)),
            ("old_log_prob", np.float32, (1,)),
        ])
        buffer = Buffer(struct, args.rollout_len, env.num_envs, mb_logger)
        collector = RolloutCollector(
            env=env,
            buffer=buffer,
            agent=agent,
            logger=mb_logger,
            rng_seed=rng_seed,
            store_normalizer_stats=False
        )
        return PPO(
            agent=agent,
            collector=collector,
            args=args,
            logger=mb_logger,
            rng_seed=rng_seed,
        )
