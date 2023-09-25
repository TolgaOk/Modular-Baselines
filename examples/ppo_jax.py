from typing import Any, Dict
import jax
import optax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RecordEpisodeStatistics
from sacred import SETTINGS
from sacred import Experiment
from sacred.observers import FileStorageObserver


from modular_baselines.algorithms.ppo.ppo_jx import PPO, PPOArgs
from modular_baselines.utils.annealings import LinearAnnealing
from modular_baselines.algorithms.jax_agent import JaxAgent
from modular_baselines.networks.network_jax import SeparateFeatureNetwork, TrainState
from modular_baselines.utils.experiment import record_log_files

SETTINGS["DISCOVER_DEPENDENCIES"] = "sys"
SETTINGS["DISCOVER_SOURCES"] = "dir"

ex = Experiment(name="PPO/walker2d-v4", base_dir="../modular_baselines")
ex.observers.append(FileStorageObserver("sacred_files"))


@ex.config
def default_configs():
    total_timesteps = 5_000_000
    rollout_len = 2048

    env_args = dict(
        use_norm_obs=True,
        use_norm_rew=True,
        env_name="Walker2d-v4",
        num_envs=16,
    )
    ppo_args = PPOArgs(
        rollout_len=rollout_len,
        ent_coef=1e-4,
        value_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        epochs=10,
        seed=12345,
        lr=LinearAnnealing(
            3e-4, 0.0, total_timesteps // (rollout_len * env_args["num_envs"])),
        clip_value=LinearAnnealing(
            0.2, 0.2, total_timesteps // (rollout_len * env_args["num_envs"])),
        batch_size=64,
        max_grad_norm=1.0,
        normalize_advantage=True,
        log_interval=1,
        total_timesteps=total_timesteps,
    )
    log_dir = "logs/ppo/test"


@ex.automain
def run(ppo_args: PPOArgs, env_args: Dict[str, Any], log_dir: str):
    rng_seed = ppo_args.seed    
    rng = jax.random.PRNGKey(rng_seed)
    _, _rng = jax.random.split(rng)
    
    logger = PPO.initialize_loggers(log_dir)
    env = gym.vector.make(env_args["env_name"], num_envs=env_args["num_envs"])
    env = RecordEpisodeStatistics(env)
    if env_args["use_norm_obs"]:
        env = NormalizeObservation(env)
    if env_args["use_norm_rew"]:
        vecenv = NormalizeReward(env, gamma=ppo_args.gamma)
    
    if isinstance(vecenv.single_action_space, Box):
            out_size = vecenv.single_action_space.shape[0] * 2

    network = SeparateFeatureNetwork(in_size=vecenv.single_observation_space.shape[0],
        out_size=out_size,
        policy_hidden_size=64,
        value_hidden_size=64,
        observation_space=vecenv.single_observation_space,
        action_space=vecenv.single_action_space)
    
    inputs = [_rng, jnp.zeros(vecenv.single_observation_space.shape[0])]
    network_params = network.init(*inputs)
    tx = optax.chain(optax.clip_by_global_norm(ppo_args.max_grad_norm), optax.adam(learning_rate=1e-4))

    train_state = TrainState.create(apply_fn=network.apply,
                                params=network_params['params'],
                                tx=tx)

    agent = JaxAgent(
        state=train_state,
        observation_space=vecenv.single_observation_space,
        action_space=vecenv.single_action_space,
        logger=logger,
        rng_seed=rng_seed
    )
    agent = PPO.setup(
        env=vecenv,
        agent=agent,
        mb_logger=logger,
        args=ppo_args,
        rng_seed=rng_seed)

    agent.learn()
    record_log_files(ex, log_dir)
