from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import gym

from modular_baselines.algorithms.ppo.ppo import PPO, PPOArgs
from modular_baselines.utils.annealings import LinearAnnealing
from modular_baselines.algorithms.agent import TorchAgent
from modular_baselines.networks.network import SeparateFeatureNetwork
from modular_baselines.algorithms.ppo.ppo import PPO, PPOArgs
from modular_baselines.environments.legacy_gym_wrapper import LegacyWrapper

if __name__ == "__main__":

    ppo_args = PPOArgs(
        rollout_len=2048,
        ent_coef=1e-4,
        value_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        epochs=10,
        lr=LinearAnnealing(3e-4, 0.0, 5_000_000 // (2048 * 16)),
        clip_value=LinearAnnealing(0.2, 0.2, 5_000_000 // (2048 * 16)),
        batch_size=64,
        max_grad_norm=1.0,
        normalize_advantage=True,
        log_interval=1,
        total_timesteps=5_000_000,
    )

    env_name = "Walker2d-v4"
    log_dir = "logs/ppo/test3"
    n_envs = 16
    vec_norm_info = {
        "norm_reward": True,
        "norm_obs": True,
        "clip_obs": 1e5,
        "clip_reward": 1e5,
    }

    logger = PPO.initialize_loggers(log_dir)
    vecenv = make_vec_env(
        lambda: LegacyWrapper(gym.make(env_name)),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv)
    vecenv = VecNormalize(
        vecenv,
        training=True,
        gamma=0.99,
        **vec_norm_info)

    network = SeparateFeatureNetwork(
        observation_space=vecenv.observation_space,
        action_space=vecenv.action_space)
    agent = TorchAgent(
        network=network,
        observation_space=vecenv.observation_space,
        action_space=vecenv.action_space,
        logger=logger
    )
    ppo = PPO.setup(
        env=vecenv,
        agent=agent,
        mb_logger=logger,
        args=ppo_args)
    ppo.learn()
