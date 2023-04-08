import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RecordEpisodeStatistics

from modular_baselines.algorithms.ppo.ppo import PPO, PPOArgs
from modular_baselines.utils.annealings import LinearAnnealing
from modular_baselines.algorithms.agent import TorchAgent
from modular_baselines.networks.network import SeparateFeatureNetwork
from modular_baselines.algorithms.ppo.ppo import PPO, PPOArgs

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

    log_dir = "logs/ppo/test"
    logger = PPO.initialize_loggers(log_dir)
    num_envs = 16

    env = gym.vector.make("Walker2d-v4", num_envs=num_envs)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    vecenv = NormalizeReward(env, gamma=ppo_args.gamma)

    network = SeparateFeatureNetwork(
        observation_space=vecenv.single_observation_space,
        action_space=vecenv.single_action_space)

    agent = TorchAgent(
        network=network,
        observation_space=vecenv.single_observation_space,
        action_space=vecenv.single_action_space,
        logger=logger
    )

    agent = PPO.setup(
        env=vecenv,
        agent=agent,
        mb_logger=logger,
        args=ppo_args)

    agent.learn()
