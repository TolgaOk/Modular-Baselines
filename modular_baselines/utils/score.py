import numpy as np
import os
import pandas


def log_score(log_dir: str, minimum_decay: float = 0.01):
    path = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("progress.csv not fould in {}".format(path))
    dataframe = pandas.read_csv(path)
    ep_rewards = dataframe["rollout/ep_rew_mean"].to_numpy()

    decay_ratio = np.exp(np.log(minimum_decay) / len(ep_rewards))
    decays = decay_ratio ** np.arange(len(ep_rewards))[::-1]
    weights = decays / np.sum(decays)

    not_nan_indices = ~np.isnan(ep_rewards)
    return np.sum(ep_rewards[not_nan_indices] * weights[not_nan_indices])
