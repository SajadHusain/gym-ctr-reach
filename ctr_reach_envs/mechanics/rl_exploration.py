"""Shared DDPG exploration for the ordinary and Jacobian-guided actors.

The paper profile reproduces the exploration values in paper_config.py and
the uniform-action mixture in PaperDDPG. It does not change the current
plant's increment limits, warmup duration, goals or optimizer schedule.
"""
import numpy as np
from stable_baselines3 import DDPG

from ctr_reach_envs.paper_config import PAPER_CONFIG


def exploration_settings(profile="paper", noise_std=None, random_exploration=None, n_joints=6):
    if profile not in ("paper", "gaussian"):
        raise ValueError("exploration-profile must be paper or gaussian")
    if noise_std is None:
        sigma = (np.asarray(PAPER_CONFIG["normalized_action_noise_std"], dtype=float)
                 if profile == "paper" else np.full(n_joints, .05))
        if sigma.shape != (n_joints,):
            raise ValueError("The paper exploration profile requires six joints, or an explicit noise-std override")
    else:
        sigma = np.full(n_joints, float(noise_std))
    epsilon = (PAPER_CONFIG["random_exploration"] if profile == "paper" else 0.) if random_exploration is None else random_exploration
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("noise-std must be finite and nonnegative")
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("random-exploration must be in [0, 1]")
    return {"profile": profile, "normalized_action_noise_std": sigma.tolist(),
            "random_exploration": float(epsilon),
            "warmup": "uniform proposals before learning-starts",
            "after_warmup": "uniform proposal with probability epsilon; otherwise clipped actor plus Gaussian noise"}


class ExplorationDDPG(DDPG):
    """SB3 DDPG updates with the legacy uniform-action exploration mixture.

    With epsilon=0 the sampler consumes exactly the same random stream as
    SB3 DDPG. Counts describe proposals before joint projection. They cannot
    establish coverage of all reachable robot states.
    """
    def __init__(self, *args, random_exploration=0., **kwargs):
        if not np.isfinite(random_exploration) or not 0 <= random_exploration <= 1:
            raise ValueError("random_exploration must be in [0, 1]")
        self.random_exploration = float(random_exploration)
        self.exploration_counts = dict(warmup_uniform=0, mixture_uniform=0, policy_with_noise=0)
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        if self.num_timesteps < learning_starts:
            self.exploration_counts["warmup_uniform"] += n_envs
            return action, buffer_action
        # Skip this draw at zero epsilon to preserve the existing zero-weight
        # and SB3 parity tests, including RNG consumption during training.
        mask = (np.random.random(n_envs) < self.random_exploration
                if self.random_exploration else np.zeros(n_envs, dtype=bool))
        for index in np.flatnonzero(mask):
            action[index] = self.action_space.sample()
            buffer_action[index] = self.policy.scale_action(action[index])
        self.exploration_counts["mixture_uniform"] += int(mask.sum())
        self.exploration_counts["policy_with_noise"] += int(n_envs-mask.sum())
        return action, buffer_action
