"""Egocentric state plus freshly reconstructed, scaled goal error."""
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EquilibriumStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, length_scale):
        if not np.isfinite(length_scale) or length_scale <= 0:
            raise ValueError("length_scale must be positive and finite")
        super().__init__(observation_space, observation_space["observation"].shape[0]+3)
        self.length_scale = float(length_scale)

    def forward(self, observations):
        error = (observations["achieved_goal"]-observations["desired_goal"])/self.length_scale
        # Never cache goal error in replay: HER changes desired_goal.
        return torch.cat((observations["observation"], error), dim=1)
