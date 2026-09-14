"""Paper actor/critic inputs extended with optional mechanical information."""
import torch
from ctr_reach_envs.paper_policy import PaperStateExtractor


class PhysicsStateExtractor(PaperStateExtractor):
    """Keep HER goal-error reconstruction and append state-only features."""

    def __init__(self, observation_space):
        super().__init__(observation_space)
        shape = observation_space["physics"].shape
        if shape not in ((19,), (29,)):
            raise ValueError("Unsupported physics observation shape")
        self._features_dim += shape[0]

    def forward(self, observations):
        return torch.cat((super().forward(observations), observations["physics"]), dim=1)
