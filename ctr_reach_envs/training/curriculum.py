"""Tolerance curriculum for the preserved paper reproduction."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GoalToleranceCurriculumCallback(BaseCallback):
    def _on_step(self):
        self.training_env.env_method("update_goal_tolerance", self.num_timesteps)
        values = self.training_env.env_method("get_goal_tolerance")
        self.logger.record("curriculum/position_tolerance_m", float(np.mean(values)))
        return True
