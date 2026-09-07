"""HER for episodic reaching where reaching the desired goal ends the task."""

from copy import deepcopy

import numpy as np
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


class GoalTerminationHerReplayBuffer(HerReplayBuffer):
    """Recompute reward AND termination after relabelling the desired goal.

    SB3 2.9's standard buffer preserves the original done flag. That is suitable
    for goal-independent termination, but not our success-terminated reaching
    task. Timeouts still bootstrap unless the relabelled goal is reached.
    Real samples and episode boundaries retain SB3's usual handling.

    The environment must expose vectorized compute_reward and compute_terminated
    methods that use the per-transition info (including position_tolerance).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.copy_info_dict:
            raise ValueError("GoalTerminationHerReplayBuffer requires copy_info_dict=True")

    def _get_virtual_samples(self, batch_indices, env_indices, env=None):
        # Advanced indexing makes copies; relabelling must not modify stored data.
        observation = {key: value[batch_indices, env_indices].copy()
                       for key, value in self.observations.items()}
        next_observation = {key: value[batch_indices, env_indices].copy()
                            for key, value in self.next_observations.items()}
        infos = deepcopy(self.infos[batch_indices, env_indices])
        goals = self._sample_goals(batch_indices, env_indices)
        observation["desired_goal"] = goals.copy()
        next_observation["desired_goal"] = goals.copy()

        if self.env is None:
            raise RuntimeError("HER requires an attached environment for goal relabelling")
        # Compute in physical coordinates BEFORE any VecNormalize transformation.
        arguments = (next_observation["achieved_goal"], goals, infos)
        rewards = np.asarray(
            self.env.env_method("compute_reward", *arguments, indices=[0])[0],
            dtype=np.float32,
        ).reshape(-1, 1)
        terminated = np.asarray(
            self.env.env_method("compute_terminated", *arguments, indices=[0])[0],
            dtype=np.float32,
        ).reshape(-1, 1)

        # Do not mask relabelled success with the ORIGINAL goal's timeout flag:
        # a transition at the time limit can still reach its new goal.
        observation = self._normalize_obs(observation, env)
        next_observation = self._normalize_obs(next_observation, env)
        return DictReplayBufferSamples(
            observations={key: self.to_torch(value) for key, value in observation.items()},
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations={key: self.to_torch(value) for key, value in next_observation.items()},
            dones=self.to_torch(terminated),
            rewards=self.to_torch(self._normalize_reward(rewards, env)),
        )
