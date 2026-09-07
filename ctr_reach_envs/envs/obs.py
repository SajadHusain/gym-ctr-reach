from __future__ import annotations

import gymnasium as gym
import numpy as np

from ctr_reach_envs.envs.obs_utils import joint2rep, prop2ego


NUM_TUBES = 3
EXT_TOL = 1e-3


class Obs:
    """Joint constraints, goal sampling, and HER-safe observation creation."""

    def __init__(self, system_parameters, initial_joints, joint_representation, constrain_alpha=False):
        self.system_parameters = system_parameters
        self.num_systems = len(system_parameters)
        self.tube_lengths = np.array(
            [[tube.L for tube in system] for system in system_parameters], dtype=np.float64
        )
        self.constrain_alpha = bool(constrain_alpha)
        self.joint_representation = joint_representation
        if joint_representation not in {"egocentric", "proprioceptive"}:
            raise ValueError("joint_representation must be egocentric or proprioceptive")
        self.joints = np.asarray(initial_joints, dtype=np.float64).copy()
        if self.joints.shape != (6,):
            raise ValueError("initial_joints must contain six values")
        self.joint_spaces, self.joint_sample_spaces = self._joint_spaces()
        self.rng = np.random.default_rng()
        self.obs = None

    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)
        for offset, space in enumerate(self.joint_sample_spaces):
            space.seed(None if seed is None else seed + offset)

    def _joint_spaces(self):
        limits = []
        samples = []
        for tube_lengths in self.tube_lengths:
            beta_low = -tube_lengths + EXT_TOL
            sample_low = np.concatenate((beta_low, np.full(NUM_TUBES, -np.pi)))
            sample_high = np.concatenate((np.zeros(NUM_TUBES), np.full(NUM_TUBES, np.pi)))
            samples.append(gym.spaces.Box(sample_low, sample_high, dtype=np.float64))
            rotation_limit = np.pi if self.constrain_alpha else np.inf
            low = np.concatenate((beta_low, np.full(NUM_TUBES, -rotation_limit)))
            high = np.concatenate((np.zeros(NUM_TUBES), np.full(NUM_TUBES, rotation_limit)))
            limits.append(gym.spaces.Box(low, high, dtype=np.float64))
        return limits, samples

    def observation_space(self, initial_tolerance: float):
        beta_lows = []
        beta_highs = []
        for lengths in self.tube_lengths:
            if self.joint_representation == "egocentric":
                beta_lows.append(np.array([-lengths[0] + EXT_TOL, 0.0, 0.0]))
                beta_highs.append(
                    np.array([0.0, lengths[0] - lengths[1], lengths[1] - lengths[2]])
                )
            else:
                beta_lows.append(-lengths + EXT_TOL)
                beta_highs.append(np.zeros(NUM_TUBES))
        beta_low = np.min(np.stack(beta_lows), axis=0)
        beta_high = np.max(np.stack(beta_highs), axis=0)

        rep_low = np.empty(9, dtype=np.float32)
        rep_high = np.empty(9, dtype=np.float32)
        for index in range(NUM_TUBES):
            rep_low[3 * index : 3 * index + 3] = [-1.0, -1.0, beta_low[index]]
            rep_high[3 * index : 3 * index + 3] = [1.0, 1.0, beta_high[index]]
        extra_low = [0.0]
        extra_high = [initial_tolerance]
        if self.num_systems > 1:
            extra_low.append(0.0)
            extra_high.append(float(self.num_systems - 1))
        state_low = np.concatenate((rep_low, np.asarray(extra_low, dtype=np.float32)))
        state_high = np.concatenate((rep_high, np.asarray(extra_high, dtype=np.float32)))
        goal_low = np.full(3, -np.inf, dtype=np.float32)
        goal_high = np.full(3, np.inf, dtype=np.float32)
        return gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(state_low, state_high, dtype=np.float32),
                "achieved_goal": gym.spaces.Box(goal_low, goal_high, dtype=np.float32),
                "desired_goal": gym.spaces.Box(goal_low, goal_high, dtype=np.float32),
            }
        )

    def get_obs(self, desired_goal, achieved_goal, goal_tolerance, system):
        represented = prop2ego(self.joints) if self.joint_representation == "egocentric" else self.joints
        state_parts = [joint2rep(represented), np.array([goal_tolerance])]
        if self.num_systems > 1:
            state_parts.append(np.array([system], dtype=np.float64))
        state = np.concatenate(state_parts).astype(np.float32)
        self.obs = {
            "observation": state,
            "achieved_goal": np.asarray(achieved_goal, dtype=np.float32).copy(),
            "desired_goal": np.asarray(desired_goal, dtype=np.float32).copy(),
        }
        return {key: value.copy() for key, value in self.obs.items()}

    def set_joints(self, joints, system):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("joints must be a finite six-element vector")
        self.joints = np.clip(joints, self.joint_spaces[system].low, self.joint_spaces[system].high)
        self._apply_extension_constraints(system)

    def set_action(self, action, system):
        self.joints = np.clip(
            self.joints + np.asarray(action, dtype=np.float64),
            self.joint_spaces[system].low,
            self.joint_spaces[system].high,
        )
        self._apply_extension_constraints(system)

    def _apply_extension_constraints(self, system):
        betas = self.joints[:NUM_TUBES].copy()
        lengths = self.tube_lengths[system]
        for index in range(1, NUM_TUBES):
            betas[index - 1] = min(betas[index - 1], betas[index])
            betas[index - 1] = max(
                betas[index - 1], lengths[index] - lengths[index - 1] + betas[index]
            )
        self.joints = np.concatenate((betas, self.joints[NUM_TUBES:]))

    def sample_goal(self, system):
        space = self.joint_sample_spaces[system]
        lengths = self.tube_lengths[system]
        for _ in range(1000):
            sample = self.rng.uniform(space.low, space.high)
            betas = sample[:NUM_TUBES]
            valid = [
                betas[index - 1] <= betas[index]
                and betas[index - 1] + lengths[index - 1] >= lengths[index] + betas[index]
                for index in range(1, NUM_TUBES)
            ]
            if all(valid):
                return sample
        raise RuntimeError("Unable to sample a feasible CTR joint configuration")

