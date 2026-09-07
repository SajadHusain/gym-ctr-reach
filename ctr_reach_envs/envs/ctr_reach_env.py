from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ctr_reach_envs.envs.CTR_Python import Tube
from ctr_reach_envs.envs.goal_tolerance import GoalTolerance
from ctr_reach_envs.envs.model import Model
from ctr_reach_envs.envs.obs import Obs


NUM_TUBES = 3


class CtrReachEnv(gym.Env):
    """Goal-conditioned, quasi-static CTR reaching environment."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        ctr_systems_parameters,
        goal_tolerance_parameters,
        noise_parameters,
        joint_representation,
        initial_joints,
        constrain_alpha,
        extension_action_limit,
        rotation_action_limit,
        max_steps_per_episode,
        n_substeps,
        evaluation,
        select_systems,
        resample_joints=True,
        length_based_sample=False,
        domain_rand=0.0,
        render_mode=None,
    ):
        super().__init__()
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")
        if any(float(value) != 0.0 for value in noise_parameters.values()):
            raise NotImplementedError("Noise parameters were inert in the supplied code and remain disabled")
        all_systems = []
        for system in ctr_systems_parameters.values():
            all_systems.append([Tube(**tube) for tube in system.values()])
        if not select_systems:
            raise ValueError("select_systems cannot be empty")
        if any(index < 0 or index >= len(all_systems) for index in select_systems):
            raise IndexError("select_systems contains an invalid system index")

        self.select_systems = list(select_systems)
        self.ctr_system_parameters = [all_systems[index] for index in self.select_systems]
        self.joint_representation = joint_representation
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.n_substeps = int(n_substeps)
        if self.max_steps_per_episode <= 0 or self.n_substeps <= 0:
            raise ValueError("max_steps_per_episode and n_substeps must be positive")
        self.starting_joints = np.asarray(initial_joints, dtype=np.float64).copy()
        self.desired_joints = self.starting_joints.copy()
        self.evaluation = bool(evaluation)
        self.resample_joints = bool(resample_joints)
        self.length_based_sample = bool(length_based_sample)
        self.domain_rand = float(domain_rand)
        self.render_mode = render_mode

        self.goal_tolerance = GoalTolerance(goal_tolerance_parameters)
        self.trig_obj = Obs(
            self.ctr_system_parameters,
            self.starting_joints,
            joint_representation,
            constrain_alpha,
        )
        self.observation_space = self.trig_obj.observation_space(self.goal_tolerance.initial)
        extension = np.full(NUM_TUBES, float(extension_action_limit), dtype=np.float64)
        rotation = np.full(
            NUM_TUBES, np.deg2rad(float(rotation_action_limit)), dtype=np.float64
        )
        self.action_scale = np.concatenate((extension, rotation))
        if not np.all(np.isfinite(self.action_scale)) or np.any(self.action_scale <= 0.0):
            raise ValueError("Action limits must be finite and positive")
        # DDPG operates in a dimensionless, consistently scaled action space. Each
        # normalized action is converted to metres/radians before it reaches the robot.
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2 * NUM_TUBES,), dtype=np.float32)

        self.model = Model(self.ctr_system_parameters)
        self.visualization = None
        self.t = 0
        self.system = 0
        self.starting_position = self.model.forward_kinematics(self.starting_joints, self.system)
        self.achieved_goal = self.starting_position.copy()
        self.desired_goal = self.starting_position.copy()

    def _sample_system(self) -> int:
        if not self.length_based_sample:
            return int(self.np_random.integers(len(self.ctr_system_parameters)))
        lengths = np.array([system[0].L for system in self.ctr_system_parameters])
        return int(self.np_random.choice(len(lengths), p=lengths / lengths.sum()))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.trig_obj.seed(seed)
        options = {} if options is None else dict(options)
        self.t = 0
        self.model.randomize_parameters(self.domain_rand, self.np_random)
        requested_system = options.get("system")
        self.system = self._sample_system() if requested_system is None else int(requested_system)
        if not 0 <= self.system < len(self.ctr_system_parameters):
            raise IndexError("Requested system is outside the selected-system range")

        requested_goal = options.get("goal")
        if requested_goal is None:
            self.desired_joints = self.trig_obj.sample_goal(self.system)
            self.desired_goal = self.model.forward_kinematics(self.desired_joints, self.system)
        else:
            self.desired_goal = self._validate_goal(requested_goal)
            self.desired_joints = None

        if "initial_joints" in options:
            self.trig_obj.set_joints(options["initial_joints"], self.system)
        elif self.resample_joints:
            self.trig_obj.set_joints(self.trig_obj.sample_goal(self.system), self.system)
        else:
            self.trig_obj.set_joints(self.trig_obj.joints, self.system)
        self.starting_joints = self.trig_obj.joints.copy()
        self.starting_position = self.model.forward_kinematics(self.starting_joints, self.system)
        self.achieved_goal = self.starting_position.copy()
        observation = self._observation()
        return observation, self._info(solver_failure=False)

    @staticmethod
    def _validate_goal(goal):
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal must be a finite three-element Cartesian position in metres")
        return goal.copy()

    def set_goal(self, goal, *, reset_step_count=True):
        """Change only the Cartesian goal while preserving the current robot state."""

        self.desired_goal = self._validate_goal(goal)
        self.desired_joints = None
        if reset_step_count:
            self.t = 0
        return self._observation()

    def get_info(self):
        return self._info(solver_failure=False)

    def _observation(self):
        return self.trig_obj.get_obs(
            self.desired_goal,
            self.achieved_goal,
            self.goal_tolerance.get_tol(),
            self.system,
        )

    def step(self, action):
        normalized_action = np.asarray(action, dtype=np.float32)
        if normalized_action.shape != self.action_space.shape or not np.all(np.isfinite(normalized_action)):
            raise ValueError("action must be a finite six-element vector")
        normalized_action = np.clip(
            normalized_action, self.action_space.low, self.action_space.high
        )
        physical_action = normalized_action.astype(np.float64) * self.action_scale
        previous_joints = self.trig_obj.joints.copy()
        previous_goal = self.achieved_goal.copy()
        for _ in range(self.n_substeps):
            self.trig_obj.set_action(physical_action, self.system)

        solver_failure = False
        try:
            self.achieved_goal = self.model.forward_kinematics(self.trig_obj.joints, self.system)
        except (RuntimeError, ValueError, FloatingPointError):
            self.trig_obj.joints = previous_joints
            self.achieved_goal = previous_goal
            solver_failure = True

        self.t += 1
        info = self._info(solver_failure=solver_failure)
        reward = float(self.compute_reward(self.achieved_goal, self.desired_goal, info))
        terminated = bool(info["is_success"])
        truncated = bool((self.t >= self.max_steps_per_episode or solver_failure) and not terminated)
        observation = self._observation()
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _info(self, *, solver_failure):
        error = float(np.linalg.norm(self.desired_goal - self.achieved_goal))
        tolerance = self.goal_tolerance.get_tol()
        info: dict[str, Any] = {
            "is_success": bool(error <= tolerance),
            "error": error,
            "position_tolerance": tolerance,
            "system_idx": self.select_systems[self.system],
            "solver_failure": bool(solver_failure),
        }
        if self.evaluation:
            info.update(
                {
                    "achieved_goal": self.achieved_goal.copy(),
                    "desired_goal": self.desired_goal.copy(),
                    "starting_position": self.starting_position.copy(),
                    "q_desired": None if self.desired_joints is None else self.desired_joints.copy(),
                    "q_achieved": self.trig_obj.joints.copy(),
                    "q_starting": self.starting_joints.copy(),
                }
            )
        return info

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved = np.asarray(achieved_goal)
        desired = np.asarray(desired_goal)
        if achieved.shape != desired.shape or achieved.shape[-1] != 3:
            raise ValueError("achieved_goal and desired_goal must have matching (..., 3) shapes")
        distance = np.linalg.norm(achieved - desired, axis=-1)
        tolerance = self._tolerance_from_info(info, distance.shape)
        return -(distance > tolerance).astype(np.float32)

    def _tolerance_from_info(self, info, target_shape):
        current = self.goal_tolerance.get_tol()
        if isinstance(info, dict):
            return float(info.get("position_tolerance", current))
        if isinstance(info, (list, tuple, np.ndarray)):
            flat = np.asarray(info, dtype=object).reshape(-1)
            values = [
                float(item.get("position_tolerance", current)) if isinstance(item, dict) else current
                for item in flat
            ]
            tolerances = np.asarray(values, dtype=np.float32)
            return tolerances.reshape(target_shape) if target_shape else float(tolerances[0])
        return current

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return np.asarray(self.compute_reward(achieved_goal, desired_goal, info) == 0.0)

    def compute_truncated(self, achieved_goal, desired_goal, info):
        shape = np.asarray(achieved_goal).shape[:-1]
        return np.zeros(shape, dtype=bool)

    def update_goal_tolerance(self, timestep):
        self.goal_tolerance.update(int(timestep))

    def get_goal_tolerance(self):
        return self.goal_tolerance.get_tol()

    def render(self):
        if self.visualization is None:
            from ctr_reach_envs.envs.ctr_3d_graph import Ctr3dGraph

            self.visualization = Ctr3dGraph()
        self.visualization.render(
            self.t,
            self.achieved_goal,
            self.desired_goal,
            self.model.r1,
            self.model.r2,
            self.model.r3,
        )

    def close(self):
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None
