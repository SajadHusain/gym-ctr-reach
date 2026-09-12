"""Goal-conditioned equilibrium reaching with joint constraints only.

Each configuration is solved from the solver's deterministic default guess.
There is no branch tracker, elastic test, reverse solve or action backtracking.
An unavailable local derivative masks the auxiliary loss, not the transition.
"""
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from .geometry import TubeParameters
from .solver import EquilibriumError, EquilibriumSolver, SolverOptions
from .sensitivity import equilibrium_sensitivity


TASK_PROFILES = ("legacy", "generalized_reach", "generalized_hold")


def make_reach_env(config, *, max_episode_steps=None, compute_jacobian=False, task_profile=None):
    """Restore task semantics; checkpoints predating profiles remain legacy."""
    settings = dict(config.get("task_settings", {}))
    settings["task_profile"] = config.get("task_profile", "legacy") if task_profile is None else task_profile
    scales = config.get("action_scales", [.001]*3+[.05]*3)
    solver_options = SolverOptions(**config["solver_options"]) if "solver_options" in config else None
    return JointConstrainedReachEnv(config.get("system", "ctr_0"),
        tolerance_m=config.get("tolerance_m", .001),
        max_episode_steps=config.get("episode_steps", 60) if max_episode_steps is None else max_episode_steps,
        compute_jacobian=compute_jacobian, solver_options=solver_options,
        translation_step_m=scales[0], rotation_step_rad=scales[len(scales)//2], **settings)


class JointConstrainedReachEnv(gym.Env):
    metadata = {"render_modes": []}
    plant_version = "joint_constraints_v1"
    observation_bounds_version = 3

    def __init__(self, system_name="ctr_0", *, tolerance_m=.001,
                 max_episode_steps=60, witness_steps=4, tubes=None,
                 compute_jacobian=True, translation_step_m=.001,
                 rotation_step_rad=.05, task_profile="legacy",
                 initial_rotation_span_rad=.15, goal_steps_min=2, goal_steps_max=8,
                 minimum_goal_distance_m=None, max_goal_sampling_attempts=32,
                 solver_options=None):
        super().__init__()
        if system_name not in CTR_SYSTEMS_PARAMETERS:
            raise ValueError("Unknown CTR system")
        for name, value in (("tolerance_m", tolerance_m), ("translation_step_m", translation_step_m),
                            ("rotation_step_rad", rotation_step_rad)):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for value in (max_episode_steps, witness_steps):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Episode and witness counts must be positive integers")
        if task_profile not in TASK_PROFILES:
            raise ValueError("Unknown task profile")
        if not np.isfinite(initial_rotation_span_rad) or initial_rotation_span_rad < 0:
            raise ValueError("Initial rotation span must be finite and nonnegative")
        for value in (goal_steps_min, goal_steps_max, max_goal_sampling_attempts):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Goal sampling counts must be positive integers")
        if goal_steps_min > goal_steps_max:
            raise ValueError("Goal step minimum exceeds maximum")
        if minimum_goal_distance_m is None:
            minimum_goal_distance_m = 2*tolerance_m
        if not np.isfinite(minimum_goal_distance_m) or minimum_goal_distance_m <= tolerance_m:
            raise ValueError("Minimum goal distance must be finite and exceed tolerance")
        self.task_profile = task_profile
        self.terminate_on_success = task_profile != "generalized_hold"
        self.task_settings = dict(initial_rotation_span_rad=float(initial_rotation_span_rad),
            goal_steps_min=goal_steps_min, goal_steps_max=goal_steps_max,
            minimum_goal_distance_m=float(minimum_goal_distance_m),
            max_goal_sampling_attempts=max_goal_sampling_attempts)
        self.goal_distribution = ("seeded aligned starts; goals from four joint-constrained commands; fixed tolerance"
            if task_profile == "legacy" else
            "uniform common rotation plus independent bounded offsets; signed independent joint directions; "
            "variable-length projected witnesses; nontrivial reachable endpoints")
        self.system_name = system_name
        if solver_options is not None and not isinstance(solver_options, SolverOptions):
            raise ValueError("solver_options must be SolverOptions")
        self.solver = EquilibriumSolver(tubes if tubes is not None else
            [TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS[system_name].values()],
            options=solver_options)
        self.n, self.length = self.solver.n, self.solver.scale
        self.tolerance_m = float(tolerance_m)
        self.max_episode_steps, self.witness_steps = max_episode_steps, witness_steps
        self.compute_jacobian = bool(compute_jacobian)
        self.action_scales = np.r_[np.full(self.n, translation_step_m), np.full(self.n, rotation_step_rad)]
        self.action_space = spaces.Box(-1., 1., (2*self.n,), dtype=np.float32)
        tolerance_bound = np.nextafter(np.float32(self.tolerance_m/self.length), np.float32(np.inf))
        if not np.isfinite(tolerance_bound):
            raise ValueError("Tolerance exceeds supported observation range")
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(np.r_[np.tile([-1., -1., -1.], self.n), 0.].astype(np.float32),
                                      np.r_[np.ones(3*self.n), tolerance_bound].astype(np.float32)),
            "achieved_goal": spaces.Box(-self.length, self.length, (3,), dtype=np.float32),
            "desired_goal": spaces.Box(-self.length, self.length, (3,), dtype=np.float32),
        })
        # Keep the cost schema compatible with previous study tables; stability
        # calls remain identically zero in this environment.
        self.costs = dict(equilibrium_calls=0, sensitivity_calls=0, stability_calls=0,
                          root_recovery_solves=0,
                          failed_calls=0, rhs_evaluations_known=0,
                          failed_calls_without_rhs_counts=0, call_seconds=0.)
        self.reset_attempts = self.failed_resets = self.transitions = self.jacobian_failures = 0
        self.rejected_trivial_goals = 0
        self.last_failed_solve_q = None
        self.equilibrium = self.goal = self._physics_cache = None
        self._finished = True

    def _call(self, kind, function, *args):
        self.costs[kind+"_calls"] += 1
        started = time.perf_counter()
        try:
            result = function(*args)
        except (EquilibriumError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            self.costs["failed_calls"] += 1
            self.costs["failed_calls_without_rhs_counts"] += 1
            raise
        else:
            self.costs["rhs_evaluations_known"] += int(result.diagnostics.get("rhs_evaluations", 0))
            return result
        finally:
            self.costs["call_seconds"] += time.perf_counter()-started

    def _solve(self, q):
        # No previous torsion or history is used to choose the next root.
        try:
            result = self._call("equilibrium", self.solver.solve, q)
        except EquilibriumError:
            self.last_failed_solve_q = np.asarray(q, dtype=float).tolist()
            raise
        if not np.all(np.isfinite(result.tip)) or not np.array_equal(result.joints, q):
            raise EquilibriumError("Equilibrium result is nonfinite or belongs to different joints")
        self.costs["root_recovery_solves"] += int(result.diagnostics.get("root_restart_accepted", False))
        return result

    def _sample_joints(self):
        for _ in range(10000):
            q = np.r_[self.np_random.uniform(-self.solver.lengths+self.solver.constraints.minimum_deployed, 0),
                      np.zeros(self.n)]
            if self.solver.constraints.is_feasible(q):
                if self.task_profile != "legacy":
                    common = self.np_random.uniform(-np.pi, np.pi)
                    span = self.task_settings["initial_rotation_span_rad"]
                    q[self.n:] = common + self.np_random.uniform(-span, span, self.n)
                return q
        raise RuntimeError("No feasible reset configuration found")

    def projected_delta(self, q, action):
        """Same extension projection and joint caps as ProjectedJacobianLoss."""
        dq = self.solver.constraints.project(q+self.action_scales*action)-q
        return dq/max(1., float(np.max(np.abs(dq/self.action_scales))))

    def _sample_generalized_goal(self, initial):
        """Sample signed endpoint witnesses, not a controller or an IK teacher.

        Only trivial Cartesian endpoints are resampled. Numerical failures are
        raised and counted, never silently filtered from the task distribution.
        Endpoint reachability does not certify every intermediate equilibrium.
        """
        settings = self.task_settings
        for attempt in range(1, settings["max_goal_sampling_attempts"]+1):
            count = int(self.np_random.integers(settings["goal_steps_min"], settings["goal_steps_max"]+1))
            action = self.np_random.uniform(-1., 1., 2*self.n)
            target = initial.joints.copy()
            for _ in range(count):
                target += self.projected_delta(target, action)
            goal = self._solve(target).tip
            distance = np.linalg.norm(goal.astype(np.float32).astype(float)-initial.tip.astype(np.float32).astype(float))
            if distance >= settings["minimum_goal_distance_m"]:
                return goal, dict(goal_joint_witness=target.copy(), goal_witness_action=action.copy(),
                                  goal_witness_steps=count, goal_sampling_attempts=attempt)
            self.rejected_trivial_goals += 1
        raise RuntimeError("No nontrivial reachable goal within the bounded sampling budget")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = {} if options is None else dict(options)
        if set(options)-{"joints", "goal"}:
            raise ValueError("Reset options support only joints and goal")
        self._finished = True
        self.equilibrium = self._physics_cache = None
        self.last_failed_solve_q = None
        self.reset_attempts += 1
        before = dict(self.costs)
        try:
            q = self.solver.constraints._array(options["joints"]).copy() if "joints" in options else self._sample_joints()
            initial = self._solve(q)
            goal_metadata = {}
            if "goal" in options:
                goal = np.asarray(options["goal"], dtype=float)
                if goal.shape != (3,) or not np.all(np.isfinite(goal)) or np.any(abs(goal) > self.length):
                    raise ValueError("Goal must be a finite Cartesian point within the declared bounds")
            elif self.task_profile != "legacy":
                goal, goal_metadata = self._sample_generalized_goal(initial)
            else:
                # Reachable joint-space witness; only its endpoint needs an FK
                # solve. This reset path performs no sensitivity/stability calls.
                target = q.copy()
                for _ in range(self.witness_steps):
                    command = np.r_[np.full(self.n, -.0005), .04+self.np_random.uniform(-.015, .015, self.n)]
                    action = np.clip(command/self.action_scales, -1., 1.)
                    target += self.projected_delta(target, action)
                goal = self._solve(target).tip
            self.equilibrium, self.goal, self.steps = initial, goal.copy(), 0
            self._finished = False
            obs = self._observation()
            info = self._task_info(obs)
            info.update(initial_q=q.copy(), trivial_goal=info["is_success"],
                        reset_costs={k: self.costs[k]-before[k] for k in self.costs}, **goal_metadata)
            return obs, info
        except Exception:
            self.failed_resets += 1
            self.equilibrium = None
            self._finished = True
            raise

    def _observation(self):
        q = self.equilibrium.joints
        beta = np.diff(q[:self.n], prepend=0)/self.length
        alpha = np.diff(q[self.n:], prepend=0)
        ego = np.column_stack((np.cos(alpha), np.sin(alpha), beta)).ravel()
        obs = {"observation": np.r_[ego, self.tolerance_m/self.length].astype(np.float32),
               "achieved_goal": self.equilibrium.tip.astype(np.float32),
               "desired_goal": self.goal.astype(np.float32)}
        if not self.observation_space.contains(obs):
            raise EquilibriumError("Equilibrium observation is outside its finite declared bounds")
        return obs

    def _tolerances(self, info):
        if isinstance(info, dict):
            return info.get("position_tolerance", self.tolerance_m)
        return np.asarray([item.get("position_tolerance", self.tolerance_m) for item in info])

    def compute_success(self, achieved_goal, desired_goal, info):
        error = np.linalg.norm(np.asarray(achieved_goal, dtype=np.float64)-
                               np.asarray(desired_goal, dtype=np.float64), axis=-1)
        return error <= self._tolerances(info)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        success = self.compute_success(achieved_goal, desired_goal, info)
        # Goal relabelling changes rewards, never ends a continuing task.
        return success if self.terminate_on_success else np.zeros_like(success, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.asarray(~np.asarray(self.compute_success(achieved_goal, desired_goal, info)), dtype=np.float32)

    def _task_info(self, obs):
        info = {"position_tolerance": self.tolerance_m}
        info["is_success"] = bool(self.compute_success(obs["achieved_goal"], obs["desired_goal"], info))
        info["error"] = float(np.linalg.norm(obs["achieved_goal"].astype(float)-obs["desired_goal"].astype(float)))
        return info

    def _physics_context(self):
        if self._physics_cache is None:
            jacobian = np.zeros((3, 2*self.n))
            valid, reason = False, "disabled"
            if self.compute_jacobian:
                try:
                    sensitivity = self._call("sensitivity", equilibrium_sensitivity, self.solver, self.equilibrium)
                    jacobian = sensitivity.tip_jacobian
                    if not np.all(np.isfinite(jacobian)):
                        raise EquilibriumError("Nonfinite positional Jacobian")
                    valid, reason = True, "analytic_variational_ode_implicit_shooting"
                except (EquilibriumError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    self.jacobian_failures += 1
                    jacobian = np.zeros((3, 2*self.n))
                    reason = f"{type(exc).__name__}: {exc}"
            self._physics_cache = {"jacobian": jacobian.copy(), "jacobian_valid": valid,
                                   "jacobian_reason": reason, "joints": self.equilibrium.joints.copy(),
                                   "action_scales": self.action_scales.copy()}
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self._physics_cache.items()}

    def step(self, action):
        if self._finished or self.equilibrium is None:
            raise RuntimeError("Call reset before stepping or after episode end")
        proposal = np.asarray(action, dtype=float)
        if proposal.shape != (2*self.n,) or not np.all(np.isfinite(proposal)) or np.any(abs(proposal) > 1):
            raise ValueError("Action must be a finite normalized joint vector in [-1, 1]")
        costs, before = dict(self.costs), self.equilibrium
        physics = self._physics_context()
        applied = self.projected_delta(before.joints, proposal)
        try:
            target = before if not np.any(applied) else self._solve(before.joints+applied)
        except Exception:
            # No invented held transition enters replay on a failed FK solve.
            self._finished = True
            raise
        self.equilibrium = target
        self._physics_cache = None
        try:
            obs = self._observation()
        except Exception:
            self.equilibrium, self._finished = before, True
            raise
        self.steps += 1
        self.transitions += 1
        info = self._task_info(obs)
        terminated = info["is_success"] and self.terminate_on_success
        truncated = self.steps >= self.max_episode_steps and not terminated
        self._finished = terminated or truncated
        executed = applied/self.action_scales
        info.update(proposed_action=proposal.astype(np.float32),
                    executed_action=np.clip(executed, -1., 1.).astype(np.float32),
                    applied_delta=applied.copy(), action_source="plant",
                    action_replaced=bool(np.max(abs(proposal-executed)) > 1e-7),
                    q_before=before.joints.copy(), q_after=target.joints.copy(),
                    numerical_decrease_verified=False,
                    reason="joint_projection_no_motion" if not np.any(applied) else "accepted",
                    mechanics_costs={k: self.costs[k]-costs[k] for k in self.costs if k != "call_seconds"},
                    physics=physics, jacobian_valid=physics["jacobian_valid"],
                    jacobian_reason=physics["jacobian_reason"])
        return obs, float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)), terminated, truncated, info

    def close(self):
        self.equilibrium = self._physics_cache = None
        self._finished = True
