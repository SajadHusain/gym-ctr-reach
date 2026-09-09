"""Branch-aware goal-independent plant and optional guided rollout collection.

The wrapper selects a physical action using the goal. Replay must store the
executed action reported in info, not the wrapper's incoming proposal.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from .geometry import TubeParameters
from .solver import EquilibriumSolver
from .tracking import BranchTracker, BranchInitializationError, TrackingOptions, _Meter
from .control import GoalController, ControlOptions


class EquilibriumReachEnv(gym.Env):
    """Local unloaded reaching task, with physical displacements as actions.

    Success terminates; the time budget truncates. A rejected mechanical action
    holds state and does not terminate. Reset failures are surfaced, not silently
    replaced with easier samples. No physical timestep or dynamics is implied.
    """
    metadata = {"render_modes": []}

    def __init__(self, system_name="ctr_0", *, tolerance_m=.001,
                 max_episode_steps=60, witness_steps=4, tubes=None,
                 tracking_options=None, legacy_observation_bounds=False):
        super().__init__()
        if system_name not in CTR_SYSTEMS_PARAMETERS:
            raise ValueError("Unknown CTR system")
        if not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("tolerance_m must be positive and finite")
        for count in (max_episode_steps, witness_steps):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("Episode and witness step counts must be positive integers")
        self.system_name = system_name
        self.solver = EquilibriumSolver(tubes if tubes is not None else
            [TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS[system_name].values()])
        self.tracking_options = tracking_options or TrackingOptions()
        self.tolerance_m = float(tolerance_m)
        self.max_episode_steps = max_episode_steps
        self.witness_steps = witness_steps
        self.n, self.length = self.solver.n, self.solver.scale
        self.observation_bounds_version = 1 if legacy_observation_bounds else 2
        self.action_scales = np.r_[np.full(self.n, self.tracking_options.max_translation_step_m),
                                   np.full(self.n, self.tracking_options.max_rotation_step_rad)]
        self.action_space = spaces.Box(-1., 1., (2*self.n,), dtype=np.float32)
        # In this unloaded isotropic model, |eta_i'| <= EI_i/GJ_i*k_i*k_max.
        # It is zero outside the curved segment; eta_i is constant in the guide.
        # Free-tip torsion therefore gives |L*eta_i(base)| <= L*Lc_i*EI_i/GJ_i*k_i*k_max.
        # Add a numerical envelope, not a claimed rigorous integrator error bound.
        curvature = np.array([np.hypot(t.x_curvature,t.y_curvature) for t in self.solver.tubes])
        ideal = self.length*np.array([t.length_curved for t in self.solver.tubes])*self.solver.ei/self.solver.gj*curvature*curvature.max()
        envelope = 1.01*ideal+100*self.solver.options.boundary_tolerance
        if not np.all(np.isfinite(envelope)) or np.max(envelope) >= np.finfo(np.float32).max:
            raise ValueError("Torsion observation envelope exceeds supported numerical range")
        self.torsion_observation_bound = np.nextafter(envelope.astype(np.float32), np.float32(np.inf))
        tol_bound = np.nextafter(np.float32(self.tolerance_m/self.length),np.float32(np.inf))
        if not np.isfinite(tol_bound):
            raise ValueError("Tolerance exceeds supported observation range")
        torsion_bound = np.full(self.n,np.inf) if legacy_observation_bounds else self.torsion_observation_bound
        low = np.r_[np.tile([-1., -1., -1.], self.n), -torsion_bound, 0.]
        high = np.r_[np.ones(3*self.n), torsion_bound, np.inf if legacy_observation_bounds else tol_bound]
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low.astype(np.float32), high.astype(np.float32)),
            "achieved_goal": spaces.Box(-self.length, self.length, (3,), dtype=np.float32),
            "desired_goal": spaces.Box(-self.length, self.length, (3,), dtype=np.float32),
        })
        self.costs = dict(_Meter().data)
        self.reset_attempts = self.failed_resets = self.transitions = 0
        self.tracker = None
        self.goal = None
        self._finished = True

    def _add_costs(self, values):
        for key in self.costs:
            self.costs[key] += values.get(key, 0)

    def _tracker(self, root):
        try:
            tracker = BranchTracker(self.solver, root, self.tracking_options)
        except BranchInitializationError as exc:
            self._add_costs(exc.diagnostics)
            raise
        self._add_costs(tracker.totals)
        return tracker

    def _sample_joints(self):
        for _ in range(10000):
            q = np.r_[self.np_random.uniform(-self.solver.lengths+self.solver.constraints.minimum_deployed, 0),
                      np.zeros(self.n)]
            if self.solver.constraints.is_feasible(q):
                return q
        raise RuntimeError("No feasible reset configuration found")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        options = {} if options is None else dict(options)
        unknown = set(options)-{"joints", "initial_torsion", "goal"}
        if unknown:
            raise ValueError(f"Unknown reset options: {unknown}")
        self._finished = True
        self.tracker = None
        self.reset_attempts += 1
        start_costs = dict(self.costs)
        meter = _Meter()
        witness_held = 0
        try:
            q = options["joints"] if "joints" in options else self._sample_joints()
            try:
                initial = meter.call("equilibrium", self.solver.solve, q,
                                     initial_torsion=options.get("initial_torsion"))
            finally:
                self._add_costs(meter.data)
            if "goal" in options:
                goal = np.asarray(options["goal"], dtype=float)
                if goal.shape != (3,) or not np.all(np.isfinite(goal)) or np.any(abs(goal) > self.length):
                    raise ValueError("Goal must be a finite three-vector inside the declared coordinate bounds")
            else:
                witness = self._tracker(initial)
                for _ in range(self.witness_steps):
                    command = np.r_[np.full(self.n, -.0005),
                                    .04+self.np_random.uniform(-.015, .015, self.n)]
                    before = witness.totals
                    move = witness.step(command)
                    self._add_costs({k:v-before[k] for k,v in witness.totals.items()})
                    witness_held += int(move.status != "accepted")
                goal = witness.state.equilibrium.tip
            self.goal = goal.copy()
            self.tracker = self._tracker(initial)
            self.steps = 0
            self._finished = False
            obs = self._observation()
            info = self._task_info(obs)
            info.update(reset_costs={k:v-start_costs[k] for k,v in self.costs.items()},
                        witness_held_commands=witness_held,
                        trivial_goal=info["is_success"],
                        initial_q=initial.joints.copy(),
                        initial_torsion=initial.base_torsional_strain.copy())
            return obs, info
        except Exception:
            self.failed_resets += 1
            self.tracker = None
            raise

    def _observation(self):
        root = self.tracker.state.equilibrium
        beta = np.diff(root.joints[:self.n], prepend=0)/self.length
        alpha = np.diff(root.joints[self.n:], prepend=0)
        ego = np.column_stack((np.cos(alpha), np.sin(alpha), beta)).ravel()
        features = np.r_[ego, self.length*root.base_torsional_strain, self.tolerance_m/self.length]
        observation = {"observation": features.astype(np.float32),
                "achieved_goal": root.tip.astype(np.float32),
                "desired_goal": self.goal.astype(np.float32)}
        if (not all(np.all(np.isfinite(v)) for v in observation.values())
                or not self.observation_space.contains(observation)):
            self._finished = True
            raise RuntimeError("Equilibrium observation violates its declared bounds; no clipping was applied")
        return observation

    def _tolerances(self, info):
        if isinstance(info, dict):
            return info.get("position_tolerance", self.tolerance_m)
        return np.asarray([item.get("position_tolerance", self.tolerance_m) for item in info])

    def compute_terminated(self, achieved_goal, desired_goal, info):
        error = np.linalg.norm(np.asarray(achieved_goal, dtype=np.float64)-
                               np.asarray(desired_goal, dtype=np.float64), axis=-1)
        return error <= self._tolerances(info)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.asarray(~np.asarray(self.compute_terminated(achieved_goal, desired_goal, info)), dtype=np.float32)

    def _task_info(self, obs):
        info = {"position_tolerance": self.tolerance_m}
        info["is_success"] = bool(self.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info))
        info["error"] = float(np.linalg.norm(obs["achieved_goal"].astype(float)-obs["desired_goal"].astype(float)))
        return info

    def _action(self, action):
        if self._finished or self.tracker is None:
            raise RuntimeError("Call reset before stepping or after episode end")
        action = np.asarray(action, dtype=float)
        if action.shape != (2*self.n,) or not np.all(np.isfinite(action)) or np.any(abs(action)>1):
            raise ValueError("Action must be a finite normalized joint vector in [-1, 1]")
        return action.copy()

    def _finish(self, proposal, before, applied, source, costs, *, decrease=False, reason="", physics=None):
        self._add_costs(costs)
        self.steps += 1
        self.transitions += 1
        root = self.tracker.state.equilibrium
        executed = applied/self.action_scales
        if np.any(abs(executed)>1+1e-8):
            raise RuntimeError("Executed action exceeds the declared replay action space")
        obs = self._observation()
        info = self._task_info(obs)
        terminated = info["is_success"]
        truncated = self.steps >= self.max_episode_steps and not terminated
        self._finished = terminated or truncated
        info.update(proposed_action=proposal.astype(np.float32),
                    executed_action=np.clip(executed, -1, 1).astype(np.float32),
                    applied_delta=applied.copy(), action_source=source,
                    action_replaced=bool(np.max(abs(proposal-executed))>1e-7),
                    q_before=before.joints.copy(), q_after=root.joints.copy(),
                    torsion_before=before.base_torsional_strain.copy(),
                    torsion_after=root.base_torsional_strain.copy(),
                    numerical_decrease_verified=bool(decrease),
                    mechanics_costs={k:costs.get(k, 0) for k in self.costs if k != "call_seconds"}, reason=reason)
        if physics is not None:
            info["physics"] = physics
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        return obs, reward, terminated, truncated, info

    def step(self, action):
        proposal = self._action(action)
        state = self.tracker.state
        before = state.equilibrium
        physics = self._physics_context(state)
        move = self.tracker.step(proposal*self.action_scales)
        return self._finish(proposal, before, move.applied_delta, "plant", move.diagnostics,
                            reason=move.diagnostics["reason"], physics=physics)

    def _physics_context(self, state):
        # Already computed and branch-checked by the tracker: no additional ODEs.
        return {"jacobian":state.sensitivity.tip_jacobian.copy(),
                "joints":state.equilibrium.joints.copy(),
                "action_scales":self.action_scales.copy()}

    def close(self):
        self.tracker = None
        self._finished = True


class GuidedRolloutWrapper(gym.Wrapper):
    """Goal-dependent behavior selection, with goal-independent plant semantics.

    Use ExecutedActionHerReplayBuffer when collecting these rollouts with DDPG.
    The actor and critic optimization and Bellman target are ordinary DDPG;
    the safeguard is behavior guidance, not a differentiable actor loss.
    """
    def __init__(self, env):
        if not isinstance(env, EquilibriumReachEnv):
            raise ValueError("Wrap EquilibriumReachEnv directly")
        super().__init__(env)
        self.controller = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.controller = GoalController(self.env.tracker,
            ControlOptions(goal_tolerance_m=self.env.tolerance_m))
        return observation, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return self.env.compute_terminated(achieved_goal, desired_goal, info)

    def step(self, action):
        proposal = self.env._action(action)
        state = self.env.tracker.state
        before = state.equilibrium
        physics = self.env._physics_context(state)
        result = self.controller.step(self.env.goal, policy_delta=proposal*self.env.action_scales)
        return self.env._finish(proposal, before, result.applied_delta, result.action_source,
                                result.diagnostics,
                                decrease=result.diagnostics["numerical_decrease_verified"],
                                reason=result.diagnostics["reason"], physics=physics)
