"""Goal-directed model control and a transactional policy-action safeguard.

The goal-dependent safeguard belongs to the action-selection layer. It is not
silently installed into the existing HER environment's transition function.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np

from .tracking import BranchTracker, GoalProgress, TrackedState, TrackingStep


@dataclass(frozen=True)
class ControlOptions:
    goal_tolerance_m: float = .001
    cartesian_gain: float = .5  # desired displacement per call, not a physical rate
    max_cartesian_step_m: float = .002
    damping: float = .05  # dimensionless, after joint/Cartesian normalization
    armijo_fraction: float = .1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.armijo_fraction >= 1:
            raise ValueError("armijo_fraction must be below one")


@dataclass(frozen=True)
class ControlStep:
    status: str
    action_source: str
    goal: np.ndarray
    applied_delta: np.ndarray
    state: TrackedState
    transition: TrackingStep | None
    diagnostics: dict


class GoalController:
    """Try a policy proposal, damped least squares, then projected gradient.

    Every candidate passes the same mechanics and measured-decrease checks.
    Failed candidates leave the tracker unchanged. A successful candidate is
    committed once. A call already inside tolerance produces no model queries.
    """
    def __init__(self, tracker, options=None):
        if not isinstance(tracker, BranchTracker):
            raise ValueError("tracker must be BranchTracker")
        self.tracker = tracker
        self.options = ControlOptions() if options is None else options
        if not isinstance(self.options, ControlOptions):
            raise ValueError("options must be ControlOptions")

    @property
    def joint_scales(self):
        n = len(self.tracker.state.equilibrium.joints)//2
        opt = self.tracker.options
        return np.r_[np.full(n, opt.max_translation_step_m), np.full(n, opt.max_rotation_step_rad)]

    def _proposals(self, state, goal):
        scales = self.joint_scales
        jacobian = state.sensitivity.tip_jacobian
        dscale = self.options.max_cartesian_step_m
        error = goal-state.equilibrium.tip
        desired = self.options.cartesian_gain*error
        desired *= min(1., dscale/max(np.linalg.norm(desired), np.finfo(float).tiny))
        a = jacobian*scales/dscale
        normalized = a.T@np.linalg.solve(a@a.T+self.options.damping**2*np.eye(3), desired/dscale)
        normalized /= max(1., float(np.max(abs(normalized))))
        gradient = a.T@(error/dscale)
        gradient /= max(1., float(np.max(abs(gradient))))
        return [("jacobian_dls", scales*normalized), ("jacobian_gradient", scales*gradient)]

    def step(self, goal, *, policy_delta=None):
        started = time.perf_counter()
        goal = np.asarray(goal, dtype=float)
        requirement = GoalProgress(goal, self.options.armijo_fraction)
        origin = self.tracker.state
        q = origin.equilibrium.joints
        if policy_delta is not None:
            policy_delta = np.asarray(policy_delta, dtype=float)
            if policy_delta.shape != q.shape or not np.all(np.isfinite(policy_delta)):
                raise ValueError("policy_delta must match the joint vector and be finite")
            policy_delta = policy_delta.copy()
        before_error = float(np.linalg.norm(origin.equilibrium.tip-goal))
        if not np.isfinite(before_error) or before_error > np.sqrt(np.finfo(float).max/2):
            raise ValueError("Goal distance exceeds the supported numerical range")
        v_before = .5*before_error**2
        counts_before = self.tracker.totals
        attempts = []
        selected, transition = "hold", None
        if before_error <= self.options.goal_tolerance_m:
            status = "goal_reached"
        else:
            candidates = [] if policy_delta is None else [("policy", policy_delta)]
            candidates += self._proposals(origin, goal)
            status = "stalled"
            for source, command in candidates:
                trial = self.tracker.step(command, goal_progress=requirement)
                attempts.append({"source": source, "proposed_delta": trial.requested_delta.tolist(),
                                 "status": trial.status, "diagnostics": trial.diagnostics})
                if trial.status == "accepted":
                    selected, transition = source, trial
                    status = "goal_reached" if np.linalg.norm(trial.state.equilibrium.tip-goal) <= self.options.goal_tolerance_m else "moving"
                    break
        state = self.tracker.state
        after_error = float(np.linalg.norm(state.equilibrium.tip-goal))
        counts = {k: value-counts_before[k] for k, value in self.tracker.totals.items()}
        applied = np.zeros_like(q) if transition is None else transition.applied_delta.copy()
        diagnostics = {**counts, "elapsed_seconds": time.perf_counter()-started,
                       "options": asdict(self.options), "candidate_attempts": attempts,
                       "error_before_m": before_error, "error_after_m": after_error,
                       "v_before_m2": v_before, "v_after_m2": .5*after_error**2,
                       "policy_supplied": policy_delta is not None,
                       "policy_used": selected == "policy",
                       "fallback_used": policy_delta is not None and selected.startswith("jacobian"),
                       "numerical_decrease_verified": transition is not None,
                       "goal_dependent_action_selection": True,
                       "controller_stability_certified": False, "global_convergence_certified": False,
                       "reason": "already_within_tolerance" if not attempts else attempts[-1]["diagnostics"]["reason"]}
        return ControlStep(status, selected, goal.copy(), applied, state, transition, diagnostics)
