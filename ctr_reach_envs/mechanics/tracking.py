"""Transactional continuation of numerically admissible unloaded equilibria.

Accepted states satisfy sampled numerical checks, not continuum or closed-loop
stability certificates. Commands and returned increments use metres/radians.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import time

import numpy as np

from .solver import Equilibrium, EquilibriumError
from .sensitivity import Sensitivity, equilibrium_sensitivity
from .stability import StabilityDiagnostic, elastic_stability


@dataclass(frozen=True)
class TrackingOptions:
    max_translation_step_m: float = .001
    max_rotation_step_rad: float = .05
    max_backtracks: int = 8
    tip_absolute_tolerance_m: float = 2e-6
    tip_relative_tolerance: float = .05
    twist_absolute_tolerance: float = .001  # dimensionless: base twist * longest L
    twist_relative_tolerance: float = .05
    max_twist_change: float = .25
    minimum_elastic_eigenvalue: float = .05
    shooting_condition_limit: float = 1e5
    event_clearance_m: float = 1e-8
    reverse_tip_tolerance_m: float = 2e-6
    reverse_twist_tolerance: float = .001

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "max_backtracks":
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 20:
                    raise ValueError("max_backtracks must be an integer in [0, 20]")
            elif isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.shooting_condition_limit <= 1:
            raise ValueError("shooting_condition_limit must exceed one")


@dataclass(frozen=True)
class TrackedState:
    equilibrium: Equilibrium
    sensitivity: Sensitivity
    stability: StabilityDiagnostic
    accepted_steps: int


@dataclass(frozen=True)
class TrackingStep:
    status: str
    requested_delta: np.ndarray
    projected_delta: np.ndarray
    applied_delta: np.ndarray
    accepted_fraction: float
    state: TrackedState
    diagnostics: dict


class BranchInitializationError(EquilibriumError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class GoalProgress:
    """Numerical Armijo decrease for V = ||tip-goal||^2/2 at a fixed goal.

    The test uses the actual joint increment and solved tip, without accepting
    floating-point slack. It does not certify the underlying continuum model.
    """
    goal: tuple
    armijo_fraction: float = .1

    def __post_init__(self):
        goal = np.asarray(self.goal, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal must be a finite Cartesian three-vector in metres")
        if not np.isfinite(self.armijo_fraction) or not 0 < self.armijo_fraction < 1:
            raise ValueError("armijo_fraction must lie strictly between zero and one")
        object.__setattr__(self, "goal", tuple(float(v) for v in goal))

    def slope(self, origin, dq):
        return float((origin.equilibrium.tip-np.asarray(self.goal))@origin.sensitivity.tip_jacobian@dq)

    def check(self, origin, root, dq):
        error = origin.equilibrium.tip-np.asarray(self.goal)
        next_error = root.tip-np.asarray(self.goal)
        before, after = .5*float(error@error), .5*float(next_error@next_error)
        slope = self.slope(origin, dq)
        upper = before+self.armijo_fraction*slope
        result = {"v_before_m2": before, "v_after_m2": after, "directional_derivative_m2": slope,
                  "armijo_upper_bound_m2": upper, "armijo_fraction": self.armijo_fraction,
                  "goal_m": list(self.goal)}
        if not all(np.isfinite(v) for v in (before, after, slope, upper)) or slope >= 0:
            raise _Reject("goal_not_descent_direction", result)
        if not after < before or after > upper:
            raise _Reject("goal_decrease_failed", result)
        return result


class _Reject(Exception):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = {} if details is None else details


def _ordering(solver, q):
    """Mechanical event identities in order; equality must hold for a segment.

    Within a straight line in joint space, event distances are affine. An
    unchanged strict ordering at both ends prevents an intervening crossing.
    Coincidences themselves are rejected by equilibrium_sensitivity.
    """
    ends = solver.lengths+q[:solver.n]
    starts = ends-np.array([t.length_curved for t in solver.tubes])
    events = [(0., "template")]+[(float(v), f"tip_{i}") for i, v in enumerate(ends)]
    events += [(float(v), f"curve_{i}") for i, v in enumerate(starts) if 0 < v < ends.max()]
    return tuple(name for _, name in sorted(events))


class _Meter:
    def __init__(self):
        self.data = {"equilibrium_calls": 0, "sensitivity_calls": 0, "stability_calls": 0,
                     "failed_calls": 0, "rhs_evaluations_known": 0,
                     "failed_calls_without_rhs_counts": 0, "call_seconds": 0.}

    def call(self, kind, function, *args, **kwargs):
        self.data[kind+"_calls"] += 1
        started = time.perf_counter()
        try:
            result = function(*args, **kwargs)
        except (EquilibriumError, ValueError, np.linalg.LinAlgError):
            self.data["failed_calls"] += 1
            # Step 1/2 exceptions do not expose partial RHS counters. Report
            # that gap explicitly; all attempted function calls are counted.
            self.data["failed_calls_without_rhs_counts"] += 1
            raise
        else:
            self.data["rhs_evaluations_known"] += int(result.diagnostics.get("rhs_evaluations", 0))
            return result
        finally:
            self.data["call_seconds"] += time.perf_counter()-started


class BranchTracker:
    """Track from an explicitly supplied equilibrium; no hidden root recovery.

    The constructor verifies the supplied root instead of cold-solving it again.
    ``state`` returns a detached copy, including the branch's base torsional
    strain. The caller must retain this branch state in future RL interfaces.
    """

    def __init__(self, solver, initial_equilibrium, options=None):
        self._solver = deepcopy(solver)
        self._options = TrackingOptions() if options is None else options
        if not isinstance(self._options, TrackingOptions):
            raise ValueError("options must be TrackingOptions")
        meter = _Meter()
        root = deepcopy(initial_equilibrium)
        self._totals = dict(meter.data)
        try:
            self._current = self._analyze(root, 0, meter)
        except (_Reject, EquilibriumError, ValueError, np.linalg.LinAlgError) as exc:
            raise BranchInitializationError(str(exc), {**meter.data, "accepted": False,
                                                      "rejection_details": getattr(exc, "details", {})}) from exc
        self._add_totals(meter)
        self.initialization_diagnostics = {**meter.data, "accepted": True,
                                           "supplied_equilibrium_solve_included": False}

    @property
    def options(self):
        return self._options

    @property
    def state(self):
        return deepcopy(self._current)

    @property
    def totals(self):
        return dict(self._totals)

    def _add_totals(self, meter):
        for key, value in meter.data.items():
            self._totals[key] += value

    def _analyze(self, root, accepted_steps, meter):
        s = meter.call("stability", elastic_stability, self._solver, root)
        details = {"elastic_status": s.status, "minimum_eigenvalues": list(s.minimum_eigenvalues)}
        if s.status != "positive_second_variation_on_tested_meshes":
            raise _Reject("elastic_"+s.status, details)
        eigenvalue = float(s.minimum_eigenvalues[-1])
        if not np.isfinite(eigenvalue) or eigenvalue <= self.options.minimum_elastic_eigenvalue:
            raise _Reject("elastic_margin_too_small", details)
        j = meter.call("sensitivity", equilibrium_sensitivity, self._solver, root,
                       condition_limit=self.options.shooting_condition_limit,
                       event_clearance_m=self.options.event_clearance_m)
        if not np.all(np.isfinite(j.tip_jacobian)) or not np.all(np.isfinite(j.base_torsion_jacobian)):
            raise _Reject("nonfinite_sensitivity")
        return TrackedState(root, j, s, accepted_steps)

    def _prediction(self, origin, root, dq):
        opt, length = self.options, self._solver.scale
        dp = root.tip-origin.equilibrium.tip
        predicted = origin.sensitivity.tip_jacobian@dq
        tip_error = float(np.linalg.norm(dp-predicted))
        tip_limit = opt.tip_absolute_tolerance_m+opt.tip_relative_tolerance*np.linalg.norm(dp)
        dz = length*(root.base_torsional_strain-origin.equilibrium.base_torsional_strain)
        predicted_dz = length*(origin.sensitivity.base_torsion_jacobian@dq)
        twist_error = float(np.linalg.norm(dz-predicted_dz, ord=np.inf))
        twist_limit = opt.twist_absolute_tolerance+opt.twist_relative_tolerance*np.linalg.norm(predicted_dz, ord=np.inf)
        return {"tip_error_m": tip_error, "tip_limit_m": float(tip_limit),
                "twist_prediction_error": twist_error, "twist_prediction_limit": float(twist_limit),
                "twist_change": float(np.linalg.norm(dz, ord=np.inf))}

    def _check_prediction(self, values):
        if not all(np.isfinite(v) for v in values.values()):
            raise _Reject("nonfinite_prediction")
        if values["tip_error_m"] > values["tip_limit_m"]:
            raise _Reject("tip_prediction_error")
        if values["twist_prediction_error"] > values["twist_prediction_limit"]:
            raise _Reject("branch_predictor_error")
        if values["twist_change"] > self.options.max_twist_change:
            raise _Reject("branch_twist_change")

    def step(self, delta_q, *, goal_progress=None):
        """Try a bounded fraction of a feasible command, committing all or none.

        Each attempt checks midpoint and endpoint equilibria and a reverse solve.
        Rejected probes never become tracker state. Status ``accepted`` may mean
        partial progress; accepted_fraction refers to projected_delta.
        """
        solver, opt, origin = self._solver, self.options, self._current
        if goal_progress is not None and not isinstance(goal_progress, GoalProgress):
            raise ValueError("goal_progress must be GoalProgress or None")
        requested = solver.constraints._array(delta_q).copy()
        q0 = origin.equilibrium.joints
        target = solver.constraints.project(q0+requested)
        projected = target-q0
        caps = np.r_[np.full(solver.n, opt.max_translation_step_m),
                     np.full(solver.n, opt.max_rotation_step_rad)]
        size = float(np.max(abs(projected)/caps))
        meter, attempts = _Meter(), []
        started = time.perf_counter()
        status, fraction, applied = "held", 0., np.zeros_like(q0)
        if size <= 1e-12:
            status = "no_motion"
        else:
            initial_fraction = min(1., 1/size)
            for backtrack in range(opt.max_backtracks+1):
                trial_fraction = initial_fraction*2.**(-backtrack)
                dq = trial_fraction*projected
                attempt = {"backtrack": backtrack, "fraction": trial_fraction,
                           "accepted": False, "reason": "", "checkpoints": []}
                attempts.append(attempt)
                try:
                    if goal_progress is not None and goal_progress.slope(origin, dq) >= 0:
                        raise _Reject("goal_not_descent_direction")
                    if _ordering(solver, q0) != _ordering(solver, q0+dq):
                        raise _Reject("event_topology_change")
                    previous = origin
                    for part in [.5, 1.]:
                        q = q0+part*dq
                        if not solver.constraints.is_feasible(q):
                            raise _Reject("infeasible_trial")
                        local_dq = q-previous.equilibrium.joints
                        guess = previous.equilibrium.base_torsional_strain+previous.sensitivity.base_torsion_jacobian@local_dq
                        root = meter.call("equilibrium", solver.solve, q, initial_torsion=guess)
                        local = self._prediction(previous, root, local_dq)
                        overall = self._prediction(origin, root, part*dq)
                        record = {"fraction_of_trial": part, "q": q.tolist(),
                                  "tip_m": root.tip.tolist(),
                                  "base_torsion": root.base_torsional_strain.tolist(),
                                  "local_prediction": local, "overall_prediction": overall}
                        attempt["checkpoints"].append(record)
                        if not np.allclose(root.joints, q, rtol=0, atol=1e-12):
                            raise _Reject("solver_joint_mismatch")
                        self._check_prediction(local)
                        self._check_prediction(overall)
                        if goal_progress is not None:
                            record["local_goal_progress"] = goal_progress.check(previous, root, local_dq)
                            record["overall_goal_progress"] = goal_progress.check(origin, root, part*dq)
                        previous = self._analyze(root, origin.accepted_steps+1, meter)
                        record.update(elastic_status=previous.stability.status,
                                      minimum_eigenvalue=float(previous.stability.minimum_eigenvalues[-1]),
                                      shooting_condition_number=previous.sensitivity.diagnostics["shooting_condition_number"])
                    endpoint = previous
                    reverse_guess = (endpoint.equilibrium.base_torsional_strain
                                     -endpoint.sensitivity.base_torsion_jacobian@dq)
                    reverse = meter.call("equilibrium", solver.solve, q0, initial_torsion=reverse_guess)
                    reverse_tip = float(np.linalg.norm(reverse.tip-origin.equilibrium.tip))
                    reverse_twist = float(solver.scale*np.linalg.norm(reverse.base_torsional_strain
                                                                     -origin.equilibrium.base_torsional_strain, ord=np.inf))
                    attempt.update(reverse_tip_error_m=reverse_tip, reverse_twist_error=reverse_twist)
                    if (not np.isfinite(reverse_tip+reverse_twist)
                            or reverse_tip > opt.reverse_tip_tolerance_m or reverse_twist > opt.reverse_twist_tolerance):
                        raise _Reject("reverse_branch_mismatch")
                except (_Reject, EquilibriumError, ValueError, np.linalg.LinAlgError) as exc:
                    attempt["reason"] = str(exc)
                    attempt["rejection_details"] = getattr(exc, "details", {})
                    if str(exc) == "goal_not_descent_direction" and not attempt["checkpoints"]:
                        # Scaling a fixed projected direction cannot change its sign.
                        break
                    continue
                # Only this assignment mutates the accepted mechanical state.
                self._current = endpoint
                status, fraction, applied = "accepted", trial_fraction, dq.copy()
                attempt.update(accepted=True, reason="accepted")
                break
        self._add_totals(meter)
        diagnostics = {**meter.data, "elapsed_seconds": time.perf_counter()-started,
                       "attempts": attempts, "options": asdict(opt),
                       "reason": attempts[-1]["reason"] if attempts else "projected_no_motion",
                       "command_projected": bool(np.any(abs(projected-requested) > 1e-12)),
                       "accepted_steps": self._current.accepted_steps,
                       "goal_progress": asdict(goal_progress) if goal_progress is not None else None,
                       "model_fingerprint": solver.model_fingerprint,
                       "elastic_stability_certified": False, "controller_stability_certified": False,
                       "continuous_path_certified": False, "global_uniqueness_certified": False}
        return TrackingStep(status, requested, projected, applied, fraction, self.state, diagnostics)
