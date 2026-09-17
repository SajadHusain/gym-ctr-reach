"""Scaled, constrained nonlinear MPC using full CTR predictions at every node.

Independent implementation of the finite-horizon NLP in Khadem et al. (2020).
SLSQP replaces their custom interior-point solver. Derivatives use stage-local
finite differences of nonlinear IVPs, not a constant Jacobian rollout.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass(frozen=True)
class MPCOptions:
    horizon: int = 2
    max_iterations: int = 25
    max_model_evaluations: int = 400
    tracking_scale_m: float = .01
    move_weight: float = .001
    terminal_weight: float = 1.
    finite_difference_step: float = 1e-4
    boundary_tolerance: float = 1e-5
    initial_tip_tolerance_m: float = 2e-5
    optimizer_ftol: float = 1e-6
    max_scaled_torsion: float = 50.
    base_separation_m: float = 0.

    def __post_init__(self):
        for key in ("horizon", "max_iterations", "max_model_evaluations"):
            value = getattr(self, key)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("tracking_scale_m", "terminal_weight", "finite_difference_step",
                    "boundary_tolerance", "initial_tip_tolerance_m", "optimizer_ftol",
                    "max_scaled_torsion"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("move_weight", "base_separation_m"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")


@dataclass(frozen=True)
class MPCResult:
    joints: np.ndarray
    torsion_scaled: np.ndarray
    predicted_tip: np.ndarray
    diagnostics: dict


class _BudgetExceeded(RuntimeError):
    pass


class NonlinearMPC:
    def __init__(self, model, constraints, joint_step_caps, options=None):
        self.model, self.constraints = model, constraints
        self.caps = np.asarray(joint_step_caps, dtype=float).copy()
        if (self.caps.shape != (2*constraints.n,) or not np.all(np.isfinite(self.caps))
                or np.any(self.caps <= 0)):
            raise ValueError("One positive finite step cap is required per joint")
        self.options = options or MPCOptions()
        self._warm = None

    def reset(self):
        self._warm = None

    def plan(self, joints, measured_tip, goal, *, torsion_scaled=None):
        """Optimize H future configurations, execute the first, then replan.

        The measured current q/tip is fixed, not an optimizable first stage.
        goal can be one endpoint (3,) or a preview of H future targets (H,3).
        Only independently feasible, non-worsening candidates are executed.
        An unsuccessful solve returns an explicit hold, never unchecked output.
        """
        started = time.perf_counter()
        opt, constraints = self.options, self.constraints
        q = constraints._array(joints).copy()
        n, nq, nt, h = constraints.n, q.size, self.model.torsion_count, opt.horizon
        width = nq+nt
        tip = np.asarray(measured_tip, dtype=float)
        targets = np.asarray(goal, dtype=float)
        if targets.shape == (3,):
            targets = np.broadcast_to(targets, (h, 3)).copy()
        if tip.shape != (3,) or targets.shape != (h, 3) or not np.all(np.isfinite(np.r_[tip, targets.ravel()])):
            raise ValueError("Expected a finite measured tip and goal/preview")
        if not constraints.is_feasible(q):
            raise ValueError("MPC requires a feasible current configuration")
        eps = opt.base_separation_m
        if eps and (q[n-1] > -eps or np.any(np.diff(q[:n]) < eps)):
            raise ValueError("Current configuration violates requested base separation")
        z = np.zeros(nt) if torsion_scaled is None else np.asarray(torsion_scaled, dtype=float)
        if z.shape != (nt,) or not np.all(np.isfinite(z)) or np.any(abs(z) > opt.max_scaled_torsion):
            raise ValueError("Invalid current scaled torsion")
        # Variables at each node: (q_k-q_current)/step_caps, L*psi_k.
        lower = np.r_[(-constraints.lengths+constraints.minimum_deployed-q[:n])/self.caps[:n],
                      np.full(n, -h), np.full(nt, -opt.max_scaled_torsion)]
        upper = np.r_[-q[:n]/self.caps[:n], np.full(n, h), np.full(nt, opt.max_scaled_torsion)]
        if constraints.constrain_alpha:
            lower[n:nq] = np.maximum(lower[n:nq], (-np.pi-q[n:])/self.caps[n:])
            upper[n:nq] = np.minimum(upper[n:nq], (np.pi-q[n:])/self.caps[n:])
        # Linear rows are normalized to dimensionless units before SLSQP.
        rows, bounds = [], []
        for k in range(h):
            for a, b in zip(constraints.A, constraints.b):
                row = np.zeros((h, width)); row[k, :n] = a*self.caps[:n]
                scale = np.max(abs(row))
                rows.append(row.ravel()/scale); bounds.append((b-a@q[:n])/scale)
            if eps:
                for a in np.vstack((np.eye(n)[-1], np.eye(n)[:-1]-np.eye(n)[1:])):
                    row = np.zeros((h, width)); row[k, :n] = a*self.caps[:n]
                    scale = np.max(abs(row))
                    rows.append(row.ravel()/scale); bounds.append((-eps-a@q[:n])/scale)
            for j in range(nq):
                row = np.zeros((h, width)); row[k, j] = 1.
                if k: row[k-1, j] = -1.
                rows.extend((row.ravel(), -row.ravel())); bounds.extend((1., 1.))
        A, b = np.asarray(rows), np.asarray(bounds)
        x_hold = np.tile(np.r_[np.zeros(nq), z], (h, 1)).ravel()
        cache, derivatives = {}, {}
        evaluations = 0

        def stage(v):
            nonlocal evaluations
            key = v.tobytes()
            if key not in cache:
                if evaluations >= opt.max_model_evaluations:
                    raise _BudgetExceeded("model evaluation budget exhausted")
                evaluations += 1
                pred = self.model.predict(q+self.caps*v[:nq], v[nq:])
                value = np.r_[pred.tip, pred.residual]
                if value.shape != (3+nt,) or not np.all(np.isfinite(value)):
                    raise RuntimeError("Nonfinite or malformed MPC prediction")
                cache[key] = value
            return cache[key]

        def linearize(v):
            key = v.tobytes()
            if key not in derivatives:
                base = stage(v)
                derivative = np.empty((3+nt, width))
                for j in range(width):
                    step = opt.finite_difference_step
                    if v[j]+step > upper[j]: step = -step
                    if v[j]+step < lower[j]:
                        step = .5*(upper[j]-lower[j])
                    perturb = v.copy(); perturb[j] += step
                    derivative[:, j] = (stage(perturb)-base)/step
                derivatives[key] = derivative
            return derivatives[key]

        weights = np.ones(h); weights[-1] = opt.terminal_weight
        best_x, best_value, best_pred = x_hold.copy(), np.inf, None
        baseline = None

        def predictions(x):
            return np.array([stage(v) for v in x.reshape(h, width)])

        def is_feasible(x, values):
            v = x.reshape(h, width)
            return (np.max(A@x-b) <= 1e-8 and np.all(v >= lower-1e-10)
                and np.all(v <= upper+1e-10)
                and (not nt or np.max(abs(values[:, 3:])) <= opt.boundary_tolerance))

        def objective(x):
            nonlocal best_x, best_value, best_pred
            values = predictions(x)
            v = x.reshape(h, width)
            increments = np.diff(np.vstack((np.zeros(nq), v[:, :nq])), axis=0)
            loss = float(np.sum(weights[:, None]*((values[:, :3]-targets)/opt.tracking_scale_m)**2)
                         + opt.move_weight*np.sum(increments**2))
            if loss < best_value and is_feasible(x, values):
                best_x, best_value, best_pred = x.copy(), loss, values.copy()
            return loss

        def jacobian(x):
            v, values = x.reshape(h, width), predictions(x)
            jac = np.empty_like(v)
            for k in range(h):
                jac[k] = (2*weights[k]*(values[k, :3]-targets[k])/opt.tracking_scale_m**2) @ linearize(v[k])[:3]
            increments = np.diff(np.vstack((np.zeros(nq), v[:, :nq])), axis=0)
            penalty = increments.copy(); penalty[:-1] -= increments[1:]
            jac[:, :nq] += 2*opt.move_weight*penalty
            return jac.ravel()

        def equalities(x):
            return predictions(x)[:, 3:].ravel()

        def equality_jacobian(x):
            out = np.zeros((h*nt, h*width))
            for k, v in enumerate(x.reshape(h, width)):
                out[k*nt:(k+1)*nt, k*width:(k+1)*width] = linearize(v)[3:]
            return out

        status, iterations, converged = "hold", 0, False
        initial_discrepancy = None

        def callback(_):
            nonlocal iterations
            iterations += 1

        try:
            current = stage(x_hold.reshape(h, width)[0])
            initial_discrepancy = float(np.linalg.norm(current[:3]-tip))
            if initial_discrepancy > opt.initial_tip_tolerance_m:
                raise RuntimeError("Measured tip disagrees with the prediction model")
            if nt and np.max(abs(current[3:])) > opt.boundary_tolerance:
                raise RuntimeError("Current torsion does not satisfy the free-tip boundary")
            baseline = objective(x_hold)
            initial = x_hold.copy()
            if self._warm is not None:
                warm_q, warm_z = self._warm
                candidate = np.c_[(warm_q-q)/self.caps, warm_z].ravel()
                if np.all(candidate.reshape(h, width) >= lower) and np.all(candidate.reshape(h, width) <= upper) and np.max(A@candidate-b) <= 1e-8:
                    if objective(candidate) < baseline: initial = candidate
            cons = [LinearConstraint(A, np.full(len(b), -np.inf), b)]
            if nt: cons.append(dict(type="eq", fun=equalities, jac=equality_jacobian))
            result = minimize(objective, initial, jac=jacobian, method="SLSQP",
                callback=callback,
                bounds=Bounds(np.tile(lower, h), np.tile(upper, h)), constraints=cons,
                options=dict(maxiter=opt.max_iterations, ftol=opt.optimizer_ftol))
            objective(result.x)  # independent validation, regardless of solver status
            status, iterations, converged = str(result.message), int(result.nit), bool(result.success)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            status = f"{type(exc).__name__}: {exc}"
        # All returned stages were checked by is_feasible, not just solver.success.
        fallback = best_pred is None or baseline is None or best_value >= baseline-1e-12
        if fallback:
            self.reset()
            return MPCResult(q.copy(), z.copy(), tip.copy(), dict(status=status,
                converged=converged, hold=True, iterations=iterations, model_evaluations=evaluations,
                objective_hold=baseline, objective_selected=baseline, boundary_residual_scaled=None,
                initial_tip_discrepancy_m=initial_discrepancy, planning_seconds=time.perf_counter()-started))
        v = best_x.reshape(h, width)
        future_q, future_z = q+self.caps*v[:, :nq], v[:, nq:]
        self._warm = (np.vstack((future_q[1:], future_q[-1])), np.vstack((future_z[1:], future_z[-1])))
        return MPCResult(future_q[0].copy(), future_z[0].copy(), best_pred[0, :3].copy(),
            dict(status=status, converged=converged, hold=False, iterations=iterations,
                model_evaluations=evaluations, objective_hold=baseline, objective_selected=best_value,
                boundary_residual_scaled=float(np.max(abs(best_pred[:, 3:]))) if nt else 0.,
                initial_tip_discrepancy_m=initial_discrepancy, planning_seconds=time.perf_counter()-started))
