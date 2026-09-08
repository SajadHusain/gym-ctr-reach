"""Local equilibrium derivatives from variational ODEs and implicit shooting.

All derivatives refer to the supplied equilibrium branch, at fixed tube geometry.
No perturbation is projected and no optimizer or finite difference is used to
compute the Jacobian. Coincident moving events are explicitly unsupported.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.integrate import solve_ivp

from .geometry import segment_tubes
from .solver import EquilibriumError, quaternion_matrix


class SensitivityError(EquilibriumError):
    """A local, numerically regular sensitivity could not be established."""


@dataclass(frozen=True)
class Sensitivity:
    # q order: beta in metres, alpha in radians. Tip units: metres.
    tip_jacobian: np.ndarray
    base_torsion_jacobian: np.ndarray
    shooting_jacobian: np.ndarray
    diagnostics: dict


def _checked_input(solver, equilibrium):
    if equilibrium.diagnostics.get("model_fingerprint") != solver.model_fingerprint:
        raise SensitivityError("Equilibrium belongs to a different model, or has no model fingerprint")
    q = solver.constraints._array(equilibrium.joints)
    eta = np.asarray(equilibrium.base_torsional_strain, dtype=float)
    if not solver.constraints.is_feasible(q) or eta.shape != (solver.n,) or not np.all(np.isfinite(eta)):
        raise SensitivityError("Invalid equilibrium joints or shooting state")
    return q.copy(), eta.copy()


def _field(solver, interval, state, derivatives=False):
    """Bishop-frame ODE and its analytical state Jacobian (no energy state)."""
    n = solver.n
    theta, uz = state[:n], state[n:2*n]
    ei = solver.ei * interval.active
    intrinsic = interval.curvature
    c, s = np.cos(theta), np.sin(theta)
    rotated = np.c_[c*intrinsic[:, 0]-s*intrinsic[:, 1],
                    s*intrinsic[:, 0]+c*intrinsic[:, 1]]
    kx, ky = curvature = (ei[:, None]*rotated).sum(axis=0)/ei.sum()
    eta_dot = ei/solver.gj * (kx*rotated[:, 1]-ky*rotated[:, 0])
    quat = state[2*n+3:2*n+7]
    w, x, y, z = quat
    omega = .5*np.array([[0, -kx, -ky, 0], [kx, 0, 0, -ky],
                         [ky, 0, 0, kx], [0, ky, -kx, 0]])
    out = np.r_[uz*interval.active, eta_dot, quaternion_matrix(quat)[:, 2], omega@quat]
    if not derivatives:
        return out
    a = np.zeros((2*n+7, 2*n+7))
    a[:n, n:2*n] = np.diag(interval.active)
    dk = (ei[:, None]*np.c_[-rotated[:, 1], rotated[:, 0]]/ei.sum()).T
    a[n:2*n, :n] = (ei/solver.gj)[:, None]*(rotated[:, 1, None]*dk[0]
                                                          - rotated[:, 0, None]*dk[1])
    a[n:2*n, :n] += np.diag(ei/solver.gj*(rotated@curvature))
    # p' uses a normalized quaternion, so differentiate that normalization too.
    norm = np.linalg.norm(quat)
    wn, xn, yn, zn = quat/norm
    dp = np.array([[2*yn, 2*zn, 2*wn, 2*xn],
                   [-2*xn, -2*wn, 2*zn, 2*yn], [0, -4*xn, -4*yn, 0]])
    a[2*n:2*n+3, 2*n+3:2*n+7] = dp@(np.eye(4)-np.outer(quat/norm, quat/norm))/norm
    a[2*n+3:2*n+7, 2*n+3:2*n+7] = omega
    dq_dk = .5*np.array([[-x, -y], [w, -z], [z, w], [-y, x]])
    a[2*n+3:2*n+7, :n] = dq_dk@dk
    return out, a


def _event_gradients(solver, q, intervals, clearance):
    """Event derivatives within one ordering cell; reject ambiguous collisions."""
    n = solver.n
    ends = solver.lengths+q[:n]
    starts = ends-np.array([t.length_curved for t in solver.tubes])
    events = [(0., np.zeros(n))]
    for i in range(n):
        gradient = np.eye(n)[i]
        events.append((ends[i], gradient))
        # Starts just outside the domain also matter to local differentiability.
        if -clearance <= starts[i] <= ends.max()+clearance:
            events.append((starts[i], gradient))
    for i, (s, ds) in enumerate(events):
        for t, dt in events[i+1:]:
            if abs(s-t) <= clearance and not np.array_equal(ds, dt):
                raise SensitivityError("Coincident or near-coincident moving segment events: full joint Jacobian is not supported here")
    gradients = []
    for point in [intervals[0].start]+[x.end for x in intervals]:
        s, ds = min(events, key=lambda event: abs(event[0]-point))
        if abs(s-point) > 16*np.finfo(float).eps*solver.scale:
            raise SensitivityError("Unable to differentiate a segment event")
        gradients.append(ds)
    return np.asarray(gradients)


def _trajectory(solver, equilibrium, variational=False, event_clearance_m=1e-9):
    """Reintegrate the given shooting root; optionally propagate sensitivities.

    Each segment is integrated on t in [0,1]. Differentiating its length in
    y_t = (b(q)-a(q)) F(y) includes all moving-interface terms exactly within
    a fixed event ordering. Unknowns are normalized q and z = L * base_twist.
    """
    q, eta = _checked_input(solver, equilibrium)
    n, m, p = solver.n, 2*solver.n+7, 3*solver.n
    qscale = np.r_[np.full(n, solver.scale), np.ones(n)]
    intervals = segment_tubes(solver.tubes, q[:n])
    gradients = _event_gradients(solver, q, intervals, event_clearance_m) if variational else None
    state = np.r_[q[n:]-q[:n]*eta, eta, np.zeros(3), [1., 0., 0., 0.]]
    if variational:
        sensitivity = np.zeros((m, p))
        sensitivity[:n, :n] = -np.diag(eta)*solver.scale
        sensitivity[:n, n:2*n] = np.eye(n)
        sensitivity[:n, 2*n:] = -np.diag(q[:n])/solver.scale
        sensitivity[n:2*n, 2*n:] = np.eye(n)/solver.scale
        state = np.r_[state, sensitivity.ravel()]
    counters = {"rhs_evaluations": 0, "ivp_integrations": 0}
    dense = []
    opt = solver.options
    for index, interval in enumerate(intervals):
        length = interval.end-interval.start
        if variational:
            dlength = np.r_[(gradients[index+1]-gradients[index])*solver.scale, np.zeros(2*n)]
        def rhs(_, y):
            counters["rhs_evaluations"] += 1
            if counters["rhs_evaluations"] > opt.max_rhs_evaluations:
                raise SensitivityError("Analysis ODE evaluation budget exceeded")
            if not variational:
                return length*_field(solver, interval, y)
            field, a = _field(solver, interval, y[:m], derivatives=True)
            jac = y[m:].reshape(m, p)
            return np.r_[length*field, (length*(a@jac)+np.outer(field, dlength)).ravel()]
        sol = solve_ivp(rhs, (0., 1.), state, method="DOP853", dense_output=True,
                        rtol=opt.rtol, atol=opt.atol, max_step=min(1., opt.max_step/length))
        counters["ivp_integrations"] += 1
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise SensitivityError(f"Analysis integration failed: {sol.message}")
        state = sol.y[:, -1]
        dense.append(sol.sol)
    residual = float(np.max(abs(state[n:2*n]))*solver.scale)
    discrepancy = float(np.linalg.norm(state[2*n:2*n+3]-equilibrium.tip))
    if residual > opt.boundary_tolerance or discrepancy > max(1e-8, 100*opt.atol):
        raise SensitivityError("Reintegration does not match the supplied equilibrium branch and boundary residual")
    counters.update(boundary_residual_scaled=residual, reintegration_tip_discrepancy_m=discrepancy)
    return state, intervals, dense, counters, qscale


def equilibrium_sensitivity(solver, equilibrium, *, condition_limit=1e8, event_clearance_m=1e-9):
    """Compute dp/dq using R(z,q)=0 and p=P(z,q) on one regular branch.

    The condition number concerns the shooting equations, NOT elastic stability.
    The derivative is before joint projection; use actual applied increments for
    local predictions, and a separate chain rule for a projected policy action.
    """
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    if not np.isfinite(event_clearance_m) or event_clearance_m <= 0:
        raise ValueError("event_clearance_m must be finite and positive")
    started = time.perf_counter()
    state, _, _, counters, qscale = _trajectory(solver, equilibrium, True, event_clearance_m)
    n, m = solver.n, 2*solver.n+7
    partial = state[m:].reshape(m, 3*n)
    rz = solver.scale*partial[n:2*n, 2*n:]
    rq = solver.scale*partial[n:2*n, :2*n]
    condition = float(np.linalg.cond(rz))
    smallest = float(np.linalg.svd(rz, compute_uv=False)[-1])
    # cond alone misses a 1x1 nearly singular matrix (whose cond is always one).
    if not np.isfinite(condition) or condition > condition_limit or smallest < 1/condition_limit:
        raise SensitivityError(f"Shooting derivative is numerically singular: condition={condition:.6g}, sigma_min={smallest:.6g}")
    dz_dq_normalized = -np.linalg.solve(rz, rq)
    tip = partial[2*n:2*n+3, :2*n]+partial[2*n:2*n+3, 2*n:]@dz_dq_normalized
    diagnostics = {**counters, "shooting_condition_number": condition,
                   "shooting_smallest_singular_value": smallest,
                   "implicit_equation_residual": float(np.linalg.norm(rz@dz_dq_normalized+rq, ord=np.inf)),
                   "seconds": time.perf_counter()-started,
                   "method": "analytic_variational_ode_implicit_shooting",
                   "q_order": "beta_m_then_alpha_rad", "q_normalization": qscale.tolist(),
                   "model_fingerprint": solver.model_fingerprint,
                   "equilibrium_base_torsion": equilibrium.base_torsional_strain.tolist(),
                   "local_branch_only": True, "elastic_stability_certified": False,
                   "joint_projection_differentiated": False,
                   "event_clearance_m": event_clearance_m,
                   "additional_equilibrium_solves": 0}
    return Sensitivity(tip/qscale, dz_dq_normalized/solver.scale/qscale, rz, diagnostics)
