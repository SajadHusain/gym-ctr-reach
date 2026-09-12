"""Evaluation-only constrained damped differential-kinematics comparator.

No learning code imports this controller. The task-space regularizer does not
use its actions as labels. Units are normalized before solving the convex QP.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def desired_displacement(error, gain=.5, max_tip_step=.002):
    error = np.asarray(error, dtype=float)
    if error.shape != (3,) or not np.all(np.isfinite(error)):
        raise ValueError("Expected three finite Cartesian error components")
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(max_tip_step) or max_tip_step <= 0:
        raise ValueError("Gain and target cap must be finite and positive")
    target = gain*error
    return target/max(1., float(np.linalg.norm(target)/max_tip_step))


def constrained_jacobian_action(jacobian, q, scales, constraints, error, *,
                                gain=.5, max_tip_step=.002, cartesian_scale=.002, damping=.05):
    """min ||J D a - d*||^2/c^2 + damping^2 ||a||^2, with joint/action bounds.

The numerical QP has explicit feasibility and optimizer-status checks. Solver failure
is reported; it is never converted to a fictitious successful zero action.
"""
    q, scales, jacobian = np.asarray(q, float), np.asarray(scales, float), np.asarray(jacobian, float)
    n = constraints.n
    if q.shape != (2*n,) or scales.shape != q.shape or jacobian.shape != (3, 2*n):
        raise ValueError("Jacobian, joints and scales have incompatible shapes")
    if not all(np.all(np.isfinite(v)) for v in (q, scales, jacobian)) or np.any(scales <= 0):
        raise ValueError("Invalid mechanics context")
    if not np.isfinite(cartesian_scale) or cartesian_scale <= 0 or not np.isfinite(damping) or damping <= 0:
        raise ValueError("Scale and damping must be positive and finite")
    if not constraints.is_feasible(q):
        raise ValueError("Source joints are infeasible")
    matrix = jacobian*scales[None, :]/cartesian_scale
    target = desired_displacement(error, gain, max_tip_step)/cartesian_scale
    hessian = matrix.T@matrix+damping**2*np.eye(2*n)
    linear = matrix.T@target
    # Scale each inequality by its achievable change in one normalized command.
    rows = np.zeros((len(constraints.b), 2*n))
    rows[:, :n] = constraints.A*scales[:n]
    row_scale = np.linalg.norm(rows, axis=1)
    margins = constraints.b-constraints.A@q[:n]
    rows, margins = rows/row_scale[:, None], margins/row_scale
    result = minimize(lambda a: float(.5*a@hessian@a-linear@a), np.zeros(2*n),
        jac=lambda a: hessian@a-linear, method="SLSQP", bounds=Bounds(-1., 1.),
        constraints=[LinearConstraint(rows, -np.inf, margins)],
        options={"maxiter":200, "ftol":1e-12})
    action = np.clip(result.x, -1., 1.)
    if not result.success or not np.all(np.isfinite(action)) or np.any(rows@action > margins+1e-8):
        raise RuntimeError(f"Constrained Jacobian QP failed: {result.message}")
    if not constraints.is_feasible(q+scales*action, atol=1e-10):
        raise RuntimeError("Constrained Jacobian QP returned infeasible joints")
    return action.astype(np.float32)
