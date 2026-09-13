"""Analytical sensitivity of the original Model IVP, including moving segments.

See docs/original_ivp_training.md for the equations and model limitations.
Finite differences are used only in tests, never in this implementation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from ctr_reach_envs.envs.CTR_Python import Segment
from ctr_reach_envs.envs.model import Model


class IVPSensitivityError(RuntimeError):
    pass


@dataclass
class IVPSensitivity:
    tip: np.ndarray
    jacobian: np.ndarray
    rhs_evaluations: int
    segment_mode: str


def field_and_derivative(y, ux0, uy0, ei, gj):
    """Return F and its analytical 18-by-18 state derivative F_y."""
    theta = y[3:6]
    angle = theta[:, None] - theta[None, :]
    c, s = np.cos(angle), np.sin(angle)
    w = ei / ei.sum()
    xt = c * ux0 + s * uy0
    yt = -s * ux0 + c * uy0
    ux, uy = xt @ w, yt @ w
    dx = np.diag(yt @ w) - yt * w
    dy = -np.diag(xt @ w) + xt * w
    ratio = np.divide(ei, gj, out=np.zeros(3), where=ei != 0)
    A = np.zeros((18, 18))
    A[:3, 3:6] = ratio[:, None] * (uy0[:, None] * dx - ux0[:, None] * dy)
    A[3:6, :3] = np.diag((ei != 0).astype(float))
    A[np.arange(6, 9), np.array([11, 14, 17])] = 1.
    R = y[9:].reshape(3, 3)
    hat = np.array([[0., -y[0], uy[0]], [y[0], 0., -ux[0]], [-uy[0], ux[0], 0.]])
    A[9:, 9:] = np.kron(np.eye(3), hat.T)
    A[9:, 0] = (R @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])).ravel()
    for j in range(3):
        dhat = np.array([[0., 0., dy[0, j]], [0., 0., -dx[0, j]], [-dy[0, j], dx[0, j], 0.]])
        A[9:, j + 3] = (R @ dhat).ravel()
    return Model._ode_eq(0., y, ux0, uy0, ei, gj), A


def tip_sensitivity(model, joint, system=0, *, rtol=1e-9, atol=1e-11, max_rhs=20000):
    """Integrate y and dy/dq without modifying model backbone or random state."""
    q = np.asarray(joint, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("joint must be a finite six-vector")
    segment = Segment(*model.current_sys_parameters[system], q[:3],
                      quantize=model.segment_mode == "legacy")
    if not segment.derivative_valid:
        raise IVPSensitivityError(segment.derivative_reason)
    c, s = np.cos(q[3]), np.sin(q[3])
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    y = np.r_[np.zeros(3), q[3:], np.zeros(3), R.ravel()]
    S = np.zeros((18, 6))
    S[3:6, 3:6] = np.eye(3)
    S[9:, 3] = np.array([[-s, -c, 0.], [c, -s, 0.], [0., 0., 0.]]).ravel()
    endpoints = np.r_[0., segment.S]
    gradients = np.vstack((np.zeros((1, 3)), segment.S_beta))
    count = 0
    active = 0
    for i in range(len(segment.S)):
        h = endpoints[i + 1] - endpoints[i] - 1e-6
        dh = np.r_[gradients[i + 1] - gradients[i], np.zeros(3)]
        if abs(h) < 1e-9 and np.any(dh):
            raise IVPSensitivityError("segment integration switch")
        if h <= 0:
            continue
        active += 1
        properties = (segment.U_x[:, i], segment.U_y[:, i], segment.EI[:, i], segment.GJ[:, i])

        def rhs(_, z):
            nonlocal count
            count += 1
            if count > max_rhs:
                raise IVPSensitivityError("variational IVP evaluation budget exceeded")
            f, A = field_and_derivative(z[:18], *properties)
            sensitivity = z[18:].reshape(18, 6)
            return np.r_[h * f, (h * (A @ sensitivity) + f[:, None] * dh).ravel()]

        result = solve_ivp(rhs, (0., 1.), np.r_[y, S.ravel()], method="DOP853", rtol=rtol, atol=atol)
        if not result.success or not np.all(np.isfinite(result.y[:, -1])):
            raise IVPSensitivityError("variational IVP failed")
        y, S = result.y[:18, -1], result.y[18:, -1].reshape(18, 6)
    if not active:
        raise IVPSensitivityError("no active backbone")
    return IVPSensitivity(y[6:9].copy(), S[6:9].copy(), count, model.segment_mode)
