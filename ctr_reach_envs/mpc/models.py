"""Prediction models for Khadem et al. (2020), DOI 10.1109/TRO.2020.2991651.

The paper model optimizes base torsion explicitly, imposing zero distal torsion
as NLP equalities. The original-IVP adapter deliberately retains the RL plant's
zero base torsion instead. These are different mechanical models.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.integrate import solve_ivp

from ctr_reach_envs.mechanics.geometry import JointConstraints, segment_tubes
from ctr_reach_envs.mechanics.integration import ScaleSafeDOP853
from ctr_reach_envs.mechanics.sensitivity import _field
from ctr_reach_envs.mechanics.solver import EquilibriumError


@dataclass(frozen=True)
class Prediction:
    tip: np.ndarray
    residual: np.ndarray


class OriginalIVPModel:
    """Private FK instance, so planning cannot mutate the live environment."""
    torsion_count = 0

    def __init__(self, model, system=0):
        self.model, self.system = model, system
        self.calls, self.seconds = 0, 0.

    def predict(self, joints, torsion_scaled=None):
        start = time.perf_counter()
        self.calls += 1
        try:
            tip = np.asarray(self.model.forward_kinematics(joints, self.system), dtype=float)
            if tip.shape != (3,) or not np.all(np.isfinite(tip)):
                raise EquilibriumError("Invalid original-IVP tip")
            return Prediction(tip.copy(), np.empty(0))
        finally:
            self.seconds += time.perf_counter()-start


class PaperIVPModel:
    """One IVP, without a nested shooting solve, per candidate (q, L*psi).

    q=[beta(m), alpha(rad)], psi=base torsional strain(1/m), L=max tube length.
    Uses the repository's unloaded Bishop-frame equations, equivalent to the
    paper's material-frame formulation. Ended tube strains freeze at their OWN
    distal ends, not at an offset sample or at the longest tube's tip.
    """
    def __init__(self, solver):
        self.solver = solver
        self.torsion_count = solver.n
        self.calls, self.seconds, self.rhs_evaluations = 0, 0., 0

    def predict(self, joints, torsion_scaled):
        started = time.perf_counter()
        self.calls += 1
        try:
            solver, n = self.solver, self.solver.n
            q = solver.constraints._array(joints)
            z = np.asarray(torsion_scaled, dtype=float)
            if z.shape != (n,) or not np.all(np.isfinite(z)):
                raise ValueError("Expected one finite scaled base strain per tube")
            eta = z / solver.scale
            state = np.r_[q[n:]-q[:n]*eta, eta, np.zeros(3), [1., 0., 0., 0.]]
            tip, calls = None, 0
            intervals = segment_tubes(solver.tubes, q[:n])
            tip_s = solver.lengths[0]+q[0]
            for interval in intervals:
                length = interval.end-interval.start

                def rhs(_, y):
                    nonlocal calls
                    calls += 1
                    self.rhs_evaluations += 1
                    if calls > solver.options.max_rhs_evaluations:
                        raise EquilibriumError("MPC prediction exceeded its ODE budget")
                    return length * _field(solver, interval, y)

                sol = solve_ivp(rhs, (0., 1.), state, method=ScaleSafeDOP853,
                    rtol=solver.options.rtol, atol=solver.options.atol,
                    max_step=min(1., solver.options.max_step/length))
                if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
                    raise EquilibriumError("MPC IVP prediction failed")
                state = sol.y[:, -1]
                if abs(interval.end-tip_s) <= 16*np.finfo(float).eps*solver.scale:
                    tip = state[2*n:2*n+3].copy()
            if tip is None:
                raise EquilibriumError("Innermost tube endpoint was not integrated")
            return Prediction(tip, solver.scale*state[n:2*n].copy())
        finally:
            self.seconds += time.perf_counter()-started


def original_joint_step(q, action, action_scales, n_substeps, lengths, constrain_alpha=False):
    """Exact NumPy counterpart of the original plant's repeated ordered clips."""
    q = np.asarray(q, dtype=float).copy()
    n = len(lengths)
    increment = np.clip(np.asarray(action, dtype=np.float32), -1., 1.) * action_scales
    for _ in range(n_substeps):
        q[:n] = np.clip(q[:n]+increment[:n], -np.asarray(lengths)+.001, 0.)
        q[n:] += increment[n:]
        if constrain_alpha:
            q[n:] = np.clip(q[n:], -np.pi, np.pi)
        for i in range(1, n):
            q[i-1] = max(min(q[i-1], q[i]), lengths[i]-lengths[i-1]+q[i])
    return q
