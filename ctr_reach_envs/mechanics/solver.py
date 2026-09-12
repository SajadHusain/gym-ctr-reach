"""Unloaded CTR equilibrium BVP in a torsion-free Bishop reference frame.

The solver returns an equilibrium, not a certificate of uniqueness, elastic
stability, or real-world accuracy. No hidden warm-start state is retained.
"""
from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .geometry import JointConstraints, TubeParameters, segment_tubes
from .integration import ScaleSafeDOP853


class EquilibriumError(RuntimeError):
    """No validated numerical equilibrium was obtained within the solve budget."""


class _RetryBudgetExceeded(RuntimeError):
    """End the local root retry without resetting the shared solve counters."""


@dataclass(frozen=True)
class SolverOptions:
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = 0.01
    boundary_tolerance: float = 1e-7  # max |tip torsional strain| * longest length
    max_shooting_evaluations: int = 250
    max_rhs_evaluations: int = 500_000
    samples_per_segment: int = 17
    # Missing fields in old checkpoint configs preserve their numerical map.
    shooting_strategy: str = "legacy"
    max_restart_evaluations: int = 200

    def __post_init__(self):
        for key in ("rtol", "atol", "max_step", "boundary_tolerance"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("max_shooting_evaluations", "max_rhs_evaluations", "samples_per_segment", "max_restart_evaluations"):
            value = getattr(self, key)
            if isinstance(value, bool) or int(value) != value or value < (2 if key == "samples_per_segment" else 1):
                raise ValueError(f"Invalid {key}")
        if self.shooting_strategy not in ("legacy", "hybr_restarts"):
            raise ValueError("Unknown shooting strategy")


@dataclass(frozen=True)
class Equilibrium:
    joints: np.ndarray
    s: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    angles: np.ndarray
    torsional_strain: np.ndarray
    base_torsional_strain: np.ndarray
    distal_torsional_strain: np.ndarray
    tube_tip_indices: np.ndarray
    elastic_energy_j: float
    diagnostics: dict

    @property
    def tip(self):
        return self.position[self.tube_tip_indices[0]].copy()


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise EquilibriumError("Invalid backbone orientation quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


class EquilibriumSolver:
    def __init__(self, tubes, options=None, minimum_deployed=1e-3):
        self.tubes = tuple(tubes)
        if not self.tubes or any(not isinstance(t, TubeParameters) for t in self.tubes):
            raise ValueError("tubes must contain TubeParameters")
        self.n = len(self.tubes)
        self.lengths = np.array([t.length for t in self.tubes])
        self.constraints = JointConstraints(self.lengths, minimum_deployed)
        for inner, outer in zip(self.tubes[:-1], self.tubes[1:]):
            if inner.diameter_outer > outer.diameter_inner:
                raise ValueError("Adjacent tubes do not fit concentrically")
        self.ei = np.array([t.EI for t in self.tubes])
        self.gj = np.array([t.GJ for t in self.tubes])
        self.scale = float(self.lengths.max())
        self.options = SolverOptions() if options is None else options

    @property
    def model_fingerprint(self):
        """Identify constitutive geometry; integration settings may be refined."""
        payload = json.dumps([asdict(t) for t in self.tubes], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def solve(self, joints, initial_torsion=None):
        q = self.constraints._array(joints).copy()
        if not self.constraints.is_feasible(q):
            raise ValueError("Joint configuration violates the complete extension constraints")
        n, opt = self.n, self.options
        beta, alpha = q[:n], q[n:]
        intervals = segment_tubes(self.tubes, beta)
        counters = {"shooting_evaluations": 0, "rhs_evaluations": 0, "ivp_integrations": 0}

        def integrate(z, shape=False, curvature_fraction=1.0):
            eta = z / self.scale
            # Below the template the guide enforces a straight centerline. Torque
            # is constant there; rotating clamps are at s = beta_i <= 0.
            angles_at_template = alpha - beta * eta
            y = np.r_[angles_at_template, eta]
            if shape:
                y = np.r_[y, np.zeros(3), [1.0, 0.0, 0.0, 0.0], 0.0]
            arc, states = [], []
            for segment_index, interval in enumerate(intervals):
                ei = self.ei * interval.active
                if ei.sum() <= 0:
                    raise EquilibriumError("Empty mechanical segment")
                intrinsic = interval.curvature * curvature_fraction

                def rhs(_, state):
                    counters["rhs_evaluations"] += 1
                    if counters["rhs_evaluations"] > opt.max_rhs_evaluations:
                        raise EquilibriumError("ODE evaluation budget exceeded")
                    theta, uz = state[:n], state[n:2*n]
                    c, s = np.cos(theta), np.sin(theta)
                    rotated = np.c_[c*intrinsic[:, 0] - s*intrinsic[:, 1],
                                    s*intrinsic[:, 0] + c*intrinsic[:, 1]]
                    curvature = np.sum(ei[:, None] * rotated, axis=0) / ei.sum()
                    ux = c*curvature[0] + s*curvature[1]
                    uy = -s*curvature[0] + c*curvature[1]
                    dtheta = uz * interval.active
                    duz = (ei / self.gj) * (ux*intrinsic[:, 1] - uy*intrinsic[:, 0])
                    out = np.r_[dtheta, duz]
                    if shape:
                        quat = state[2*n + 3:2*n + 7]
                        w, x, yq, zq = quat
                        kx, ky = curvature
                        dquat = 0.5 * np.array([-x*kx-yq*ky, w*kx-zq*ky,
                                               w*ky+zq*kx, x*ky-yq*kx])
                        rotation = quaternion_matrix(quat)
                        energy = 0.5 * np.sum(ei*((ux-intrinsic[:, 0])**2 + (uy-intrinsic[:, 1])**2)
                                             + self.gj*interval.active*uz**2)
                        out = np.r_[out, rotation[:, 2], dquat, energy]
                    return out

                samples = np.linspace(interval.start, interval.end, opt.samples_per_segment) if shape else None
                sol = solve_ivp(rhs, (interval.start, interval.end), y, method=ScaleSafeDOP853,
                                rtol=opt.rtol, atol=opt.atol, max_step=opt.max_step, t_eval=samples)
                counters["ivp_integrations"] += 1
                if not sol.success or not np.all(np.isfinite(sol.y)):
                    raise EquilibriumError(f"Rod integration failed: {sol.message}")
                y = sol.y[:, -1]
                if shape:
                    skip = 0 if segment_index == 0 else 1
                    arc.extend(sol.t[skip:])
                    states.extend(sol.y[:, skip:].T)
            if shape:
                return np.array(arc), np.array(states)
            # Ended tubes are frozen, so these are strains AT their own tips.
            return y[n:2*n] * self.scale

        fraction = 1.0

        def residual(z):
            counters["shooting_evaluations"] += 1
            if counters["shooting_evaluations"] > opt.max_shooting_evaluations:
                raise EquilibriumError("Shooting evaluation budget exceeded")
            if not np.all(np.isfinite(z)):
                raise EquilibriumError("Non-finite shooting iterate")
            return integrate(z, curvature_fraction=fraction)

        z0 = np.zeros(n)
        if initial_torsion is not None:
            eta = np.asarray(initial_torsion, dtype=float)
            if eta.shape != (n,) or not np.all(np.isfinite(eta)):
                raise ValueError("initial_torsion must contain one finite strain per tube")
            z0 = eta * self.scale
        first = residual(z0)
        if np.max(np.abs(first)) <= opt.boundary_tolerance:
            z, root_message = z0, "Initial guess satisfies boundary tolerance"
        else:
            solution = root(residual, z0, method="hybr", options={"xtol": 1e-9, "maxfev": opt.max_shooting_evaluations})
            z, root_message = solution.x, str(solution.message)
        final = residual(z)
        retry = {"root_restart_attempts": 0, "root_restart_accepted": False,
                 "root_restart_evaluations": 0, "root_restart_message": "not_attempted",
                 "root_restart_seed": None}
        if (np.max(np.abs(final)) > opt.boundary_tolerance and initial_torsion is None
                and opt.shooting_strategy == "hybr_restarts"):
            # Fixed, goal/history-independent guesses for z = L * base torsion.
            # Project each signed unit vector onto zero total base torque.
            # This is a bounded numerical search, not an elastic-stability test.
            retry_start = counters["shooting_evaluations"]

            def retry_residual(candidate):
                if counters["shooting_evaluations"]-retry_start >= opt.max_restart_evaluations:
                    raise _RetryBudgetExceeded("Root restart evaluation budget exceeded")
                return residual(candidate)

            for axis in range(n):
                if retry["root_restart_accepted"]:
                    break
                for sign in (-1, 1):
                    guess = sign*np.eye(n)[axis]
                    guess -= (self.gj @ guess)/self.gj.sum()
                    if not np.any(guess):
                        continue
                    retry["root_restart_attempts"] += 1
                    try:
                        trial = root(retry_residual, guess, method="hybr",
                                     options={"xtol": 1e-9, "maxfev": 45})
                        error = retry_residual(trial.x)
                        retry["root_restart_message"] = str(trial.message)
                        # scipy success alone is never an acceptance criterion.
                        if np.max(np.abs(error)) <= opt.boundary_tolerance:
                            z, final, root_message = trial.x, error, str(trial.message)
                            retry["root_restart_accepted"] = True
                            retry["root_restart_seed"] = {"axis": axis, "sign": sign}
                            break
                    except _RetryBudgetExceeded as exc:
                        retry["root_restart_message"] = str(exc)
                        break
                    finally:
                        retry["root_restart_evaluations"] = counters["shooting_evaluations"]-retry_start
                if retry["root_restart_evaluations"] >= opt.max_restart_evaluations:
                    break
        used_continuation = False
        continuation_stages = 0
        if np.max(np.abs(final)) > opt.boundary_tolerance and initial_torsion is None:
            # Deterministic continuation from straight intrinsic geometry. This
            # changes only the root-finding path; the returned shape always uses
            # the full prescribed curvatures. No workspace points are resampled.
            used_continuation = True
            accepted_fraction, step, z = 0.0, 0.25, np.zeros(n)
            while accepted_fraction < 1.0:
                fraction = min(1.0, accepted_fraction + step)
                trial = root(residual, z, method="hybr", options={"xtol": 1e-9, "maxfev": 45})
                error = residual(trial.x)
                if np.max(np.abs(error)) <= opt.boundary_tolerance:
                    z = trial.x
                    accepted_fraction = fraction
                    continuation_stages += 1
                    step = min(.25, step*1.5)
                    root_message = str(trial.message)
                else:
                    step *= .5
                    if step < 1/128:
                        raise EquilibriumError(f"Continuation failed at curvature fraction {fraction:.6g}; residual {np.max(abs(error)):.3g}")
            fraction = 1.0
            final = residual(z)
        if np.max(np.abs(final)) > opt.boundary_tolerance:
            raise EquilibriumError(f"Free-end torsion residual {np.max(np.abs(final)):.3g} exceeds tolerance; {root_message}")
        arc, states = integrate(z, shape=True)
        final_shape_residual = states[-1, n:2*n] * self.scale
        if np.max(np.abs(final_shape_residual)) > opt.boundary_tolerance:
            raise EquilibriumError("Full-shape integration did not preserve the BVP boundary tolerance")
        rotations = np.stack([quaternion_matrix(row[2*n+3:2*n+7]) for row in states])
        ends = self.lengths + beta
        tip_indices = np.array([int(np.argmin(abs(arc-end))) for end in ends])
        if np.max(np.abs(arc[tip_indices] - ends)) > 16*np.finfo(float).eps*self.scale:
            raise EquilibriumError("A tube tip was not represented by an exact segment event")
        eta = z / self.scale
        # Energy inside the straight guide: intrinsic bending is suppressed and
        # torsional strain is constant. The bending term is independent of twist.
        guide_curved = np.maximum(0, np.array([t.length_curved for t in self.tubes]) - ends)
        intrinsic_squared = np.array([t.x_curvature**2+t.y_curvature**2 for t in self.tubes])
        guide_energy = 0.5*np.sum(self.gj*(-beta)*eta**2 + self.ei*guide_curved*intrinsic_squared)
        diag = {**counters, "options": asdict(opt), "integrator": "scale_safe_dop853_v1", "root_message": root_message,
                "shooting_algorithm": "hybr_restarts_continuation_v2" if opt.shooting_strategy == "hybr_restarts" else "hybr_continuation_v1",
                **retry,
                "model_fingerprint": self.model_fingerprint,
                "boundary_residual_scaled": float(np.max(np.abs(final_shape_residual))),
                "base_torque_sum_nm": float(self.gj @ eta),
                "max_quaternion_norm_error": float(np.max(abs(np.linalg.norm(states[:, 2*n+3:2*n+7], axis=1)-1))),
                "elastic_stability": "not_certified", "uniqueness": "not_certified",
                "initial_guess": "zero" if initial_torsion is None else "explicit",
                "used_curvature_continuation": used_continuation,
                "continuation_stages": continuation_stages}
        return Equilibrium(q, arc, states[:, 2*n:2*n+3].copy(), rotations,
                           states[:, :n].copy(), states[:, n:2*n].copy(), eta.copy(),
                           states[tip_indices, n+np.arange(n)].copy(), tip_indices,
                           float(states[-1, -1]+guide_energy), diag)
