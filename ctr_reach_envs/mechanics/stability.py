"""Mesh-refined second-variation diagnostic for unloaded CTR equilibria.

Positive finite-element eigenvalues do not certify continuum stability. Negative
modes provide numerical evidence of an elastic instability. Neither conclusion
is a Lyapunov certificate for a controller or a proof of global uniqueness.
"""
from dataclasses import dataclass
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

from .sensitivity import _trajectory


@dataclass(frozen=True)
class StabilityDiagnostic:
    status: str
    minimum_eigenvalues: tuple
    mesh_elements_per_segment: tuple
    mode_s: np.ndarray
    mode_angles: np.ndarray
    diagnostics: dict


def bending_hessian(ei, rotated_intrinsic):
    """Hessian of minimized bending energy with respect to Bishop tube angles."""
    weights = np.outer(ei, ei)/ei.sum() * (rotated_intrinsic@rotated_intrinsic.T)
    np.fill_diagonal(weights, 0.)
    return np.diag(weights.sum(axis=1))-weights


def _eigenmode(solver, equilibrium, intervals, dense, elements):
    n = solver.n
    nodes = np.r_[0., np.concatenate([np.linspace(x.start, x.end, elements+1)[1:] for x in intervals])]
    ends = solver.lengths+equilibrium.joints[:n]
    tolerance = 16*np.finfo(float).eps*solver.scale
    present = nodes[:, None] <= ends[None, :]+tolerance
    dofs = np.full(present.shape, -1, dtype=int)
    dofs[present] = np.arange(present.sum())
    count = int(present.sum())
    stiffness, mass = np.zeros((count, count)), np.zeros((count, count))
    points, weights = leggauss(3)
    points, weights = (points+1)/2, weights/2
    for seg_index, (interval, curve) in enumerate(zip(intervals, dense)):
        active = np.flatnonzero(interval.active)
        ei = solver.ei*interval.active
        for cell in range(elements):
            left = seg_index*elements+cell
            h = nodes[left+1]-nodes[left]
            for i in active:
                index = dofs[[left, left+1], i]
                stiffness[np.ix_(index, index)] += solver.gj[i]/h*np.array([[1, -1], [-1, 1]])
                mass[np.ix_(index, index)] += solver.gj[i]/solver.scale**2*h/6*np.array([[2, 1], [1, 2]])
            for xi, weight in zip(points, weights):
                theta = curve((cell+xi)/elements)[:n]
                c, s = np.cos(theta), np.sin(theta)
                intrinsic = interval.curvature
                rotated = np.c_[c*intrinsic[:, 0]-s*intrinsic[:, 1],
                                s*intrinsic[:, 0]+c*intrinsic[:, 1]]
                hb = bending_hessian(ei, rotated)[np.ix_(active, active)]
                shape = np.array([1-xi, xi])
                index = dofs[np.ix_([left, left+1], active)].ravel()
                stiffness[np.ix_(index, index)] += h*weight*np.kron(np.outer(shape, shape), hb)
    free = np.ones(count, dtype=bool)
    for i, beta in enumerate(equilibrium.joints[:n]):
        base = dofs[0, i]
        if beta >= 0:
            free[base] = False  # clamp is at the template; perturbation is zero
        else:
            # Exact static condensation of the straight guide's twist energy.
            stiffness[base, base] += solver.gj[i]/(-beta)
    eigenvalue, eigenvector = eigh(stiffness[np.ix_(free, free)], mass[np.ix_(free, free)],
                                  subset_by_index=[0, 0], check_finite=True)
    vector = np.zeros(count)
    vector[free] = eigenvector[:, 0]
    vector /= max(np.max(abs(vector)), np.finfo(float).tiny)
    mode = np.full(present.shape, np.nan)
    mode[present] = vector[dofs[present]]
    return float(eigenvalue[0]), nodes, mode, count


def elastic_stability(solver, equilibrium, *, mesh_levels=(8, 16, 32),
                      sign_tolerance=1e-3, refinement_tolerance=1e-2):
    """Check the second variation at fixed clamps and fixed tube insertions.

    Generalized mass is integral sum_i GJ_i/L^2 * v_i^2 ds over deployed
    portions. Eigenvalues are dimensionless; guide stiffness is condensed into
    a boundary spring. The sign, not its magnitude, is the physical criterion.
    """
    mesh_levels = tuple(mesh_levels)
    if (len(mesh_levels) < 2 or any(isinstance(v, bool) or int(v) != v or v < 2 for v in mesh_levels)
            or any(b != 2*a for a, b in zip(mesh_levels[:-1], mesh_levels[1:]))):
        raise ValueError("Use at least two nested meshes, doubling elements per segment")
    for value in (sign_tolerance, refinement_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Stability diagnostic tolerances must be positive and finite")
    started = time.perf_counter()
    _, intervals, dense, counters, _ = _trajectory(solver, equilibrium)
    values, dofs = [], []
    for elements in mesh_levels:
        value, nodes, mode, count = _eigenmode(solver, equilibrium, intervals, dense, int(elements))
        values.append(value)
        dofs.append(count)
    difference = abs(values[-1]-values[-2])
    converged = difference <= refinement_tolerance*max(1., abs(values[-1]))
    # Near zero, mesh error can change the sign; keep those cases inconclusive.
    margin = max(sign_tolerance, 2*difference)
    if not converged or abs(values[-1]) <= margin:
        status = "inconclusive"
    elif values[-1] < -margin:
        status = "negative_second_variation"
    else:
        status = "positive_second_variation_on_tested_meshes"
    diagnostics = {**counters, "seconds": time.perf_counter()-started,
                   "mesh_dofs": dofs, "last_refinement_change": difference,
                   "refinement_converged": converged, "classification_margin": margin,
                   "sign_tolerance": sign_tolerance, "refinement_tolerance": refinement_tolerance,
                   "model_fingerprint": solver.model_fingerprint,
                   "equilibrium_base_torsion": equilibrium.base_torsional_strain.tolist(),
                   "method": "reduced_torsional_energy_second_variation_linear_fem",
                   "elastic_stability_certified": False, "global_uniqueness_certified": False,
                   "controller_stability_certified": False, "additional_equilibrium_solves": 0}
    return StabilityDiagnostic(status, tuple(values), mesh_levels, nodes, mode, diagnostics)
