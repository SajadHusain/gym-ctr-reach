"""Independent checks of Burgner (2014) Eq. (7), frames and physical units."""
import numpy as np
import pytest

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from ctr_reach_envs.mechanics import TubeParameters, EquilibriumSolver, SolverOptions, equilibrium_sensitivity


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def material_rotation(eq):
    i = eq.tube_tip_indices[0]
    c, s = np.cos(eq.angles[i, 0]), np.sin(eq.angles[i, 0])
    return eq.rotation[i] @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@pytest.mark.parametrize("system", list(CTR_SYSTEMS_PARAMETERS))
def test_paper_jacobian_matches_perturbed_positions_and_material_rotations(system):
    tubes = [TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS[system].values()]
    assert all(t.y_curvature == 0 for t in tubes)
    # Tighten the boundary solve too: otherwise a perturbed input can accept
    # the unperturbed shooting guess and corrupt the finite-difference check.
    solver = EquilibriumSolver(tubes, SolverOptions(rtol=1e-11, atol=1e-13,
                                                   max_step=.002, boundary_tolerance=1e-11))
    # Interior extensions: all tips and material boundaries remain distinct.
    q = np.r_[-.11*solver.lengths, [.13, -.08, .19]]
    eq = solver.solve(q)
    derivative = equilibrium_sensitivity(solver, eq)
    paper = derivative.paper
    n = solver.n
    assert paper.E_q.shape == (6, 2*n) and paper.V_q.shape == (2*n-1, 2*n)
    np.testing.assert_allclose(paper.material_tip_rotation, material_rotation(eq), atol=1e-10)
    np.testing.assert_allclose(paper.B_u@derivative.base_torsion_jacobian+paper.B_q, 0., atol=1e-10)
    implicit = paper.E_q-paper.E_u@np.linalg.solve(paper.B_u, paper.B_q)
    np.testing.assert_allclose(paper.spatial_jacobian, implicit, atol=1e-12)
    np.testing.assert_allclose(derivative.tip_jacobian,
                               implicit[:3]-skew(eq.tip)@implicit[3:], atol=1e-9)
    np.testing.assert_array_equal(paper.spatial_jacobian_alpha_beta,
                                 paper.spatial_jacobian[:, [3, 4, 5, 0, 1, 2]])
    steps = np.r_[np.full(n, 2e-6), np.full(n, 2e-5)]
    dp, dw = [], []
    for j, h in enumerate(steps):
        delta = np.eye(2*n)[j]*h
        # Perturb actual physical inputs, without projecting them. Supplying
        # the converged root isolates a local derivative for this validation.
        plus = solver.solve(q+delta, initial_torsion=eq.base_torsional_strain)
        minus = solver.solve(q-delta, initial_torsion=eq.base_torsional_strain)
        dp.append((plus.tip-minus.tip)/(2*h))
        w_hat = ((material_rotation(plus)-material_rotation(minus))/(2*h))@material_rotation(eq).T
        dw.append([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    np.testing.assert_allclose(derivative.tip_jacobian, np.asarray(dp).T, atol=3e-6, rtol=3e-4)
    np.testing.assert_allclose(paper.spatial_jacobian[3:], np.asarray(dw).T, atol=2e-5, rtol=3e-4)
    np.testing.assert_allclose(paper.body_jacobian,
        np.vstack((material_rotation(eq).T@derivative.tip_jacobian,
                   material_rotation(eq).T@paper.spatial_jacobian[3:])), atol=1e-9)


@pytest.mark.parametrize("curvature", [0., 4.])
def test_single_tube_closed_form_and_no_extra_equilibrium_solves(curvature, monkeypatch):
    solver = EquilibriumSolver([TubeParameters(.2, .12, .001, .002, 50e9, 23e9, curvature)])
    q = np.array([-.03, .4])
    eq = solver.solve(q)
    monkeypatch.setattr(solver, "solve", lambda *a, **k: pytest.fail("Jacobian calculation called FK"))
    d = equilibrium_sensitivity(solver, eq)
    np.testing.assert_allclose(d.tip_jacobian[:, 0], [0., 0., 1.], atol=1e-9)
    np.testing.assert_allclose(d.tip_jacobian[:, 1], np.cross([0., 0., 1.], eq.tip), atol=1e-9)
    np.testing.assert_allclose(d.paper.spatial_jacobian[3:, 1], [0., 0., 1.], atol=1e-9)
    np.testing.assert_allclose(d.paper.spatial_jacobian[3:, 0], 0., atol=1e-9)
    assert d.diagnostics["additional_equilibrium_solves"] == 0
