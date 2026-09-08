"""Independent analytical stability and complete-BVP derivative checks."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from ctr_reach_envs.mechanics import (TubeParameters, EquilibriumSolver, SolverOptions,
                                    SensitivityError, equilibrium_sensitivity, elastic_stability)
from ctr_reach_envs.mechanics.sensitivity import _field
from ctr_reach_envs.mechanics.geometry import segment_tubes
from ctr_reach_envs.mechanics.stability import bending_hessian


def nominal_solver():
    return EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()])


def two_tube_case(threshold_ratio, guide=0., aligned=False):
    """Set C*L^2 explicitly for delta'' = C sin(delta), with equal tube lengths."""
    length = .1
    tubes = [TubeParameters(length+guide, length+guide, .0005, .0007, 50e9, 23e9, 1.),
             TubeParameters(length+guide, length+guide, .001, .0012, 50e9, 23e9, 1.)]
    ei = np.array([t.EI for t in tubes]); gj = np.array([t.GJ for t in tubes])
    factor = ei.prod()/ei.sum()*np.sum(1/gj)
    # Robin v(0)-guide*v'(0)=0; Neumann v'(L)=0 gives cos(kL)-guide*k*sin(kL)=0.
    k0 = np.pi/(2*length) if guide == 0 else brentq(lambda k: np.cos(k*length)-guide*k*np.sin(k*length), 0, np.pi/(2*length))
    curvature = k0*np.sqrt(threshold_ratio/factor)
    solver = EquilibriumSolver([replace(t, x_curvature=curvature) for t in tubes])
    equilibrium = solver.solve([-guide, -guide, 0., 0. if aligned else np.pi])
    return solver, equilibrium, k0


@pytest.mark.parametrize("ratio,status", [(.5, "positive_second_variation_on_tested_meshes"),
                                         (1., "inconclusive"),
                                         (1.5, "negative_second_variation")])
def test_two_tube_analytical_buckling_threshold(ratio, status):
    solver, equilibrium, k0 = two_tube_case(ratio)
    result = elastic_stability(solver, equilibrium)
    expected = (solver.scale*k0)**2*(1-ratio)
    assert result.status == status
    assert abs(result.minimum_eigenvalues[-1]-expected) < 6e-4
    # Rayleigh-Ritz eigenvalues approach the exact eigenvalue from above.
    assert np.all(np.diff(result.minimum_eigenvalues) < 0)
    assert result.minimum_eigenvalues[-1] > expected-1e-9
    assert not result.diagnostics["elastic_stability_certified"]


@pytest.mark.parametrize("ratio,status", [(.5, "positive_second_variation_on_tested_meshes"),
                                         (1., "inconclusive"),
                                         (1.5, "negative_second_variation")])
def test_guide_torsion_shifts_analytical_threshold(ratio, status):
    solver, equilibrium, _ = two_tube_case(ratio, guide=.03)
    assert elastic_stability(solver, equilibrium).status == status


def test_aligned_curvature_and_straight_tube_positive_modes():
    solver, equilibrium, _ = two_tube_case(3., aligned=True)
    result = elastic_stability(solver, equilibrium)
    assert result.status == "positive_second_variation_on_tested_meshes"
    assert result.minimum_eigenvalues[-1] == pytest.approx(np.pi**2/4, abs=6e-4)
    single = EquilibriumSolver([replace(solver.tubes[0], x_curvature=0.)])
    assert elastic_stability(single, single.solve([0., .7])).minimum_eigenvalues[-1] == pytest.approx(np.pi**2/4, abs=6e-4)


def test_negative_mode_reduces_actual_reduced_energy():
    solver, equilibrium, _ = two_tube_case(1.5)
    result = elastic_stability(solver, equilibrium)
    s, mode = result.mode_s, result.mode_angles
    # Independently evaluate energy change, including nonlinear cos bending.
    def energy(amplitude):
        theta = np.array([0., np.pi])+amplitude*mode
        derivative = np.diff(theta, axis=0)/np.diff(s)[:, None]
        mid = (theta[:-1]+theta[1:])/2
        k = np.array([t.x_curvature for t in solver.tubes])
        curvature_sum = np.sum(solver.ei*k*np.exp(1j*mid), axis=1)
        bend = .5*(np.sum(solver.ei*k*k)-abs(curvature_sum)**2/solver.ei.sum())
        twist = .5*np.sum(solver.gj*derivative**2, axis=1)
        return np.sum(np.diff(s)*(bend+twist))
    assert energy(.001) < energy(0.)
    assert energy(-.001) < energy(0.)


def test_same_clamps_can_have_distinct_stable_and_unstable_equilibria():
    solver, unstable, _ = two_tube_case(1.5)
    eta = np.array([-solver.gj[1], solver.gj[0]])/solver.gj.sum()*30.
    stable = solver.solve(unstable.joints, initial_torsion=eta)
    assert np.linalg.norm(stable.base_torsional_strain-unstable.base_torsional_strain) > 1.
    assert stable.elastic_energy_j < unstable.elastic_energy_j
    assert elastic_stability(solver, unstable).status == "negative_second_variation"
    assert elastic_stability(solver, stable).status == "positive_second_variation_on_tested_meshes"
    np.testing.assert_array_equal(unstable.base_torsional_strain, np.zeros(2))


def test_bending_hessian_against_energy_second_difference():
    ei = np.array([.03, .09, .05])
    intrinsic = np.array([[5., 2.], [7., -3.], [-1., 4.]])
    theta = np.array([.4, -.7, 1.2])
    direction = np.array([.7, -.2, .3])
    def rotated(angles):
        c, s = np.cos(angles), np.sin(angles)
        return np.c_[c*intrinsic[:, 0]-s*intrinsic[:, 1], s*intrinsic[:, 0]+c*intrinsic[:, 1]]
    def energy(angles):
        vector = (ei[:, None]*rotated(angles)).sum(axis=0)
        return .5*np.sum(ei*np.sum(intrinsic**2, axis=1))-.5*(vector@vector)/ei.sum()
    h = 1e-4
    fd = (energy(theta+h*direction)-2*energy(theta)+energy(theta-h*direction))/h**2
    hessian = bending_hessian(ei, rotated(theta))
    assert direction@hessian@direction == pytest.approx(fd, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(hessian@np.ones(3), 0, atol=1e-14)


def test_analytic_ode_state_derivatives_with_general_curvature_and_nonunit_quaternion():
    solver = nominal_solver()
    solver = EquilibriumSolver([replace(t, y_curvature=i+2.) for i, t in enumerate(solver.tubes)])
    interval = segment_tubes(solver.tubes, [-.1, -.05, -.02])[2]
    y = np.r_[[.3, -.7, 1.2], [.2, -.4, .8], [.1, .2, -.3], [.9, .2, -.1, .3]]
    _, analytic = _field(solver, interval, y, True)
    fd = np.column_stack([(_field(solver, interval, y+v*1e-6)-_field(solver, interval, y-v*1e-6))/(2e-6)
                          for v in np.eye(len(y))])
    np.testing.assert_allclose(analytic, fd, rtol=2e-6, atol=2e-7)


def test_implicit_sensitivity_against_complete_bvp_and_local_prediction():
    solver = nominal_solver()
    q = np.array([-.1, -.05, -.02, .4, -.7, 1.2])
    equilibrium = solver.solve(q)
    result = equilibrium_sensitivity(solver, equilibrium)
    for i, h in enumerate([1e-6]*3+[1e-5]*3):
        dq = np.eye(6)[i]*h
        a = solver.solve(q+dq, initial_torsion=equilibrium.base_torsional_strain)
        b = solver.solve(q-dq, initial_torsion=equilibrium.base_torsional_strain)
        np.testing.assert_allclose(result.tip_jacobian[:, i], (a.tip-b.tip)/(2*h), rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(result.base_torsion_jacobian[:, i],
                                   (a.base_torsional_strain-b.base_torsional_strain)/(2*h), rtol=2e-5, atol=1e-5)
    direction = np.array([-.0004, .0002, .0001, .004, -.003, .002])
    errors = []
    for scale in [1., .5, .25]:
        dq = direction*scale
        moved = solver.solve(q+dq, initial_torsion=equilibrium.base_torsional_strain)
        errors.append(np.linalg.norm(moved.tip-equilibrium.tip-result.tip_jacobian@dq))
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5
    # Common clamp rotation is a rigid rotation of the entire shape about z.
    np.testing.assert_allclose(result.tip_jacobian[:, 3:].sum(axis=1),
                               np.cross([0, 0, 1], equilibrium.tip), atol=2e-8)
    assert result.diagnostics["implicit_equation_residual"] < 1e-10


def test_single_tube_jacobian_closed_form_and_pure_analysis():
    tube = TubeParameters(.2, .1, .001, .002, 50e9, 23e9, 10.)
    solver = EquilibriumSolver([tube])
    equilibrium = solver.solve([-.03, .4])
    old = equilibrium.position.copy()
    result = equilibrium_sensitivity(solver, equilibrium)
    np.testing.assert_allclose(result.tip_jacobian[:, 0], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(result.tip_jacobian[:, 1], np.cross([0, 0, 1], equilibrium.tip), atol=1e-9)
    elastic_stability(solver, equilibrium)
    np.testing.assert_array_equal(equilibrium.position, old)


def test_model_mismatch_and_event_collision_are_rejected():
    solver, equilibrium, _ = two_tube_case(.5)
    with pytest.raises(SensitivityError, match="moving segment events"):
        equilibrium_sensitivity(solver, equilibrium)
    other = EquilibriumSolver([replace(t, stiffness=t.stiffness*1.01) for t in solver.tubes])
    with pytest.raises(SensitivityError, match="different model"):
        elastic_stability(other, equilibrium)
    with pytest.raises(SensitivityError, match="different model"):
        equilibrium_sensitivity(other, equilibrium)


def test_condition_guard_and_analysis_budget_fail_explicitly():
    solver = nominal_solver()
    equilibrium = solver.solve([-.1, -.05, -.02, .4, -.7, 1.2])
    with pytest.raises(SensitivityError, match="numerically singular"):
        equilibrium_sensitivity(solver, equilibrium, condition_limit=2.)
    solver.options = SolverOptions(max_rhs_evaluations=1)
    with pytest.raises(SensitivityError, match="budget"):
        equilibrium_sensitivity(solver, equilibrium)
    with pytest.raises(SensitivityError, match="budget"):
        elastic_stability(solver, equilibrium)
