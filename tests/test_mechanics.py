"""Independent mechanics checks, not a claim of whole-workspace stability."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import minimize

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS, default_env_kwargs
from ctr_reach_envs.mechanics import (TubeParameters, JointConstraints, segment_tubes,
                                    EquilibriumSolver, EquilibriumError, SolverOptions)


def tube(length=.2, curved=.1, curvature=10., inner=.001, outer=.002):
    return TubeParameters(length, curved, inner, outer, 50e9, 23e9, curvature)


def nominal_solver(options=None):
    return EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()], options)


@pytest.mark.parametrize("change", [dict(length=np.nan), dict(length_curved=.3),
                                  dict(diameter_inner=.003), dict(stiffness=0),
                                  dict(x_curvature=np.inf)])
def test_invalid_parameters(change):
    with pytest.raises(ValueError):
        replace(tube(), **change)


def test_complete_constraint_projection_and_optimality():
    c = JointConstraints([.431, .332, .174])
    q = np.array([-.258, -.159, -.001, 0., 0., 0.])
    action = np.array([0., -.001, .001, 0., 0., 0.])
    for _ in range(10):
        q = c.project(q + action)
        assert c.is_feasible(q)
    rng = np.random.default_rng(51)
    for index in range(1000):
        proposal = np.r_[rng.uniform(-.6, .2, 3), rng.normal(size=3)]
        projected = c.project(proposal)
        assert c.is_feasible(projected)
        np.testing.assert_allclose(c.project(projected), projected, atol=1e-12)
        np.testing.assert_array_equal(projected[3:], proposal[3:])
        if index < 12:
            ref = minimize(lambda b: .5*np.sum((b-proposal[:3])**2), np.zeros(3),
                           jac=lambda b: b-proposal[:3], method="SLSQP",
                           constraints={"type": "ineq", "fun": lambda b: c.b-c.A@b,
                                        "jac": lambda b: -c.A},
                           options={"ftol": 1e-13, "maxiter": 200})
            assert ref.success
            np.testing.assert_allclose(projected[:3], ref.x, atol=2e-9)


def test_convex_substep_path_and_rotation_limits():
    c = JointConstraints([.431, .332, .174], constrain_alpha=True)
    start = c.project([-.4, -.2, -.1, 4., -8., .2])
    end = c.project([-.1, -.05, -.001, -5., 9., .2])
    for t in np.linspace(0, 1, 101):
        assert c.is_feasible((1-t)*start+t*end)
    assert np.max(abs(end[3:])) <= np.pi


def test_segment_events_preserve_small_intervals_and_exact_tips():
    tubes = [tube(.2, .1), tube(.15, .05, inner=.003, outer=.004),
             tube(.1, .03, inner=.005, outer=.006)]
    beta = np.array([-.02, -.020000005, -.01])
    intervals = segment_tubes(tubes, beta)
    events = np.r_[intervals[0].start, [x.end for x in intervals]]
    assert np.min(np.diff(events)) < 1e-8
    assert np.all(np.diff(events) > 0)
    for end in np.array([t.length for t in tubes])+beta:
        assert np.min(abs(events-end)) < 1e-15
    assert events[-1] == tubes[0].length+beta[0]


def test_straight_tube_closed_form():
    result = EquilibriumSolver([tube(curved=0, curvature=0)]).solve([-.03, 1.2])
    np.testing.assert_allclose(result.tip, [0, 0, .17], atol=1e-12)
    assert result.elastic_energy_j == 0
    np.testing.assert_allclose(result.base_torsional_strain, 0, atol=1e-12)


@pytest.mark.parametrize("beta", [0., -.03, -.15])
def test_single_curved_tube_closed_form_and_guide_energy(beta):
    rod = tube()
    result = EquilibriumSolver([rod]).solve([beta, 0.])
    exposed = rod.length + beta
    curved = min(exposed, rod.length_curved)
    straight = exposed-curved
    k = rod.x_curvature
    expected = [0., (np.cos(k*curved)-1)/k, straight+np.sin(k*curved)/k]
    np.testing.assert_allclose(result.tip, expected, atol=2e-10)
    guide_curved = max(0, rod.length_curved-exposed)
    assert result.elastic_energy_j == pytest.approx(.5*rod.EI*k*k*guide_curved, abs=1e-12)


def test_aligned_tubes_constant_curvature_zero_strain_energy():
    tubes = [tube(.2, .2), tube(.15, .15, inner=.003, outer=.004),
             tube(.1, .1, inner=.005, outer=.006)]
    result = EquilibriumSolver(tubes).solve(np.zeros(6))
    np.testing.assert_allclose(result.tip, [0., (np.cos(2)-1)/10, np.sin(2)/10], atol=2e-10)
    assert result.elastic_energy_j < 1e-20
    np.testing.assert_allclose(result.distal_torsional_strain, 0, atol=1e-10)


def test_twisting_tubes_against_independent_collocation():
    # The two-tube torsion BVP reduces analytically to delta'' = c sin(delta).
    # The Robin condition at zero accounts for torque transmission in the guide.
    tubes = [tube(.12, .12, 5., .0005, .0007), tube(.12, .12, 7., .001, .0012)]
    solver = EquilibriumSolver(tubes)
    q = np.array([-.02, -.02, .2, 1.])
    result = solver.solve(q)
    ej = np.array([t.EI for t in tubes]); gj = np.array([t.GJ for t in tubes])
    coefficient = ej.prod()/ej.sum()*5*7*np.sum(1/gj)
    mesh = np.linspace(0, .1, 101)
    ref = solve_bvp(lambda s, y: np.vstack((y[1], coefficient*np.sin(y[0]))),
                    lambda a, b: np.array([a[0]-.02*a[1]-.8, b[1]]),
                    mesh, np.vstack((np.full(mesh.size, .8), np.zeros(mesh.size))),
                    tol=1e-10, max_nodes=5000)
    assert ref.success
    np.testing.assert_allclose(result.angles[:, 1]-result.angles[:, 0], ref.sol(result.s)[0], atol=2e-9)
    np.testing.assert_allclose(result.torsional_strain[:, 1]-result.torsional_strain[:, 0], ref.sol(result.s)[1], atol=2e-8)
    assert np.linalg.norm(result.base_torsional_strain) > .01
    # Independent matrix-frame integration, using the collocation torsion solution.
    weighted_angle = gj @ q[2:] / gj.sum()
    def shape_rhs(s, y):
        delta = ref.sol(s)[0]
        theta = weighted_angle + np.array([-gj[1], gj[0]])/gj.sum()*delta
        ux = np.sum(ej*np.array([5, 7])*np.cos(theta))/ej.sum()
        uy = np.sum(ej*np.array([5, 7])*np.sin(theta))/ej.sum()
        hat = np.array([[0., 0., uy], [0., 0., -ux], [-uy, ux, 0.]])
        rotation = y[3:].reshape(3, 3)
        return np.r_[rotation[:, 2], (rotation@hat).ravel()]
    shape = solve_ivp(shape_rhs, (0, .1), np.r_[np.zeros(3), np.eye(3).ravel()],
                      method="DOP853", rtol=1e-11, atol=1e-13)
    assert shape.success
    np.testing.assert_allclose(result.tip, shape.y[:3, -1], atol=2e-9)


def test_nonlinear_refinement_torque_balance_rotation_and_covariance():
    q = np.array([-.1, -.05, -.02, .4, -.7, 1.2])
    standard = nominal_solver().solve(q)
    refined = nominal_solver(SolverOptions(rtol=1e-11, atol=1e-13, max_step=.002)).solve(q)
    assert np.linalg.norm(standard.tip-refined.tip) < 20e-6
    assert standard.diagnostics["boundary_residual_scaled"] < 1e-7
    assert abs(standard.diagnostics["base_torque_sum_nm"]) < 1e-8
    np.testing.assert_allclose(standard.rotation.transpose(0, 2, 1)@standard.rotation,
                               np.broadcast_to(np.eye(3), standard.rotation.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(standard.rotation), 1., atol=1e-12)
    angle = .63
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    q[3:] += angle
    shifted = nominal_solver().solve(q)
    np.testing.assert_allclose(shifted.tip, rotation@standard.tip, atol=2e-8)
    assert shifted.elastic_energy_j == pytest.approx(standard.elastic_energy_j, rel=1e-7)


def test_derivatives_do_not_have_the_old_segment_rounding_jump():
    solver = nominal_solver()
    q = np.array([-.1, -.05, -.02, .4, -.7, 1.2])
    slopes = []
    for h in [1e-4, 1e-5, 1e-6]:
        delta = np.zeros(6); delta[0] = h
        slopes.append((solver.solve(q+delta).tip-solver.solve(q-delta).tip)/(2*h))
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(slopes[1], slopes[2], rtol=1e-3, atol=1e-5)


def test_budget_and_bad_boundary_result_fail_explicitly(monkeypatch):
    q = np.array([-.1, -.05, -.02, .4, -.7, 1.2])
    with pytest.raises(EquilibriumError, match="budget"):
        nominal_solver(SolverOptions(max_rhs_evaluations=1)).solve(q)
    from types import SimpleNamespace
    monkeypatch.setattr("ctr_reach_envs.mechanics.solver.root",
                        lambda fun, guess, **kw: SimpleNamespace(x=guess, message="forced unsuccessful root"))
    with pytest.raises(EquilibriumError, match="residual|budget"):
        nominal_solver().solve(q)


def test_no_implicit_projection_or_stability_claim():
    with pytest.raises(ValueError, match="constraints"):
        nominal_solver().solve([-.258, -.158, 0, 0, 0, 0])
    result = nominal_solver().solve(np.zeros(6))
    assert result.diagnostics["elastic_stability"] == "not_certified"
    assert result.diagnostics["uniqueness"] == "not_certified"


def test_gymnasium_bridge_pure_solve_and_baseline_registration():
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from ctr_reach_envs.mechanics.env import register_equilibrium_env
    from ctr_reach_envs.envs.model import Model
    baseline_entry = gym.spec("CTR-Reach-v1").entry_point
    env = gym.make(register_equilibrium_env()).unwrapped
    try:
        assert gym.spec("CTR-Reach-v1").entry_point == baseline_entry
        assert not isinstance(env.model, Model)
        obs, info = env.reset(seed=22)
        assert env.observation_space.contains(obs)
        joints = env.trig_obj.joints.copy()
        backbone = env.model.r.copy()
        pose = env.model.last_result
        probe = env.trig_obj.constraints[env.system].project(joints + np.array([-.001, 0, 0, .01, 0, 0]))
        env.model.solve(probe, env.system)
        np.testing.assert_array_equal(env.model.r, backbone)
        assert env.model.last_result is pose
        check_env(env, skip_render_check=True)
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert not info["solver_failure"]
        assert info["elastic_stability_certified"] is False
        assert env.trig_obj.constraints[env.system].is_feasible(env.trig_obj.joints)
    finally:
        env.close()
