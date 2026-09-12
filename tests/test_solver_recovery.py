"""Full-curvature root recovery, strict residuals and checkpoint semantics."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ctr_reach_envs.mechanics.solver import EquilibriumError, SolverOptions
from ctr_reach_envs.mechanics.simple_rl_env import make_reach_env
from ctr_reach_envs.mechanics.sensitivity import equilibrium_sensitivity


FAILED_Q = np.array([-.21087420446586644, -.14110996395226102, -.010941076481799957,
                     -.874262339815487, -3.5503030133542666, -.15839325809198898])


def solver(options=None):
    options = options or SolverOptions(max_shooting_evaluations=500, shooting_strategy="hybr_restarts")
    return make_reach_env({"system":"ctr_0", "task_profile":"generalized_reach",
                           "solver_options":options.__dict__}).solver


def validate_result(s, result):
    assert np.all(np.isfinite(result.position))
    assert np.all(np.isfinite(result.rotation))
    assert np.isfinite(result.elastic_energy_j) and result.elastic_energy_j >= 0
    assert 0 < result.diagnostics["shooting_evaluations"] <= s.options.max_shooting_evaluations
    assert result.diagnostics["rhs_evaluations"] <= s.options.max_rhs_evaluations
    assert result.diagnostics["root_restart_evaluations"] <= s.options.max_restart_evaluations
    assert result.diagnostics["boundary_residual_scaled"] <= s.options.boundary_tolerance
    assert max(abs(result.distal_torsional_strain))*s.scale <= s.options.boundary_tolerance
    np.testing.assert_allclose(result.rotation.transpose(0,2,1)@result.rotation, np.broadcast_to(np.eye(3),result.rotation.shape), atol=1e-12)


@pytest.mark.parametrize("axis,sign", [(None,0)]+[(i,j) for i in range(6) for j in (-1,1)])
def test_reported_failure_and_one_command_neighbors(axis, sign):
    s = solver()
    q = FAILED_Q.copy()
    if axis is not None:
        q[axis] += sign*([.001]*3+[.05]*3)[axis]
    result = s.solve(q)
    validate_result(s, result)
    np.testing.assert_array_equal(result.joints,q)
    assert result.diagnostics["shooting_algorithm"] == "hybr_restarts_continuation_v2"


def test_recovered_solution_is_reproducible_and_survives_tighter_integration():
    s = solver()
    a, b = s.solve(FAILED_Q), s.solve(FAILED_Q)
    np.testing.assert_array_equal(a.tip,b.tip)
    np.testing.assert_array_equal(a.base_torsional_strain,b.base_torsional_strain)
    reference = solver(replace(s.options, rtol=1e-11, atol=1e-13, max_step=.002))
    # Explicit torsion fixes the equilibrium being checked, not a new root search.
    refined = reference.solve(FAILED_Q, initial_torsion=a.base_torsional_strain)
    validate_result(reference,refined)
    assert np.linalg.norm(a.tip-refined.tip) < 2e-7
    assert not refined.diagnostics["root_restart_accepted"]


def test_recovered_root_still_supports_the_analytical_tip_jacobian():
    s = solver()
    state = s.solve(FAILED_Q)
    derivative = equilibrium_sensitivity(s,state)
    reference=[]
    for i,h in enumerate([1e-7]*3+[1e-5]*3):
        dq=np.eye(6)[i]*h
        plus=s.solve(FAILED_Q+dq,initial_torsion=state.base_torsional_strain)
        minus=s.solve(FAILED_Q-dq,initial_torsion=state.base_torsional_strain)
        reference.append((plus.tip-minus.tip)/(2*h))
    # Finite differences are a test oracle only, not the production Jacobian.
    np.testing.assert_allclose(derivative.tip_jacobian,np.array(reference).T,rtol=2e-4,atol=2e-6)


def test_primary_successes_are_unchanged_and_explicit_guesses_do_not_restart():
    old = solver(SolverOptions(max_shooting_evaluations=500))
    new = solver()
    q = np.r_[[-.15,-.1,-.05], [0.,0.,0.]]
    a,b = old.solve(q),new.solve(q)
    np.testing.assert_array_equal(a.position,b.position)
    assert b.diagnostics["root_restart_attempts"] == 0
    explicit = new.solve(q,initial_torsion=np.zeros(3))
    assert explicit.diagnostics["root_restart_attempts"] == 0


def test_solver_strategy_restores_from_config_and_old_configs_stay_legacy():
    old = make_reach_env({"system":"ctr_0","solver_options":{"max_shooting_evaluations":500}})
    assert old.solver.options.shooting_strategy == "legacy"
    config={"system":"ctr_0","solver_options":solver().options.__dict__}
    new=make_reach_env(config)
    assert new.solver.options.shooting_strategy == "hybr_restarts"
    new.reset(options={"joints":FAILED_Q,"goal":[0.,0.,.2]})
    assert new.costs["root_recovery_solves"] == int(new.equilibrium.diagnostics["root_restart_accepted"])
    assert new.costs["sensitivity_calls"] == 0
    old.close();new.close()


@pytest.mark.parametrize("options,match", [
    (SolverOptions(max_shooting_evaluations=1,shooting_strategy="hybr_restarts"),"Shooting evaluation budget"),
    (SolverOptions(max_rhs_evaluations=1,shooting_strategy="hybr_restarts"),"ODE evaluation budget")])
def test_retries_never_reset_global_evaluation_budgets(options,match):
    with pytest.raises(EquilibriumError,match=match):
        solver(options).solve(FAILED_Q)


def test_scipy_success_flag_cannot_bypass_the_boundary_residual(monkeypatch):
    import ctr_reach_envs.mechanics.solver as module
    # Pretend every root search succeeded while returning an invalid zero twist.
    monkeypatch.setattr(module,"root",lambda *args,**kwargs:SimpleNamespace(x=np.zeros(3),success=True,message="success"))
    with pytest.raises(EquilibriumError):
        solver().solve(FAILED_Q)


@pytest.mark.parametrize("settings", [{"shooting_strategy":"unknown"},
                                      {"max_restart_evaluations":0},{"max_restart_evaluations":True}])
def test_invalid_recovery_options(settings):
    with pytest.raises(ValueError):
        SolverOptions(**settings)
