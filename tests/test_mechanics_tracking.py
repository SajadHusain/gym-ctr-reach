"""Mechanical continuation, finite-action acceptance, and rollback tests."""
from dataclasses import replace

import numpy as np
import pytest

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from ctr_reach_envs.mechanics import (TubeParameters, EquilibriumSolver, EquilibriumError,
                                    BranchTracker, BranchInitializationError, TrackingOptions, SolverOptions, elastic_stability)


def single_tracker(beta=-.03, options=None):
    rod = TubeParameters(.2, .1, .001, .002, 50e9, 23e9, 10.)
    solver = EquilibriumSolver([rod])
    return BranchTracker(solver, solver.solve([beta, .4]), options)


def nominal_tracker():
    solver = EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()])
    return BranchTracker(solver, solver.solve([-.1, -.05, -.02, .4, -.7, 1.2]))


def assert_same_state(a, b):
    assert a.accepted_steps == b.accepted_steps
    for field in ("joints", "position", "angles", "base_torsional_strain"):
        np.testing.assert_array_equal(getattr(a.equilibrium, field), getattr(b.equilibrium, field))
    np.testing.assert_array_equal(a.sensitivity.tip_jacobian, b.sensitivity.tip_jacobian)
    assert a.stability.minimum_eigenvalues == b.stability.minimum_eigenvalues


def test_success_checks_midpoint_endpoint_and_reverse():
    tracker = nominal_tracker()
    before = tracker.state
    delta = np.array([-.0002, .0001, 0., .02, -.01, .01])
    result = tracker.step(delta)
    assert result.status == "accepted"
    assert result.accepted_fraction == 1.
    np.testing.assert_allclose(result.state.equilibrium.joints, before.equilibrium.joints+delta, atol=1e-15)
    np.testing.assert_allclose(result.applied_delta, delta, atol=1e-15)
    assert result.state.accepted_steps == 1
    assert result.diagnostics["equilibrium_calls"] == 3
    assert result.diagnostics["stability_calls"] == 2
    assert result.diagnostics["sensitivity_calls"] == 2
    attempt = result.diagnostics["attempts"][-1]
    assert [x["fraction_of_trial"] for x in attempt["checkpoints"]] == [.5, 1.]
    for check in attempt["checkpoints"]:
        assert check["minimum_eigenvalue"] > tracker.options.minimum_elastic_eigenvalue
        for field in ("local_prediction", "overall_prediction"):
            assert check[field]["tip_error_m"] <= check[field]["tip_limit_m"]
            assert check[field]["twist_prediction_error"] <= check[field]["twist_prediction_limit"]
    assert attempt["reverse_tip_error_m"] <= tracker.options.reverse_tip_tolerance_m
    assert not result.diagnostics["continuous_path_certified"]
    assert not result.diagnostics["controller_stability_certified"]


def test_tip_prediction_backtracking_reduces_applied_motion():
    tracker = single_tracker(options=TrackingOptions(tip_absolute_tolerance_m=1e-7, tip_relative_tolerance=1e-6))
    result = tracker.step([0., .05])
    assert result.status == "accepted"
    assert 0 < result.accepted_fraction < 1
    assert result.diagnostics["attempts"][0]["reason"] == "tip_prediction_error"
    np.testing.assert_allclose(result.applied_delta, result.projected_delta*result.accepted_fraction)
    assert result.state.accepted_steps == 1


def test_accepted_step_matches_refined_quarter_point_continuation():
    tracker = nominal_tracker()
    before = tracker.state
    result = tracker.step([-.0002, .0001, 0., .02, -.01, .01])
    assert result.status == "accepted"
    reference = EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()],
                                  SolverOptions(rtol=1e-11, atol=1e-13, max_step=.002))
    previous = before.equilibrium
    for fraction in [.25, .5, .75, 1.]:
        q = before.equilibrium.joints+fraction*result.applied_delta
        previous = reference.solve(q, initial_torsion=previous.base_torsional_strain)
        assert elastic_stability(reference, previous).status == "positive_second_variation_on_tested_meshes"
    np.testing.assert_allclose(previous.tip, result.state.equilibrium.tip, atol=1e-8)
    np.testing.assert_allclose(previous.base_torsional_strain, result.state.equilibrium.base_torsional_strain, atol=1e-6)


def test_twist_branch_mismatch_is_rejected_even_when_tip_matches(monkeypatch):
    tracker = single_tracker(options=TrackingOptions(max_backtracks=0))
    before = tracker.state
    original = tracker._solver.solve
    def wrong_twist(*args, **kwargs):
        value = original(*args, **kwargs)
        return replace(value, base_torsional_strain=value.base_torsional_strain+1.)
    monkeypatch.setattr(tracker._solver, "solve", wrong_twist)
    result = tracker.step([.0001, .001])
    assert result.status == "held"
    assert result.diagnostics["reason"] == "branch_predictor_error"
    assert_same_state(before, tracker.state)


def test_rejected_endpoint_does_not_commit_the_valid_midpoint():
    tracker = single_tracker(options=TrackingOptions(max_backtracks=0, tip_absolute_tolerance_m=2e-5,
                                                    tip_relative_tolerance=1e-8))
    before = tracker.state
    result = tracker.step([0., .05])
    assert result.status == "held"
    assert result.diagnostics["reason"] == "tip_prediction_error"
    assert len(result.diagnostics["attempts"][0]["checkpoints"]) == 2
    assert "elastic_status" in result.diagnostics["attempts"][0]["checkpoints"][0]
    assert_same_state(before, tracker.state)
    assert_same_state(before, result.state)
    np.testing.assert_array_equal(result.applied_delta, np.zeros(2))


def test_reverse_inconsistency_is_rejected(monkeypatch):
    tracker = single_tracker(options=TrackingOptions(max_backtracks=0))
    before = tracker.state
    original = tracker._solver.solve
    calls = []
    def corrupted_reverse(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 3:
            value = replace(value, position=value.position+np.array([.01, 0., 0.]))
        return value
    monkeypatch.setattr(tracker._solver, "solve", corrupted_reverse)
    result = tracker.step([.0001, .001])
    assert result.status == "held"
    assert result.diagnostics["reason"] == "reverse_branch_mismatch"
    assert result.diagnostics["equilibrium_calls"] == 3
    assert_same_state(before, tracker.state)


def test_failed_solver_calls_are_counted_and_state_is_preserved(monkeypatch):
    tracker = single_tracker(options=TrackingOptions(max_backtracks=2))
    before = tracker.state
    def failure(*args, **kwargs):
        raise EquilibriumError("forced solver budget failure")
    monkeypatch.setattr(tracker._solver, "solve", failure)
    result = tracker.step([.0001, .001])
    assert result.status == "held"
    assert result.diagnostics["equilibrium_calls"] == 3
    assert result.diagnostics["failed_calls"] == 3
    assert result.diagnostics["failed_calls_without_rhs_counts"] == 3
    assert result.diagnostics["rhs_evaluations_known"] == 0
    assert_same_state(before, tracker.state)


def test_inconclusive_trial_is_rejected(monkeypatch):
    tracker = single_tracker(options=TrackingOptions(max_backtracks=0))
    before = tracker.state
    import ctr_reach_envs.mechanics.tracking as module
    original = module.elastic_stability
    monkeypatch.setattr(module, "elastic_stability", lambda *a, **k: replace(original(*a, **k), status="inconclusive"))
    result = tracker.step([.0001, .001])
    assert result.status == "held"
    assert result.diagnostics["reason"] == "elastic_inconclusive"
    assert_same_state(before, tracker.state)


def test_event_crossing_is_backtracked_before_solve():
    tracker = single_tracker(beta=-.09975)
    result = tracker.step([-.001, 0.])
    assert result.status == "accepted"
    assert result.accepted_fraction < .25
    assert result.state.equilibrium.joints[0] > -.1
    assert result.diagnostics["attempts"][0]["reason"] == "event_topology_change"
    assert result.diagnostics["equilibrium_calls"] == 3


def test_projection_caps_and_no_motion_accounting():
    tracker = single_tracker()
    before = tracker.state.equilibrium.joints.copy()
    result = tracker.step([.1, 0.])
    assert result.status == "accepted"
    assert result.diagnostics["command_projected"]
    assert result.applied_delta[0] == pytest.approx(.001)
    np.testing.assert_allclose(result.state.equilibrium.joints-before, result.applied_delta, atol=1e-15)
    capped = single_tracker(beta=0.)
    unchanged = capped.step([.001, 0.])
    assert unchanged.status == "no_motion"
    assert unchanged.diagnostics["equilibrium_calls"] == 0
    assert unchanged.diagnostics["attempts"] == []


def test_initial_negative_root_and_changed_model_are_rejected():
    solver = EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()])
    q = [-.19599808468096527, -.12600990538805137, -.003737862197937286,
         2.2994960463889864, -.8594645906266152, -1.4681794936637027]
    root = solver.solve(q)
    with pytest.raises(BranchInitializationError, match="negative_second_variation") as error:
        BranchTracker(solver, root)
    assert error.value.diagnostics["rejection_details"]["minimum_eigenvalues"][-1] < 0
    other = EquilibriumSolver([replace(t, stiffness=t.stiffness*1.01) for t in solver.tubes])
    with pytest.raises(BranchInitializationError, match="different model"):
        BranchTracker(other, root)


def test_returned_snapshots_cannot_mutate_tracker_or_caller_solver():
    rod = TubeParameters(.2, .1, .001, .002, 50e9, 23e9, 10.)
    solver = EquilibriumSolver([rod])
    root = solver.solve([-.03, .4])
    tracker = BranchTracker(solver, root)
    before = tracker.state
    root.position[:] = 100
    solver.ei[:] = 0
    view = tracker.state
    view.equilibrium.position[:] = 100
    view.sensitivity.tip_jacobian[:] = 100
    assert_same_state(before, tracker.state)
    result = tracker.step([.0001, .001])
    current = tracker.state
    result.state.equilibrium.joints[:] = 100
    assert_same_state(current, tracker.state)


@pytest.mark.parametrize("change", [dict(max_backtracks=-1), dict(max_backtracks=True),
                                  dict(max_translation_step_m=0), dict(shooting_condition_limit=1)])
def test_invalid_options(change):
    with pytest.raises(ValueError):
        TrackingOptions(**change)


def test_invalid_action_does_not_mutate_state():
    tracker = single_tracker()
    before = tracker.state
    with pytest.raises(ValueError):
        tracker.step([np.nan, 0.])
    assert_same_state(before, tracker.state)
