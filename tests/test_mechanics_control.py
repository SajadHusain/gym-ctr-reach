"""Measured decrease, transactionality, action semantics and model control."""
from dataclasses import replace

import numpy as np
import pytest

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from ctr_reach_envs.mechanics import (TubeParameters, EquilibriumSolver, BranchTracker, TrackingOptions,
                                    GoalProgress, GoalController, ControlOptions)


def straight_tracker(options=None):
    solver = EquilibriumSolver([TubeParameters(.2, 0., .001, .002, 50e9, 23e9, 0.)])
    return BranchTracker(solver, solver.solve([-.03, .4]), options)


def test_projected_nondescent_rejects_without_any_probe():
    tracker = straight_tracker()
    before = tracker.state
    goal = before.equilibrium.tip+[0, 0, .005]
    result = tracker.step([-.001, 0.], goal_progress=GoalProgress(goal))
    assert result.status == "held"
    assert result.diagnostics["equilibrium_calls"] == 0
    assert len(result.diagnostics["attempts"]) == 1
    assert result.diagnostics["reason"] == "goal_not_descent_direction"
    np.testing.assert_array_equal(tracker.state.equilibrium.position, before.equilibrium.position)
    np.testing.assert_array_equal(tracker.state.sensitivity.tip_jacobian, before.sensitivity.tip_jacobian)


def test_nonlinear_merit_backtracks_and_checks_each_sample():
    tracker = straight_tracker()
    before = tracker.state
    goal = before.equilibrium.tip+[0., 0., .0001]
    result = tracker.step([.001, 0.], goal_progress=GoalProgress(goal))
    assert result.status == "accepted"
    assert result.accepted_fraction < 1
    assert result.diagnostics["attempts"][0]["reason"] == "goal_decrease_failed"
    last_v = .5*np.linalg.norm(before.equilibrium.tip-goal)**2
    for point in result.diagnostics["attempts"][-1]["checkpoints"]:
        values = point["local_goal_progress"]
        assert values["v_after_m2"] < values["v_before_m2"] <= last_v+1e-20
        assert values["v_after_m2"] <= values["armijo_upper_bound_m2"]
        assert values["directional_derivative_m2"] < 0
        last_v = values["v_after_m2"]
    assert result.state.accepted_steps == 1
    # Replaying the applied physical action through the goal-independent plant
    # reproduces the transition. Replaying the original proposal would not.
    replay = straight_tracker().step(result.applied_delta)
    np.testing.assert_allclose(replay.state.equilibrium.tip, result.state.equilibrium.tip, atol=1e-12)


def test_goal_rejection_rolls_back_when_backtracking_budget_is_zero():
    tracker = straight_tracker(TrackingOptions(max_backtracks=0))
    before = tracker.state
    result = tracker.step([.001, 0.], goal_progress=GoalProgress(before.equilibrium.tip+[0., 0., .0001]))
    assert result.status == "held"
    np.testing.assert_array_equal(result.applied_delta, np.zeros(2))
    np.testing.assert_array_equal(result.state.equilibrium.position, before.equilibrium.position)
    assert result.state.accepted_steps == before.accepted_steps


def test_goal_controller_reaches_and_holds_without_extra_queries():
    controller = GoalController(straight_tracker())
    goal = controller.tracker.state.equilibrium.tip+[0., 0., .005]
    errors = []
    for _ in range(12):
        result = controller.step(goal)
        errors.append(result.diagnostics["error_after_m"])
        if result.status == "goal_reached":
            break
    assert result.status == "goal_reached"
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] <= .001
    still = controller.step(goal)
    assert still.status == "goal_reached"
    assert still.transition is None
    assert still.diagnostics["equilibrium_calls"] == 0
    assert still.diagnostics["sensitivity_calls"] == 0
    np.testing.assert_array_equal(still.applied_delta, np.zeros(2))


def test_bad_policy_uses_fallback_and_commits_once():
    controller = GoalController(straight_tracker())
    goal = controller.tracker.state.equilibrium.tip+[0., 0., .005]
    result = controller.step(goal, policy_delta=[-.001, 0.])
    assert result.action_source == "jacobian_dls"
    assert result.diagnostics["fallback_used"]
    assert result.diagnostics["candidate_attempts"][0]["status"] == "held"
    assert result.state.accepted_steps == 1
    assert result.diagnostics["equilibrium_calls"] == 3
    assert result.diagnostics["v_after_m2"] < result.diagnostics["v_before_m2"]


def test_useful_policy_action_is_retained():
    controller = GoalController(straight_tracker())
    goal = controller.tracker.state.equilibrium.tip+[0., 0., .005]
    result = controller.step(goal, policy_delta=[.0005, 0.])
    assert result.action_source == "policy"
    assert not result.diagnostics["fallback_used"]
    np.testing.assert_allclose(result.applied_delta, [.0005, 0.], atol=1e-15)


def test_gradient_fallback_is_available(monkeypatch):
    controller = GoalController(straight_tracker())
    goal = controller.tracker.state.equilibrium.tip+[0., 0., .005]
    monkeypatch.setattr(controller, "_proposals", lambda *a: [("jacobian_dls", [-.001, 0.]), ("jacobian_gradient", [.001, 0.])])
    result = controller.step(goal)
    assert result.action_source == "jacobian_gradient"
    assert result.state.accepted_steps == 1


def test_unreachable_transverse_goal_stalls_without_claiming_convergence():
    controller = GoalController(straight_tracker())
    before = controller.tracker.state
    result = controller.step(before.equilibrium.tip+[.01, 0., 0.])
    assert result.status == "stalled"
    assert result.diagnostics["error_after_m"] == pytest.approx(.01)
    assert not result.diagnostics["global_convergence_certified"]
    assert not result.diagnostics["controller_stability_certified"]
    assert result.diagnostics["equilibrium_calls"] == 0
    np.testing.assert_array_equal(result.state.equilibrium.position, before.equilibrium.position)


def test_three_tube_local_goal_and_changed_waypoint():
    solver = EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()])
    q = np.array([-.1, -.05, -.02, .4, -.7, 1.2])
    initial = solver.solve(q)
    witness = BranchTracker(solver, initial)
    for _ in range(3):
        assert witness.step([0, 0, 0, .025, .015, -.01]).status == "accepted"
    goal = witness.state.equilibrium.tip
    controller = GoalController(BranchTracker(solver, initial))
    for _ in range(25):
        step = controller.step(goal)
        assert step.diagnostics["v_after_m2"] <= step.diagnostics["v_before_m2"]
        if step.status != "moving":
            break
    assert step.status == "goal_reached"
    # A new waypoint defines a new V; no claim relates V across goal changes.
    back = controller.step(initial.tip)
    assert back.diagnostics["error_before_m"] > controller.options.goal_tolerance_m
    assert back.diagnostics["v_after_m2"] < back.diagnostics["v_before_m2"]


@pytest.mark.parametrize("goal", [[np.nan, 0, 0], [0, 0], [0, 0, np.inf]])
def test_invalid_goals(goal):
    with pytest.raises(ValueError):
        GoalProgress(goal)


def test_invalid_configuration_and_action():
    with pytest.raises(ValueError):
        ControlOptions(armijo_fraction=1.)
    with pytest.raises(ValueError):
        GoalProgress([0, 0, 0], armijo_fraction=0.)
    controller = GoalController(straight_tracker())
    before = controller.tracker.state
    with pytest.raises(ValueError):
        controller.step(before.equilibrium.tip, policy_delta=[np.nan, 0.])
    np.testing.assert_array_equal(controller.tracker.state.equilibrium.position, before.equilibrium.position)
