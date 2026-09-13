"""Distance rejection, bounded start retries, and saved evaluation semantics."""
from dataclasses import asdict

import numpy as np
import pytest

from ctr_reach_envs.mechanics.curriculum import ToleranceCurriculum
from ctr_reach_envs.mechanics.geometry import TubeParameters
from ctr_reach_envs.mechanics.simple_rl_env import GoalSamplingError, JointConstrainedReachEnv, make_reach_env
from ctr_reach_envs.mechanics.solver import EquilibriumError
from ctr_reach_envs.training.cli import mechanics_defaults
from ctr_reach_envs.training.mechanics import arguments


def straight(**kwargs):
    return JointConstrainedReachEnv(
        tubes=[TubeParameters(.2, 0., .001, .002, 50e9, 23e9, 0.)],
        task_profile="generalized_reach", compute_jacobian=False, **kwargs)


def test_nine_mm_floor_cannot_be_met_by_eight_one_mm_straight_commands():
    # Analytic counterexample: on a straight tube rotation changes no tip
    # position, and eight bounded translations displace the tip by <=8 mm.
    env = straight(minimum_goal_distance_m=.009, max_goal_sampling_attempts=3,
                   max_reset_sampling_attempts=8)
    try:
        with pytest.raises(GoalSamplingError, match="required 0.009"):
            env.reset(seed=7101, options={"joints": [-.05, 0.]})
        assert env.goal_sampling_exhaustions == 1
        assert env.resampled_initial_states == 0  # Explicit joints are honored.
        assert env.rejected_trivial_goals == 3
        assert env.costs["equilibrium_calls"] == 4
        assert env.costs["failed_calls"] == 0
        assert env.failed_resets == 1
    finally:
        env.close()


def test_distance_rejection_budget_remains_bounded_across_starts():
    env = straight(minimum_goal_distance_m=.5, max_goal_sampling_attempts=2,
                   max_reset_sampling_attempts=3)
    try:
        with pytest.raises(GoalSamplingError, match="Reset exhausted 3"):
            env.reset(seed=1)
        assert env.goal_sampling_exhaustions == 3
        assert env.resampled_initial_states == 2
        assert env.rejected_trivial_goals == 6
        assert env.costs["equilibrium_calls"] == 9
        assert env.transitions == 0 and env.failed_resets == 1
        assert env.equilibrium is None and env._finished
    finally:
        env.close()


def test_reported_seed_reset_failure_recovers_without_changing_goal_floor():
    # The original seed-7101 run fails during its third reset. Steps do not
    # consume the environment's goal RNG, so this reproduces it without training.
    schedule = ToleranceCurriculum(.0085, .001, 50000)
    env = JointConstrainedReachEnv(task_profile="generalized_reach", compute_jacobian=False,
        minimum_goal_distance_m=.009, tolerance_curriculum=schedule,
        max_reset_sampling_attempts=8)
    try:
        results = []
        for i in range(3):
            obs, info = env.reset(seed=7101 if i == 0 else None)
            assert info["error"] >= .009 and not info["trivial_goal"]
            assert env.observation_space.contains(obs)
            results.append((obs, info))
        assert results[2][1]["reset_sampling_attempts"] > 1
        assert env.resampled_initial_states > 0 and env.failed_resets == 0
        assert env.transitions == 0 and env.costs["sensitivity_calls"] == 0
        for i in range(3):
            obs, info = env.reset(seed=7101 if i == 0 else None)
            for key in obs:
                np.testing.assert_array_equal(obs[key], results[i][0][key])
            assert info["reset_sampling_counts"] == results[i][1]["reset_sampling_counts"]
    finally:
        env.close()


def test_longer_witness_preserves_action_caps_and_threshold():
    env = straight(minimum_goal_distance_m=.009, goal_steps_max=20)
    try:
        obs, info = env.reset(seed=7101, options={"joints": [-.05, 0.]})
        assert info["error"] >= .009 and 9 <= info["goal_witness_steps"] <= 20
        q = info["initial_q"].copy()
        for _ in range(info["goal_witness_steps"]):
            dq = env.projected_delta(q, info["goal_witness_action"])
            assert np.max(np.abs(dq / env.action_scales)) <= 1 + 1e-12
            q += dq
            assert env.solver.constraints.is_feasible(q)
        np.testing.assert_allclose(q, info["goal_joint_witness"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(env.solver.solve(q).tip, obs["desired_goal"], atol=1e-8)
    finally:
        env.close()


def test_solver_failure_is_not_hidden_by_enabled_start_retries(monkeypatch):
    env = straight(max_reset_sampling_attempts=8)
    solve = env.solver.solve
    calls = []
    def fail_goal(q):
        calls.append(q.copy())
        if len(calls) == 2:
            raise EquilibriumError("deliberate numerical failure")
        return solve(q)
    monkeypatch.setattr(env.solver, "solve", fail_goal)
    try:
        with pytest.raises(EquilibriumError, match="deliberate numerical failure"):
            env.reset(seed=1)
        assert len(calls) == 2 and env.costs["failed_calls"] == 1
        assert env.resampled_initial_states == env.goal_sampling_exhaustions == 0
        assert env.failed_resets == 1
    finally:
        env.close()


def test_sampling_settings_are_matched_and_restored_for_fixed_evaluation():
    argv = ["--tolerance-curriculum", "exponential", "--initial-tolerance-m", ".0085",
            "--minimum-goal-distance-m", ".009", "--goal-steps-max", "20",
            "--max-goal-sampling-attempts", "32", "--max-reset-sampling-attempts", "8"]
    baseline = arguments(argv, mechanics_defaults(True), baseline=True)
    guided = arguments(argv, mechanics_defaults(False))
    for name in ("goal_steps_min", "goal_steps_max", "max_goal_sampling_attempts",
                 "max_reset_sampling_attempts", "minimum_goal_distance_m", "curriculum"):
        assert getattr(baseline, name) == getattr(guided, name)
    settings = dict(goal_steps_min=2, goal_steps_max=20, max_goal_sampling_attempts=32,
                    max_reset_sampling_attempts=8, minimum_goal_distance_m=.009)
    training = JointConstrainedReachEnv(task_profile="generalized_reach", compute_jacobian=False,
        tolerance_curriculum=ToleranceCurriculum(.0085, .001, 50000), **settings)
    config = dict(system="ctr_0", task_profile="generalized_reach", tolerance_m=.001,
        evaluation_tolerance_m=.001, observation_tolerance_bound_m=.0085,
        task_settings=training.task_settings, solver_options=asdict(training.solver.options))
    evaluation = make_reach_env(config)
    try:
        assert evaluation.task_settings == training.task_settings
        assert evaluation.tolerance_curriculum is None and evaluation.tolerance_m == .001
        assert evaluation.observation_space == training.observation_space
        for seed in (7101, 810000):
            train_obs, train_info = training.reset(seed=seed)
            eval_obs, eval_info = evaluation.reset(seed=seed)
            for key in ("achieved_goal", "desired_goal"):
                np.testing.assert_array_equal(train_obs[key], eval_obs[key])
            np.testing.assert_array_equal(train_info["initial_q"], eval_info["initial_q"])
    finally:
        evaluation.close(); training.close()


@pytest.mark.parametrize("flag", ["--max-goal-sampling-attempts", "--max-reset-sampling-attempts"])
def test_invalid_sampling_budgets_rejected_by_cli(flag):
    with pytest.raises(SystemExit):
        arguments([flag, "0"], mechanics_defaults(True), baseline=True)
