"""Behavioral checks for the mpcrl adapter, including the original CTR plant."""
import json

import numpy as np
import pytest

from ctr_reach_envs.mpc.learning import LearningOptions, MPCQLearner, td_residual
from ctr_reach_envs.ivp.config import load_spec, resolve, make_env


def source_config():
    spec, env, provenance = load_spec()
    return resolve(spec, env, segment_mode="continuous", seed=10, physics={}, provenance=provenance)


def linear_learner(options=None):
    lengths = np.array([.431, .332, .174])
    return MPCQLearner(lambda q: q[:3]+lengths, lengths, np.r_[np.full(3, .01), np.full(3, .1)],
                       options=options or LearningOptions(horizon=1))


def test_terminal_and_truncated_targets_are_distinct():
    assert td_residual(0., 2., 99., .95, True) == -2.
    assert td_residual(1., 2., 5., .95, False) == pytest.approx(3.75)


@pytest.mark.parametrize("changes", [dict(horizon=0), dict(gamma=1.1),
    dict(learning_rate=0.), dict(max_model_evaluations=-1), dict(exploration_strength=-1.)])
def test_options_validate(changes):
    with pytest.raises(ValueError): LearningOptions(**changes)


def test_mpc_is_policy_and_q_fixes_the_supplied_action():
    learner = linear_learner()
    q = np.array([-.3, -.23, -.12, 0., 0., 0.])
    goal = q[:3]+learner.constraints.lengths+.001
    state = learner.state(q, goal)
    action, v, _ = learner.solve(state)
    assert np.all(action[:3] > .09)
    assert np.max(abs(action)) <= 1.
    _, qsol, _ = learner.solve(state, action=action)
    assert qsol.f == pytest.approx(v.f, abs=1e-5)
    _, bad_qsol, _ = learner.solve(state, action=np.zeros(6))
    assert bad_qsol.f > qsol.f
    np.testing.assert_allclose(np.asarray(bad_qsol.vals["action"])[:, 0], 0., atol=1e-7)


def test_lagrangian_parameter_sensitivity_matches_resolved_finite_difference():
    learner = linear_learner(LearningOptions(horizon=2))
    q = np.array([-.3, -.23, -.12, 0., 0., 0.])
    state = learner.state(q, q[:3]+learner.constraints.lengths+.002)
    action = np.array([.03, .04, .05, 0., 0., 0.])
    _, sol, _ = learner.solve(state, action=action)
    derivative = np.asarray(learner.agent._sensitivity(sol)).reshape(-1)
    old = learner.agent.learnable_parameters.value.copy()
    fd = np.zeros_like(old)
    for index in (0, 3, 5, 8):
        delta = 1e-4
        plus, minus = old.copy(), old.copy()
        plus[index] += delta; minus[index] -= delta
        learner.agent.learnable_parameters.update_values(plus)
        high = learner.solve(state, action=action)[1].f
        learner.agent.learnable_parameters.update_values(minus)
        low = learner.solve(state, action=action)[1].f
        fd[index] = (high-low)/(2*delta)
    learner.agent.learnable_parameters.update_values(old)
    np.testing.assert_allclose(derivative[[0, 3, 5, 8]], fd[[0, 3, 5, 8]], atol=1e-4, rtol=.01)
    assert derivative[-1] == pytest.approx(1.)


def test_td_learning_changes_parameters_within_bounds():
    learner = linear_learner()
    q = np.array([-.3, -.23, -.12, 0., 0., 0.])
    state = learner.state(q, q[:3]+learner.constraints.lengths+.001)
    action, _, _ = learner.solve(state)
    _, sol, _ = learner.solve(state, action=action)
    old = learner.agent.learnable_parameters.value.copy()
    record = learner.learn_transition(0., sol, terminated=True)
    assert record["td_error"] == pytest.approx(-sol.f)
    assert record["parameter_change_norm"] > 0
    assert not np.array_equal(old, learner.agent.learnable_parameters.value)
    pars = learner.agent.learnable_parameters
    assert np.all(pars.value >= pars.lb) and np.all(pars.value <= pars.ub)


def test_exploration_is_constrained_and_bellman_values_are_unperturbed():
    learner = linear_learner()
    q = np.array([-.3, -.23, -.12, 0., 0., 0.])
    state = learner.state(q, q[:3]+learner.constraints.lengths+.001)
    a, baseline, _ = learner.solve(state)
    noisy, _, _ = learner.solve(state, perturbation=np.array([.5, 0., 0., 0., 0., 0.]))
    assert abs(noisy[0]-a[0]) > .05
    assert np.max(abs(noisy)) <= 1.
    _, again, _ = learner.solve(state)
    assert again.f == pytest.approx(baseline.f, abs=1e-7)
    _, qsol, _ = learner.solve(state, action=noisy)
    assert qsol.f > baseline.f


def test_update_cap_is_relative_even_for_small_effort_weights():
    learner = linear_learner()
    q = np.array([-.3, -.23, -.12, 0., 0., 0.])
    state = learner.state(q, q[:3]+learner.constraints.lengths+.001)
    action, _, _ = learner.solve(state)
    _, sol, _ = learner.solve(state, action=action)
    old = learner.agent.learnable_parameters.value.copy()
    learner.learn_transition(1000., sol, terminated=True)
    new = learner.agent.learnable_parameters.value
    limits = learner.options.max_parameter_change*abs(old)
    limits[-1] = learner.options.max_parameter_change
    assert np.all(abs(new-old) <= limits+1e-9)
    assert abs(new[3]-old[3]) == pytest.approx(limits[3], abs=1e-9)


def test_model_budget_aborts_without_an_unvalidated_action():
    learner = linear_learner(LearningOptions(max_model_evaluations=1))
    q = np.array([-.3, -.23, -.12, 0., 0., 0.])
    with pytest.raises(RuntimeError):
        learner.solve(learner.state(q, q[:3]+learner.constraints.lengths+.01))
    assert learner.callback.calls <= 1
    assert learner.updates == 0


def test_actual_ctr_prediction_and_normalized_action_match_live_plant():
    config = source_config(); env = make_env(config, evaluation=True, tolerance=.0015)
    try:
        q = np.array([-.28, -.2, -.1, .1, -.1, .05])
        goal = env.model.forward_kinematics(q+[.002, .001, .001, .03, -.02, .01], 0)
        env.reset(seed=10, options=dict(initial_joints=q, goal=goal))
        learner = MPCQLearner.from_env(env, config, LearningOptions(horizon=1))
        state = learner.state(env.trig_obj.joints, env.desired_goal)
        initial_error = np.linalg.norm(env.achieved_goal-env.desired_goal)
        action, sol, _ = learner.solve(state)
        learner.check_executed_action(env, action, sol)
        predicted_q = np.asarray(sol.vals["state"])[:6, 1]*learner.caps
        predicted_tip = learner.callback.tip(predicted_q/learner.caps)*learner.options.tracking_scale_m
        env.step(action)
        np.testing.assert_allclose(env.trig_obj.joints, predicted_q, atol=2e-7)
        np.testing.assert_allclose(env.achieved_goal, predicted_tip, atol=2e-7)
        assert np.linalg.norm(env.achieved_goal-env.desired_goal) < initial_error
    finally:
        env.close()


def test_training_checkpoint_evaluation_and_integrity(tmp_path, monkeypatch):
    import run_ctr_mpcrl as entry
    config = source_config(); path = tmp_path/"source.json"; path.write_text(json.dumps(config))
    original_make = entry.make_env

    def make_fixture_env(*args, **kwargs):
        env = original_make(*args, **kwargs)
        reset = env.reset
        q = np.array([-.28, -.2, -.1, .1, -.1, .05])
        goal = env.model.forward_kinematics(q+[.0001, .0001, .0001, 0., 0., 0.], 0)
        env.reset = lambda seed=None: reset(seed=seed, options=dict(initial_joints=q, goal=goal))
        return env

    monkeypatch.setattr(entry, "make_env", make_fixture_env)
    out = tmp_path/"train"
    result = entry.main(["train", "--config", str(path), "--steps", "1", "--horizon", "1",
        "--exploration-strength", "0", "--output-dir", str(out)])
    assert result["complete"], result
    assert result["updates"] == result["timesteps"] == 1
    saved = entry.read_checkpoint(out/"checkpoint_final.json")
    assert saved["parameters"] == result["parameters"]
    evaluated = entry.main(["evaluate", "--checkpoint", str(out/"checkpoint_final.json"),
        "--episodes", "1", "--output-dir", str(tmp_path/"eval")])
    assert evaluated["complete"] and evaluated["updates"] == 0
    assert evaluated["parameters"] == saved["parameters"]
    assert evaluated["environment_fingerprint"] == config["environment_fingerprint"]
    damaged = json.loads((out/"checkpoint_final.json").read_text()); damaged["parameters"]["offset"] = [99.]
    (tmp_path/"bad.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError): entry.read_checkpoint(tmp_path/"bad.json")


def test_short_training_on_sampled_ctr_task(tmp_path):
    """Exercise actual reset sampling and nonterminal Q/V updates at horizon 2."""
    import run_ctr_mpcrl as entry
    config = source_config(); path = tmp_path/"source.json"
    path.write_text(json.dumps(config))
    result = entry.main(["train", "--config", str(path), "--steps", "3",
        "--seed", "10", "--horizon", "2", "--output-dir", str(tmp_path/"pilot")])
    assert result["complete"], result
    assert result["updates"] == result["timesteps"] == 3
    assert result["failures"] == 0
