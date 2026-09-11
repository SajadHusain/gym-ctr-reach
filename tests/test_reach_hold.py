"""Generalized goals, continuing HER targets, and sustained-precision metrics."""
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from ctr_reach_envs.mechanics.geometry import TubeParameters
from ctr_reach_envs.mechanics.solver import EquilibriumError, SolverOptions
from ctr_reach_envs.mechanics.simple_rl_env import JointConstrainedReachEnv, make_reach_env
from ctr_reach_envs.mechanics.rl_metrics import ReachHoldMetrics
from ctr_reach_envs.mechanics.rl_replay import ExecutedActionHerReplayBuffer
from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG, JacobianHerReplayBuffer
from ctr_reach_envs.mechanics.rl_exploration import ExplorationDDPG
from ctr_reach_envs.mechanics.rl_policy import EquilibriumStateExtractor
from ctr_reach_envs.paper_policy import PaperMlpPolicy


def straight_env(**kwargs):
    return JointConstrainedReachEnv(tubes=[TubeParameters(.2, 0., .001, .002, 50e9, 23e9, 0.)],
        task_profile="generalized_hold", max_episode_steps=4, **kwargs)


def batch(obs):
    return {k: v[None].copy() for k, v in obs.items()}


def test_success_does_not_end_episode_and_goal_can_be_lost():
    env = straight_env(compute_jacobian=False)
    obs, _ = env.reset(options={"joints": [-.03, 0.], "goal": [0., 0., .1712]})
    _, reward, term, trunc, info = env.step([.4, 0.])
    assert info["is_success"] and reward == 0 and not term and not trunc
    _, reward, term, trunc, info = env.step([-1., 0.])
    assert not info["is_success"] and reward == -1 and not term and not trunc
    env.step([0., 0.])
    _, _, term, trunc, _ = env.step([0., 0.])
    assert not term and trunc
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0., 0.])


def test_success_at_timeout_still_bootstraps():
    env = straight_env(compute_jacobian=False)
    env.reset(options={"joints": [-.03, 0.], "goal": [0., 0., .17]})
    for step in range(4):
        _, reward, term, trunc, info = env.step([0., 0.])
        assert reward == 0 and info["is_success"] and not term
        assert trunc == (step == 3)


def test_vectorized_rewards_are_separate_from_termination():
    env = straight_env(compute_jacobian=False)
    achieved = np.array([[0., 0., .1], [0., 0., .2]])
    goals = np.array([[0., 0., .1005], [0., 0., .21]])
    infos = [{"position_tolerance": .001}]*2
    np.testing.assert_array_equal(env.compute_success(achieved, goals, infos), [True, False])
    np.testing.assert_array_equal(env.compute_reward(achieved, goals, infos), [0., -1.])
    np.testing.assert_array_equal(env.compute_terminated(achieved, goals, infos), [False, False])
    assert env.compute_reward(achieved, goals, infos).dtype == np.float32


@pytest.mark.parametrize("buffer_class", [ExecutedActionHerReplayBuffer, JacobianHerReplayBuffer])
def test_her_relabelled_hits_keep_nonterminal_targets_including_timeout(monkeypatch, buffer_class):
    env = straight_env(compute_jacobian=buffer_class is JacobianHerReplayBuffer)
    vec = DummyVecEnv([lambda: straight_env(compute_jacobian=False)])
    try:
        replay = buffer_class(20, vec.observation_space, vec.action_space, env=vec,
                             copy_info_dict=True, device="cpu", action_semantics="proposal")
        obs, _ = env.reset(options={"joints": [-.03, 0.], "goal": [0., 0., .1705]})
        for step in range(4):
            action = np.array([.1, .1])
            nxt, reward, term, trunc, info = env.step(action)
            info["TimeLimit.truncated"] = trunc
            replay.add(batch(obs), batch(nxt), action[None], np.array([reward]), np.array([term or trunc]), [info])
            obs = nxt
        indices, env_indices = np.arange(4), np.zeros(4, dtype=int)
        monkeypatch.setattr(replay, "_sample_goals", lambda b, e: replay.next_observations["achieved_goal"][b,e].copy())
        real = replay._get_real_samples(indices, env_indices)
        virtual = replay._get_virtual_samples(indices, env_indices)
        assert torch.count_nonzero(real.dones) == 0
        assert torch.count_nonzero(virtual.dones) == 0
        assert torch.count_nonzero(virtual.rewards) == 0
        assert replay.dones[3,0] == replay.timeouts[3,0] == 1
        # No future success flags or episode history are included in the observation.
        assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    finally:
        vec.close(); env.close()


def test_signed_goal_and_start_diversity_reproducibility_and_witnesses():
    env = JointConstrainedReachEnv(task_profile="generalized_hold", compute_jacobian=False)
    commands, starts, lengths = [], [], []
    for seed in range(16):
        obs, info = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        assert info["error"] >= .002 and not info["trivial_goal"]
        q = info["initial_q"].copy()
        for _ in range(info["goal_witness_steps"]):
            q += env.projected_delta(q, info["goal_witness_action"])
        np.testing.assert_allclose(q, info["goal_joint_witness"], rtol=0, atol=1e-14)
        assert env.solver.constraints.is_feasible(q)
        commands.append(info["goal_witness_action"]); starts.append(info["initial_q"]); lengths.append(info["goal_witness_steps"])
    commands, starts = np.array(commands), np.array(starts)
    assert np.all(commands.min(0) < 0) and np.all(commands.max(0) > 0)
    assert starts[:,3:].min() < -1 and starts[:,3:].max() > 1
    assert np.any(np.abs(starts[:,4]-starts[:,3]) > .01)
    assert min(lengths) >= 2 and max(lengths) <= 8 and len(set(lengths)) > 1
    obs1, info1 = env.reset(seed=7101)
    obs2, info2 = env.reset(seed=7101)
    for key in obs1:
        np.testing.assert_array_equal(obs1[key], obs2[key])
    np.testing.assert_array_equal(info1["goal_joint_witness"], info2["goal_joint_witness"])
    assert env.costs["sensitivity_calls"] == env.costs["stability_calls"] == 0


def test_numerical_goal_failure_is_not_silently_resampled(monkeypatch):
    env = straight_env(compute_jacobian=False)
    original = env.solver.solve
    calls = []
    def fail_goal(q):
        calls.append(q.copy())
        if len(calls) == 2:
            raise EquilibriumError("goal shooting budget")
        return original(q)
    monkeypatch.setattr(env.solver, "solve", fail_goal)
    with pytest.raises(EquilibriumError, match="goal shooting"):
        env.reset(seed=1)
    assert len(calls) == 2 and env.costs["failed_calls"] == 1 and env.failed_resets == 1
    np.testing.assert_array_equal(env.last_failed_solve_q, calls[-1])


def test_trivial_goal_sampling_is_bounded_and_counted():
    env = straight_env(compute_jacobian=False, minimum_goal_distance_m=.5, max_goal_sampling_attempts=2)
    with pytest.raises(RuntimeError, match="bounded sampling budget"):
        env.reset(seed=1)
    assert env.rejected_trivial_goals == 2 and env.costs["equilibrium_calls"] == 3


@pytest.mark.parametrize("settings", [{"task_profile":"unknown"}, {"goal_steps_min":9},
    {"goal_steps_max":0}, {"initial_rotation_span_rad":float("nan")}, {"minimum_goal_distance_m":.0001}])
def test_invalid_task_settings(settings):
    with pytest.raises(ValueError):
        JointConstrainedReachEnv(**settings)


def test_checkpoint_task_restore_and_legacy_compatibility():
    old = make_reach_env({"system":"ctr_0"})
    assert old.terminate_on_success and old.task_profile == "legacy"
    config = {"system":"ctr_0", "task_profile":"generalized_hold", "episode_steps":40,
              "task_settings":{"initial_rotation_span_rad":.12,"goal_steps_min":3,"goal_steps_max":6}}
    new = make_reach_env(config)
    assert new.max_episode_steps == 40 and not new.terminate_on_success
    assert new.task_settings["initial_rotation_span_rad"] == .12
    assert make_reach_env(config, task_profile="legacy").terminate_on_success


def test_checkpoint_restores_solver_budget_and_hard_reported_configuration():
    config = {"system":"ctr_0", "task_profile":"generalized_hold",
              "solver_options":{**SolverOptions().__dict__, "max_shooting_evaluations":500}}
    env = make_reach_env(config)
    assert env.solver.options.max_shooting_evaluations == 500
    # Regression configuration from the Windows seed-7101 budget failure.
    # Require convergence within the restored budget, not an exact evaluation
    # count: numerical libraries/platforms can take different shooting paths.
    q = [-0.30848364632511605, -0.27811755904387275, -0.1541822453017143,
         -0.510274642092774, -3.88315894370072, -2.0636713227539984]
    result = env.solver.solve(q)
    assert 0 < result.diagnostics["shooting_evaluations"] <= env.solver.options.max_shooting_evaluations
    assert np.all(np.isfinite(result.position))
    assert result.diagnostics["boundary_residual_scaled"] <= env.solver.options.boundary_tolerance
    assert np.max(np.abs(result.distal_torsional_strain))*env.solver.scale <= env.solver.options.boundary_tolerance
    env.close()


def test_solver_options_type_is_explicit():
    with pytest.raises(ValueError, match="SolverOptions"):
        JointConstrainedReachEnv(solver_options={"max_shooting_evaluations":500})


def test_holding_requires_a_complete_final_window_and_survives_late_recovery():
    m = ReachHoldMetrics(.001, 3)
    for value in [.002, .0008, .0011, .0009]:
        m.add(value)
    result = m.summary()
    assert result["reached"] and result["first_success_step"] == 2
    assert result["final_success"] and not result["sustained_success"]
    m.add(.0005); m.add(.0006)
    result = m.summary()
    assert result["sustained_success"] and result["hold_window_max_error_m"] == .0009
    assert not m.summary(episode_complete=False)["sustained_success"]
    short = ReachHoldMetrics(.001, 3); short.add(.0001)
    assert not short.summary()["sustained_success"] and short.summary()["hold_window_max_error_m"] is None


@pytest.mark.parametrize("guided", [False, True])
def test_continuing_training_updates_and_actor_only_reload(tmp_path, guided):
    torch.set_num_threads(1)
    env = straight_env(compute_jacobian=guided)
    cls = JacobianDDPG if guided else ExplorationDDPG
    kwargs = dict(physics_lengths=[.2], physics_weight=.1, physics_final_weight=.1) if guided else {}
    model = cls(PaperMlpPolicy, env, seed=9, learning_starts=4, buffer_size=100,
        batch_size=8, train_freq=1, gradient_steps=1, device="cpu", **kwargs,
        policy_kwargs={"net_arch":[16,16], "features_extractor_class":EquilibriumStateExtractor,
                       "features_extractor_kwargs":{"length_scale":.2}},
        replay_buffer_class=JacobianHerReplayBuffer if guided else ExecutedActionHerReplayBuffer,
        replay_buffer_kwargs={"copy_info_dict":True,"action_semantics":"proposal"})
    model.learn(12)
    assert model._n_updates == 8
    assert all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
    sample = model.replay_buffer.sample(32)
    assert torch.count_nonzero(sample.dones) == 0
    path = tmp_path/"model.zip"; model.save(path)
    evaluation = straight_env(compute_jacobian=False)
    loaded = cls.load(path, env=evaluation, device="cpu")
    obs, _ = evaluation.reset(seed=7101)
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0], loaded.predict(obs,deterministic=True)[0])
    evaluation.step(loaded.predict(obs,deterministic=True)[0])
    assert evaluation.costs["sensitivity_calls"] == 0
    model.get_env().close(); evaluation.close()


def test_generalized_gymnasium_and_sb3_contract():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = straight_env(compute_jacobian=False)
    check_env(env, skip_render_check=True)
    sb3_check(env)
