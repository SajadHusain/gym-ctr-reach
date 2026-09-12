import numpy as np
import pytest
import torch

from ctr_reach_envs.envs.goal_tolerance import GoalTolerance
from ctr_reach_envs.paper_config import env_kwargs_for_profile, paper_configuration
from ctr_reach_envs.paper_policy import LateActionQ, PaperDDPG, PaperStateExtractor
from rl_utils import load_policy, make_env
from ctr_reach_envs.training.paper import build_model


def test_paper_curriculum_and_evaluation_are_distinct():
    train = env_kwargs_for_profile("paper-2024")
    evaluation = env_kwargs_for_profile("paper-2024", evaluation=True)
    schedule = GoalTolerance(train["goal_tolerance_parameters"])
    assert schedule.get_tol() == 0.02
    schedule.update(750_000)
    assert schedule.get_tol() == pytest.approx(np.sqrt(0.02 * 0.001))
    schedule.update(1_500_000)
    assert schedule.get_tol() == pytest.approx(0.001)
    schedule.update(3_000_000)
    assert schedule.get_tol() == pytest.approx(0.001)
    fixed = GoalTolerance(evaluation["goal_tolerance_parameters"])
    fixed.update(0)
    assert fixed.get_tol() == 0.001
    assert train["max_steps_per_episode"] == 200
    assert train["constrain_alpha"] is False
    assert env_kwargs_for_profile("current")["max_steps_per_episode"] == 150


def test_relabelled_goal_changes_policy_error_without_stale_state():
    env = make_env(evaluation=False, seed=1, profile="paper-2024")
    try:
        obs, _ = env.reset(seed=1)
        tensors = {key: torch.as_tensor(value[None]) for key, value in obs.items()}
        extractor = PaperStateExtractor(env.observation_space)
        original = extractor(tensors)
        relabelled = {key: value.clone() for key, value in tensors.items()}
        relabelled["desired_goal"] += torch.tensor([[0.002, 0.003, -0.004]])
        changed = extractor(relabelled)
        assert original.shape == (1, 13)
        torch.testing.assert_close(original[:, :9], changed[:, :9])
        torch.testing.assert_close(original[:, 12:], changed[:, 12:])
        torch.testing.assert_close(changed[:, 9:12] - original[:, 9:12],
                                   torch.tensor([[-0.002, -0.003, 0.004]]))
    finally:
        env.close()


def test_late_action_critic_structure_and_action_gradient():
    q = LateActionQ(13, 6, [256, 256, 256], torch.nn.ReLU)
    assert q.state_layer[0].in_features == 13
    assert q.value_layers[0].in_features == 262
    x = torch.randn(4, 19, requires_grad=True)
    output = q(x)
    output.sum().backward()
    assert output.shape == (4, 1)
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, 13:].abs().sum() > 0


def test_training_exploration_replay_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    env = make_env(evaluation=False, seed=2, profile="paper-2024")
    restored_env = make_env(evaluation=True, seed=3, profile="paper-2024")
    try:
        spec = paper_configuration()
        spec["buffer_size"] = 2000
        model = build_model(env, spec, seed=2, verbose=0)
        assert model.actor.mu[0].in_features == 13
        assert [layer.out_features for layer in model.actor.mu if isinstance(layer, torch.nn.Linear)] == [256, 256, 256, 6]
        assert isinstance(model.critic.q_networks[0], LateActionQ)
        np.testing.assert_allclose(model.action_noise._sigma, [0.0018] * 3 + [0.025] * 3)
        model._last_obs = model.env.reset()
        model.random_exploration = 1.0
        for _ in range(10):
            action, replay_action = model._sample_action(0)
            np.testing.assert_allclose(action, replay_action, atol=1e-7, rtol=0)
            assert model.action_space.contains(action[0])
        model.random_exploration = 0.0
        expected, _ = model.predict(model._last_obs, deterministic=True)
        actual, _ = model._sample_action(0)
        np.testing.assert_allclose(actual, expected)
        model.random_exploration = spec["random_exploration"]
        initial_weights = model.actor.mu[0].weight.detach().clone()
        model.learn(total_timesteps=400)
        assert model._n_updates > 0
        assert not torch.equal(initial_weights, model.actor.mu[0].weight)
        path = tmp_path / "paper_model.zip"
        model.save(path)
        obs, _ = restored_env.reset(seed=8)
        expected, _ = model.predict(obs, deterministic=True)
        loaded = load_policy(path, restored_env, "paper-2024")
        actual, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
        assert loaded.paper_configuration["total_timesteps"] == 3_000_000
        # Inference loading uses DDPG; resuming the subclass preserves its hooks.
        restored = PaperDDPG.load(path, env=restored_env)
        assert restored.random_exploration == spec["random_exploration"]
        with pytest.raises(ValueError, match="Checkpoint profile"):
            load_policy(path, restored_env, "current")
    finally:
        env.close()
        restored_env.close()
