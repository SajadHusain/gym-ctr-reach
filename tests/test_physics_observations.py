import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import DDPG

from ctr_reach_envs.ivp.config import load_spec, make_env, resolve
from ctr_reach_envs.ivp.observations import extension_margins, observation_settings, physics_features
from ctr_reach_envs.ivp.sensitivity import tip_sensitivity, IVPSensitivityError
from ctr_reach_envs.paper_policy import PaperDDPG
from ctr_reach_envs.training.original import AuditCallback, build_model, main


def configuration(mode="jacobian_limits", weight=0.):
    spec, environment, provenance = load_spec()
    spec.update(max_steps_per_episode=8, batch_size=8, buffer_size=128,
                hidden_layers=[16, 16, 16], total_timesteps=24, curriculum_steps=8,
                physics_observation=dict(mode=mode, scale_m=.002, version=1))
    spec["legacy_defaults"].update(rollout_steps=8, gradient_steps=2)
    physics = dict(weight=weight, final_weight=0., anneal_steps=8,
                   integration="rl_priority", max_aux_ratio=.1, gain=.5,
                   scale_m=.002, max_tip_step_m=.002, objective="hybrid")
    return resolve(spec, environment, segment_mode="continuous", seed=7101,
                   physics=physics, provenance=provenance)


def independent_encoding(env, joints):
    result = tip_sensitivity(env.model, joints, env.system)
    scaled = result.jacobian * env.action_scale * env.n_substeps / .002
    return (scaled / (1. + abs(scaled))).ravel().astype(np.float32)


def test_feature_units_and_extension_boundaries():
    context = dict(jacobian=np.ones((3, 6)), jacobian_valid=True,
                   joints=np.array([-.15, -.1, -.05, 0., 0., 0.]),
                   action_scales=np.array([.001] * 3 + [.01] * 3))
    encoded = physics_features(context, [.3, .2, .1], 2, observation_settings({"mode": "jacobian_limits"}))
    np.testing.assert_allclose(encoded[:18].reshape(3, 6)[:, :3], .5)
    np.testing.assert_allclose(encoded[:18].reshape(3, 6)[:, 3:], 10. / 11.)
    assert encoded[18] == 1. and encoded.dtype == np.float32
    # Fully inserted tubes touch the beta upper boxes and one ordering face.
    margins = extension_margins(np.zeros(6), [.3, .2, .1])
    np.testing.assert_array_equal(margins, [1., 1., 1., 0., 0., 0., 0., 0., 1., 1.])


@pytest.mark.parametrize("mode,count", [("none", 0), ("jacobian", 19), ("jacobian_limits", 29), ("zeros", 29)])
def test_modes_preserve_plant_and_random_sampling(mode, count):
    env, baseline = make_env(configuration(mode)), make_env(configuration("none"))
    try:
        obs, _ = env.reset(seed=7101)
        other, _ = baseline.reset(seed=7101)
        assert env.observation_space.contains(obs)
        assert ("physics" in obs) == bool(count)
        if count:
            assert obs["physics"].shape == (count,)
        rng = np.random.default_rng(9)
        for _ in range(3):
            action = rng.uniform(-.1, .1, 6).astype(np.float32)
            a, b = env.step(action), baseline.step(action)
            assert a[1:4] == b[1:4]
            for key in other:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            np.testing.assert_array_equal(a[4]["q_after"], b[4]["q_after"])
            assert env.observation_space.contains(a[0])
        if mode in ("none", "zeros"):
            assert env.costs["sensitivity_calls"] == 0
        if mode == "zeros":
            assert not np.any(a[0]["physics"])
    finally:
        env.close(); baseline.close()


def test_current_and_next_features_are_aligned_and_cache_is_isolated():
    env = make_env(configuration())
    try:
        obs, _ = env.reset(seed=7101)
        q0 = env.trig_obj.joints.copy()
        np.testing.assert_allclose(obs["physics"][:18], independent_encoding(env, q0), atol=1e-7)
        assert env.costs["sensitivity_calls"] == 1
        changed_goal = env.set_goal(env.desired_goal + [.001, 0., 0.])
        np.testing.assert_array_equal(changed_goal["physics"], obs["physics"])
        context = env.source_physics()
        context["jacobian"][:] = 0.
        context["joints"][:] = 0.
        assert env.costs["sensitivity_calls"] == 1
        action = np.array([.02, -.03, .04, .01, -.02, .03], dtype=np.float32)
        after, _, _, _, info = env.step(action)
        assert env.costs["sensitivity_calls"] == 2
        np.testing.assert_array_equal(info["physics"]["joints"], q0)
        np.testing.assert_allclose(info["physics"]["jacobian"], tip_sensitivity(env.model, q0).jacobian)
        np.testing.assert_allclose(after["physics"][:18], independent_encoding(env, env.trig_obj.joints), atol=1e-7)
        assert not np.array_equal(after["physics"][:18], obs["physics"][:18])
        env.source_physics()
        assert env.costs["sensitivity_calls"] == 2
        env.reset(seed=7101)
        assert env.costs["sensitivity_calls"] == 3
    finally:
        env.close()


@pytest.mark.parametrize("failure", ["exception", "nan"])
def test_invalid_jacobian_is_masked_without_losing_joint_information(monkeypatch, failure):
    env, baseline = make_env(configuration()), make_env(configuration("none"))
    def invalid(*args):
        if failure == "exception":
            raise IVPSensitivityError("test unavailable derivative")
        return SimpleNamespace(jacobian=np.full((3, 6), np.nan), tip=np.zeros(3), rhs_evaluations=0)
    monkeypatch.setattr("ctr_reach_envs.ivp.env.tip_sensitivity", invalid)
    try:
        obs, _ = env.reset(seed=7101)
        baseline.reset(seed=7101)
        assert not np.any(obs["physics"][:19])
        assert np.any(obs["physics"][19:])
        action = np.full(6, .01, dtype=np.float32)
        a, b = env.step(action), baseline.step(action)
        assert a[1:4] == b[1:4]
        np.testing.assert_array_equal(a[4]["q_after"], b[4]["q_after"])
        assert not np.any(a[0]["physics"][:19]) and not a[4]["physics"]["jacobian_valid"]
        assert env.costs["invalid_jacobians"] == 2
        assert env.observation_space.contains(a[0])
    finally:
        env.close(); baseline.close()


@pytest.mark.parametrize("weight", [0., .1])
def test_her_features_survive_relabel_and_guidance_shutdown(tmp_path, weight):
    torch.set_num_threads(1)
    cfg = configuration(weight=weight)
    env = make_env(cfg, compute_jacobian=weight > 0)
    model = build_model(env, cfg)
    audit = AuditCallback(env, cfg, tmp_path, checkpoint_freq=0, progress_every=100)
    try:
        before = {name: p.detach().clone() for name, p in model.actor.named_parameters()}
        model.learn(24, callback=audit)
        assert model._n_updates > 0 and env.compute_jacobian
        assert env.costs["sensitivity_calls"] > 24
        assert any(not torch.equal(before[name], p) for name, p in model.actor.named_parameters())
        if weight == 0:
            assert type(model) is PaperDDPG
            assert not hasattr(model, "_physics_loss")
        replay = model.replay_buffer
        indices, env_indices = np.array([0]), np.array([0])
        new_goal = replay.next_observations["achieved_goal"][0:1, 0].copy()
        original_goal = replay.observations["desired_goal"][0, 0].copy()
        replay._sample_goals = lambda b, e: new_goal
        sample = replay._get_virtual_samples(indices, env_indices)
        for field, storage in (("observations", replay.observations), ("next_observations", replay.next_observations)):
            np.testing.assert_array_equal(getattr(sample, field)["physics"].cpu().numpy()[0], storage["physics"][0, 0])
        np.testing.assert_array_equal(replay.observations["desired_goal"][0, 0], original_goal)
        assert sample.rewards.item() == 0. and sample.dones.item() == 1.
        features = model.actor.features_extractor(sample.observations)
        torch.testing.assert_close(features[:, 9:12], sample.observations["achieved_goal"] - sample.observations["desired_goal"])
        torch.testing.assert_close(features[:, 13:], sample.observations["physics"])
        # The features must actually reach BOTH networks, not just be stored.
        obs = {k: v.detach().clone() for k, v in sample.observations.items() if not k.startswith("__physics_")}
        obs["physics"].requires_grad_(True)
        action = model.actor(obs)
        actor_gradient = torch.autograd.grad(action.sum(), obs["physics"], retain_graph=True)[0]
        critic_gradient = torch.autograd.grad(model.critic(obs, action.detach())[0].sum(), obs["physics"])[0]
        assert actor_gradient.abs().max() > 0 and critic_gradient.abs().max() > 0
        # The final replay observation belongs to the terminal state, not the
        # automatically reset state used for the next episode.
        terminal = np.flatnonzero(replay.dones[:replay.pos, 0])[0]
        info = replay.infos[terminal, 0]
        final_features = replay.next_observations["physics"][terminal, 0]
        np.testing.assert_allclose(final_features[19:], extension_margins(info["q_after"], env.trig_obj.tube_lengths[0]), atol=1e-7)
        if final_features[18]:
            np.testing.assert_allclose(final_features[:18], independent_encoding(env, info["q_after"]), atol=1e-7)
        else:
            assert not np.any(final_features[:18])
    finally:
        audit.close(); env.close()


def test_checkpoint_evaluation_restores_information_and_reports_usage(tmp_path, capsys):
    from evaluate_original_ddpg_her import main as evaluate
    run = tmp_path / "run"
    main(["--physics-observation", "jacobian_limits", "--physics-observation-scale-m", ".004",
          "--segment-mode", "continuous",
          "--total-timesteps", "16", "--episode-steps", "8", "--batch-size", "8",
          "--buffer-size", "64", "--hidden-width", "16", "--train-freq", "8",
          "--gradient-steps", "2", "--checkpoint-freq", "0", "--output-dir", str(run)], baseline=True)
    cfg = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"] and summary["physics_updates"] == 0
    assert summary["actor_uses_jacobian"] and summary["gradient_updates"] > 0
    env = make_env(cfg, evaluation=True, compute_jacobian=False)
    try:
        model = DDPG.load(run / "final_model.zip", env=env)
        obs, _ = env.reset(seed=99)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (6,) and env.compute_jacobian
        assert model.actor.features_extractor.features_dim == 42
    finally:
        env.close()
    output = run / "evaluation"
    evaluate([str(run / "final_model.zip"), "--episodes", "2", "--max-steps", "2",
              "--output-dir", str(output)])
    evaluation = json.loads((output / "summary.json").read_text())
    assert evaluation["complete"] and evaluation["failures"] == 0
    assert evaluation["actor_uses_jacobian"] and evaluation["costs"]["sensitivity_calls"] >= 2
    assert evaluation["physics_observation"]["scale_m"] == .004
    capsys.readouterr()
    main(["--config", str(run / "config.json"), "--dry-run", "--output-dir", str(tmp_path / "copy")], baseline=True)
    restored_config = json.loads(capsys.readouterr().out)
    assert restored_config["spec"]["physics_observation"] == cfg["spec"]["physics_observation"]
    main(["--config", str(run / "config.json"), "--dry-run", "--physics-observation", "none",
          "--physics-observation-scale-m", ".01", "--output-dir", str(tmp_path / "override")], baseline=True)
    override = json.loads(capsys.readouterr().out)["spec"]["physics_observation"]
    assert override == dict(mode="none", scale_m=.01, version=1)
    restored, _, _ = load_spec(run / "config.json")
    assert restored["physics_observation"] == cfg["spec"]["physics_observation"]


def test_old_config_defaults_and_feature_control_architecture():
    old = configuration("none")
    del old["spec"]["physics_observation"]
    env = make_env(old, evaluation=True)
    try:
        assert "physics" not in env.observation_space.spaces
        assert not env.compute_jacobian
    finally:
        env.close()
    settings = [configuration(mode) for mode in ("none", "jacobian_limits", "zeros")]
    assert len({cfg["environment_fingerprint"] for cfg in settings}) == 1
    states = []
    for cfg in settings[1:]:
        env = make_env(cfg)
        try:
            model = build_model(env, cfg)
            states.append({key: value.clone() for key, value in model.policy.state_dict().items()})
        finally:
            env.close()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


def test_feature_environment_contract():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = make_env(configuration())
    try:
        check_env(env, skip_render_check=True)
        sb3_check(env)
    finally:
        env.close()


@pytest.mark.parametrize("settings", [{"mode": "bad"}, {"scale_m": 0}, {"scale_m": float("nan")}, {"version": 2}])
def test_invalid_feature_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        observation_settings(settings)
