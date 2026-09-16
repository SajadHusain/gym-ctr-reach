"""Independent gradient, plant-parity, budget and saved-training regressions."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import DDPG

from ctr_reach_envs.ivp.config import make_env
from ctr_reach_envs.ivp.rl import OriginalJacobianDDPG, OriginalJacobianLoss
from ctr_reach_envs.ivp.short_horizon import (
    IVPBackwardBackend, ShortHorizonReturn, UnavailableWindow,
    differentiable_tip, observation_from_state, terminal_value, validate_short_horizon,
)
from ctr_reach_envs.training.cli import guided_main
from ctr_reach_envs.training.original import build_model
from test_original_ivp import configuration


def short_config():
    cfg = configuration()
    cfg["physics"].update(loss_kind="short_horizon", max_aux_ratio=.1,
        short_horizon_steps=2, short_horizon_batch_size=2, short_horizon_every=2,
        short_horizon_start_steps=0)
    return cfg


class FeedbackActor(torch.nn.Module):
    """Smooth actor with both trainable bias and nonzero feedback derivatives."""
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([.021, -.017, .011, .031, -.027, .019], dtype=torch.float64))

    def forward(self, obs):
        tip = obs["achieved_goal"].double()
        feedback = torch.cat((tip, tip), dim=1) * .7
        return torch.tanh(self.bias[None] + feedback)


class SmoothCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = torch.nn.Identity()
        self.q = torch.nn.Linear(9, 1, bias=False).double()
        with torch.no_grad():
            self.q.weight[:] = torch.tensor([[3., -5., 7., .11, -.13, .17, .19, -.23, .29]])
        self.q_networks = [self.q]

    def extract_features(self, observation, extractor):
        return observation["achieved_goal"].double()


def real_sample(env):
    # Away from joint clipping and segment-order ties.
    q = np.array([-.201234567, -.121345678, -.032456789, .3, -.6, 1.1])
    tip = env.model.forward_kinematics(q, 0)
    goal = tip + np.array([.12, -.09, .05])  # Stay away from success discontinuity.
    observation = observation_from_state(torch.tensor(q[None]), torch.tensor(tip[None]),
                                        torch.tensor(goal[None]), torch.tensor([[.0015]]))
    return SimpleNamespace(joints=torch.tensor(q[None]), action_scales=torch.tensor(env.action_scale[None]),
                           observations=observation, jacobian_valid=torch.tensor([False]))


def test_observation_and_joint_update_match_the_real_plant():
    cfg = short_config()
    env = make_env(cfg)
    try:
        obs, _ = env.reset(seed=701)
        q = torch.tensor(env.trig_obj.joints[None], requires_grad=True)
        reconstructed = observation_from_state(q, torch.tensor(obs["achieved_goal"][None]),
            torch.tensor(obs["desired_goal"][None]), torch.tensor(obs["observation"][None, 9:10]))
        for name in obs:
            np.testing.assert_array_equal(reconstructed[name].detach().numpy()[0], obs[name])
        projection = OriginalJacobianLoss(env.trig_obj.tube_lengths[0], env.n_substeps)
        for a in ([.02, -.03, .01, .04, -.02, .03], [-.04, .03, .01, -.01, .02, -.03]):
            action = torch.tensor([a], dtype=torch.float32, requires_grad=True)
            q = q + projection.projected_delta(action, q, torch.tensor(env.action_scale[None]), detach_source=False)
            env.step(np.array(a, dtype=np.float32))
            np.testing.assert_allclose(q.detach().numpy()[0], env.trig_obj.joints, rtol=0, atol=2e-15)
        assert q.requires_grad
    finally:
        env.close()


def test_nonlinear_ivp_backward_matches_tip_finite_differences():
    backend = IVPBackwardBackend(short_config())
    try:
        sample = real_sample(backend.env)
        q = sample.joints.requires_grad_(True)
        assert torch.autograd.gradcheck(lambda z: differentiable_tip(z, backend), (q,),
                                       eps=1e-6, atol=3e-5, rtol=3e-4)
    finally:
        backend.close()


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_complete_actor_gradient_through_fresh_multistep_ivp_matches_finite_difference(horizon):
    backend = IVPBackwardBackend(short_config())
    actor, target_actor, critic = FeedbackActor(), FeedbackActor(), SmoothCritic()
    try:
        sample = real_sample(backend.env)
        projection = OriginalJacobianLoss(backend.env.trig_obj.tube_lengths[0], backend.env.n_substeps)
        objective = ShortHorizonReturn(backend, projection, horizon, .95)

        def value():
            loss, metrics = objective.loss(actor, target_actor, critic, actor(sample.observations), sample, [0])
            assert metrics["valid_windows"] == 1 and metrics["terminal_windows"] == 0
            return loss

        gradient, = torch.autograd.grad(value(), actor.bias)
        assert gradient.abs().max() > 1e-4
        numeric = []
        epsilon = 2e-3  # Actual observations are float32; avoid roundoff-dominated differences.
        initial = actor.bias.detach().clone()
        for k in range(6):
            with torch.no_grad():
                actor.bias.copy_(initial); actor.bias[k] += epsilon
            plus = float(value().detach())
            with torch.no_grad():
                actor.bias.copy_(initial); actor.bias[k] -= epsilon
            minus = float(value().detach())
            numeric.append((plus - minus) / (2 * epsilon))
        np.testing.assert_allclose(gradient.numpy(), numeric, atol=3e-5, rtol=5e-3)
        assert all(p.grad is None for p in target_actor.parameters())
        assert all(p.grad is None for p in critic.parameters())
    finally:
        backend.close()


def test_terminal_value_keeps_state_gradient_but_freezes_weights():
    cfg = short_config()
    env = make_env(cfg)
    try:
        model = build_model(env, cfg)
        # Remove every state-to-action derivative: only the critic's direct
        # state path can make this test pass (q1_forward would cut that path).
        with torch.no_grad():
            for p in model.actor_target.parameters():
                p.zero_()
        observation = {k: torch.tensor(v[None], requires_grad=True) for k, v in env.reset(seed=701)[0].items()}
        flags = [p.requires_grad for m in (model.actor_target, model.critic_target) for p in m.parameters()]
        value = terminal_value(model.actor_target, model.critic_target, observation)
        value.backward()
        assert observation["achieved_goal"].grad.abs().sum() > 0
        assert observation["observation"].grad.abs().sum() > 0
        parameters = [p for m in (model.actor_target, model.critic_target) for p in m.parameters()]
        assert [p.requires_grad for p in parameters] == flags
        assert all(p.grad is None for p in parameters)
    finally:
        env.close()


class LinearBackend:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = 0, fail_on
        self.totals = dict.fromkeys(("attempted_windows", "valid_windows", "invalid_windows",
            "initial_terminal_windows", "terminal_windows", "rollout_steps", "scheduled_updates"), 0)
        self.totals.update(invalid_reasons={}, rollout_seconds=0.)

    def forward_and_jacobian(self, q):
        self.calls += 1
        if self.calls == self.fail_on:
            raise UnavailableWindow("test branch derivative unavailable")
        return q[3:], np.c_[np.zeros((3, 3)), np.eye(3)]


def linear_problem():
    q = torch.tensor([[-.1, -.05, -.01, 0., 0., 0.]], dtype=torch.float64)
    obs = observation_from_state(q, torch.zeros(1, 3), torch.tensor([[.02, 0., 0.]]), torch.tensor([[.001]]))
    return SimpleNamespace(joints=q, observations=obs,
        action_scales=torch.tensor([[.001, .001, .001, .02, .02, .02]]))


def test_success_terminates_window_without_bootstrapping(monkeypatch):
    backend, sample = LinearBackend(), linear_problem()
    action = torch.tensor([[0., 0., 0., 1., 0., 0.]], requires_grad=True)
    def forbidden(*args):
        pytest.fail("Terminal states must not bootstrap")
    monkeypatch.setattr("ctr_reach_envs.ivp.short_horizon.terminal_value", forbidden)
    obj = ShortHorizonReturn(backend, OriginalJacobianLoss([.3, .2, .1], n_substeps=1), 4, .95)
    loss, metrics = obj.loss(forbidden, None, None, action, sample, [0])
    assert loss == 0 and metrics["terminal_windows"] == 1 and backend.calls == 1
    loss.backward()
    torch.testing.assert_close(action.grad, torch.zeros_like(action))


def test_invalid_later_derivative_discards_entire_window():
    backend, sample = LinearBackend(fail_on=2), linear_problem()
    action = torch.tensor([[0., 0., 0., -.1, 0., 0.]], requires_grad=True)
    obj = ShortHorizonReturn(backend, OriginalJacobianLoss([.3, .2, .1], n_substeps=1), 2, .95)
    loss, metrics = obj.loss(lambda obs: action, FeedbackActor(), SmoothCritic(), action, sample, [0])
    loss.backward()
    assert metrics["invalid_windows"] == 1 and metrics["valid_windows"] == 0
    assert backend.totals["invalid_reasons"] == {"test branch derivative unavailable": 1}
    assert loss == 0 and torch.count_nonzero(action.grad) == 0


def test_already_satisfied_her_source_does_not_roll_out():
    backend, sample = LinearBackend(), linear_problem()
    sample.observations["desired_goal"] = sample.observations["achieved_goal"].clone()
    action = torch.zeros(1, 6, requires_grad=True)
    obj = ShortHorizonReturn(backend, OriginalJacobianLoss([.3, .2, .1]), 2, .95)
    loss, metrics = obj.loss(None, None, None, action, sample, [0])
    assert metrics["initial_terminal_windows"] == 1 and backend.calls == 0
    loss.backward()
    assert torch.count_nonzero(action.grad) == 0


@pytest.mark.parametrize("field,value", [("integration", "sum"), ("objective", "mechanics_only"),
    ("diagnostics", True), ("short_horizon_steps", 9), ("short_horizon_every", 0),
    ("short_horizon_start_steps", -1)])
def test_incompatible_configuration_is_rejected(field, value):
    cfg = short_config()
    cfg["physics"][field] = value
    with pytest.raises(ValueError):
        validate_short_horizon(cfg)


def test_public_cli_model_budget_and_checkpoint_resume(tmp_path, capsys):
    torch.set_num_threads(1)
    cfg = short_config()
    source, output = tmp_path / "config.json", tmp_path / "run"
    source.write_text(json.dumps(cfg))
    guided_main(["--profile", "original", "--config", str(source), "--seed", "10",
        "--physics-loss", "short_horizon", "--physics-weight", ".1", "--physics-final-weight", ".1",
        "--short-horizon-steps", "2", "--short-horizon-batch-size", "2", "--short-horizon-every", "2",
        "--short-horizon-start-steps", "0", "--total-timesteps", "64", "--checkpoint-freq", "0",
        "--output-dir", str(output)])
    saved = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] and summary["all_parameters_finite"]
    assert saved["environment_fingerprint"] == cfg["environment_fingerprint"]
    costs = summary["short_horizon_costs"]
    assert costs["scheduled_updates"] == summary["gradient_updates"] // 2
    assert costs["valid_windows"] > 0
    assert costs["sensitivity_calls"] <= costs["scheduled_updates"] * 2 * 2
    assert costs["forward_calls"] == costs["sensitivity_calls"] + 1
    assert summary["costs"]["sensitivity_calls"] == 0  # No unused source Jacobians.
    assert summary["last_physics_metrics"]["weighted_aux_norm_ratio_after"] <= .1 + 1e-12
    assert summary["last_physics_metrics"]["short_horizon_active_weighted_aux_norm_ratio_after"] <= .1 + 1e-12
    env = make_env(saved)
    try:
        loaded = OriginalJacobianDDPG.load(output / "final_model.zip", env=env, device="cpu")
        assert loaded.physics_loss_kind == "short_horizon" and loaded._short_horizon is None
        assert loaded.short_horizon_totals == costs
        # Inference through the evaluator's generic DDPG loader stays unchanged.
        evaluator = DDPG.load(output / "final_model.zip", env=env, device="cpu")
        obs, _ = env.reset(seed=920000)
        np.testing.assert_array_equal(loaded.predict(obs)[0], evaluator.predict(obs)[0])
        calls_before = loaded.short_horizon_totals["forward_calls"]
        loaded.learn(16, reset_num_timesteps=False)
        assert loaded.short_horizon_totals["forward_calls"] > calls_before
        loaded.close_short_horizon()
    finally:
        env.close()
    capsys.readouterr()


def test_zero_weight_short_horizon_never_constructs_a_model_backend():
    cfg = short_config()
    cfg["physics"].update(weight=0., final_weight=0.)
    env = make_env(cfg)
    try:
        model = build_model(env, cfg, force_guided_class=True)
        model.learn(16)
        assert model._short_horizon is None and model.short_horizon_totals == {}
    finally:
        env.close()


def test_critic_warmup_trains_without_model_rollouts():
    cfg = short_config()
    cfg["physics"]["short_horizon_start_steps"] = 10000
    env = make_env(cfg)
    try:
        model = build_model(env, cfg)
        model.learn(16)
        assert model._n_updates > 0 and model._short_horizon is None
        assert model.short_horizon_totals == {} and model.physics_update_count == 0
    finally:
        env.close()
