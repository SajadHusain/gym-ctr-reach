"""Behavior, gradients, and the actual CLI/checkpoint boundary for progress loss."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ctr_reach_envs.ivp.config import make_env
from ctr_reach_envs.ivp.rl import OriginalJacobianDDPG, OriginalJacobianLoss
from ctr_reach_envs.training.cli import guided_main
from ctr_reach_envs.training.original import parse_args
from test_original_ivp import configuration


def local_problem(goal=(.01, 0., 0.), count=1):
    """An exactly linear synthetic tip map driven by the rotation coordinates."""
    sample = SimpleNamespace(
        observations={"achieved_goal": torch.zeros(count, 3, dtype=torch.float64),
                      "desired_goal": torch.tensor([goal] * count, dtype=torch.float64)},
        jacobian=torch.cat((torch.zeros(3, 3), torch.eye(3)), dim=1).double().repeat(count, 1, 1),
        joints=torch.tensor([[-.1, -.05, -.01, 0., 0., 0.]] * count, dtype=torch.float64),
        action_scales=torch.tensor([[.001, .001, .001, .02, .02, .02]] * count, dtype=torch.float64),
        jacobian_valid=torch.ones(count, dtype=torch.bool))
    return sample


def objective(kind="progress", **kwargs):
    return OriginalJacobianLoss([.3, .2, .1], n_substeps=1, loss_kind=kind, **kwargs)


def action(x=0., y=0., count=1):
    return torch.tensor([[0., 0., 0., x, y, 0.]] * count, dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize("x", [.2, .4, .6])
def test_sufficient_progress_is_unpenalized_even_beyond_tracking_target(x):
    sample, a = local_problem(), action(x)
    loss = objective()(a, sample)
    loss.backward()
    assert loss.item() == 0.
    torch.testing.assert_close(a.grad, torch.zeros_like(a))
    assert objective("tracking")(a, sample).item() > 0.


@pytest.mark.parametrize("x,y,expected", [(0., 0., 1.), (-.1, 0., 4.), (1., 0., 1.)])
def test_stalling_away_motion_and_large_overshoot_are_penalized(x, y, expected):
    assert objective()(action(x, y), local_problem()).item() == pytest.approx(expected)


def test_sideways_motion_is_not_mistaken_for_goal_progress():
    sample = local_problem()
    assert objective()(action(0., .2), sample) > objective()(action(), sample)


def test_active_progress_gradient_matches_finite_differences():
    sample, a = local_problem(), action(.02, .07)
    assert torch.autograd.gradcheck(lambda value: objective()(value, sample), (a,),
                                    eps=1e-6, atol=1e-7, rtol=1e-5)
    before = objective()(a, sample)
    gradient, = torch.autograd.grad(before, a)
    # In this exactly linear model, a small descent step must improve actual
    # Euclidean goal progress, independently of the hinge's scalar value.
    candidate = a.detach() - 1e-5 * gradient
    tip_before = .02 * a.detach()[:, 3:]
    tip_after = .02 * candidate[:, 3:]
    goal = sample.observations["desired_goal"]
    assert torch.linalg.vector_norm(goal - tip_after) < torch.linalg.vector_norm(goal - tip_before)


def test_her_goal_relabeling_reverses_gradient_without_differentiating_context():
    sample, a = local_problem(), action()
    for tensor in (sample.jacobian, sample.joints, sample.action_scales,
                   *sample.observations.values()):
        tensor.requires_grad_(True)
    objective()(a, sample).backward()
    positive = a.grad.clone()
    a.grad = None
    sample.observations["desired_goal"] = -sample.observations["desired_goal"].detach()
    objective()(a, sample).backward()
    assert positive[0, 3] < 0. < a.grad[0, 3]
    torch.testing.assert_close(a.grad, -positive)
    for tensor in (sample.jacobian, sample.joints, sample.action_scales,
                   *sample.observations.values()):
        assert tensor.grad is None


def test_invalid_samples_are_excluded_from_value_and_gradient():
    sample, a = local_problem(count=2), action(count=2)
    sample.jacobian_valid[1] = False
    with torch.no_grad():
        a[1, 3] = -1.
    loss = objective()(a, sample)
    loss.backward()
    assert loss.item() == pytest.approx(1.)  # Normalize by valid samples only.
    torch.testing.assert_close(a.grad[1], torch.zeros(6, dtype=torch.float64))
    sample.jacobian_valid[:] = False
    a.grad = None
    loss = objective()(a, sample)
    loss.backward()
    assert loss.item() == 0.
    torch.testing.assert_close(a.grad, torch.zeros_like(a))


@pytest.mark.parametrize("x", [0., .1])
def test_zero_goal_error_has_finite_loss_and_gradient(x):
    a = action(x)
    loss = objective()(a, local_problem(goal=(0., 0., 0.)))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
    assert (loss.item() == 0.) == (x == 0.)


def test_progress_requirement_shrinks_near_goal_and_never_exceeds_error():
    sample = local_problem(goal=(.001, 0., 0.))
    assert objective()(action(.025), sample).item() == pytest.approx(0., abs=1e-20)
    assert objective(gain=2.)(action(.05), sample).item() == pytest.approx(0., abs=1e-20)


def test_default_tracking_loss_is_backward_compatible():
    default = OriginalJacobianLoss([.3, .2, .1], n_substeps=1)
    assert default.loss_kind == "tracking"
    # Original objective: (4 mm prediction - 2 mm target)^2 / (2 mm)^2.
    assert default(action(.2), local_problem()).item() == pytest.approx(1.)
    assert parse_args(["--output-dir", "unused"]).physics_loss == "tracking"
    with pytest.raises(ValueError, match="Unknown"):
        objective("typo")


def test_public_training_cli_saves_and_reloads_progress_mode(tmp_path, capsys):
    output = tmp_path/"run"
    previous = configuration()
    source_path = tmp_path/"previous_config.json"
    source_path.write_text(json.dumps(previous), encoding="utf-8")
    source_bytes = source_path.read_bytes()
    guided_main(["--profile", "original", "--config", str(source_path), "--seed", "10",
        "--physics-loss", "progress", "--physics-weight", "0.1", "--physics-final-weight", "0.1",
        "--physics-integration", "rl_priority", "--physics-max-aux-ratio", "0.1",
        "--physics-diagnostics", "--total-timesteps", "16", "--checkpoint-freq", "0",
        "--output-dir", str(output)])
    cfg = json.loads((output/"config.json").read_text())
    summary = json.loads((output/"summary.json").read_text())
    assert cfg["physics"]["loss_kind"] == "progress"
    assert cfg["seed"] == 10
    assert cfg["spec"] == dict(previous["spec"], total_timesteps=16)
    assert cfg["environment_fingerprint"] == previous["environment_fingerprint"]
    assert source_path.read_bytes() == source_bytes
    assert summary["complete"] and summary["physics_updates"] > 0
    assert summary["last_physics_metrics"]["physics_loss_kind"] == "progress"
    assert np.isfinite(summary["last_physics_metrics"]["jacobian_loss_after"])
    assert summary["last_physics_metrics"]["weighted_aux_norm_ratio_after"] <= .1 + 1e-12
    env = make_env(cfg, compute_jacobian=True)
    try:
        loaded = OriginalJacobianDDPG.load(output/"final_model.zip", env=env, device="cpu")
        assert loaded.physics_loss_kind == "progress"
        assert loaded._physics_loss is None
        assert loaded._make_physics_loss().loss_kind == "progress"
        assert loaded.original_ivp_config["physics"]["loss_kind"] == "progress"
    finally:
        env.close()
    capsys.readouterr()
