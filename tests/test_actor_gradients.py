"""Gradient conflict, Adam momentum, finite steps, rollback and integration."""
from copy import deepcopy

import pytest
import torch

from ctr_reach_envs.mechanics.actor_gradients import combine_gradients, actor_step


@pytest.mark.parametrize("auxiliary,expected", [([1., 1.], [1., 1.]), ([-1., 1.], [0., 1.])])
def test_projection_preserves_support_and_removes_only_conflict(auxiliary, expected):
    r = torch.tensor([1., 0.], dtype=torch.float64)
    direction, m = combine_gradients(r, torch.tensor(auxiliary), .1)
    torch.testing.assert_close(direction, r+.1*torch.tensor(expected, dtype=r.dtype))
    assert m["projected_gradient_dot"] >= 0
    assert r.dot(direction) >= r.dot(r)


def test_aligned_auxiliary_is_capped_and_zero_rl_gradient_disables_it():
    r = torch.tensor([1., 0.])
    direction, m = combine_gradients(r, torch.tensor([100., 100.]), .1, max_aux_ratio=.5)
    assert m["weighted_aux_norm_ratio_before"] > 1
    assert m["weighted_aux_norm_ratio_after"] == pytest.approx(.5)
    assert torch.linalg.norm(direction-r) == pytest.approx(.5)
    direction, _ = combine_gradients(torch.zeros(2), torch.ones(2), .1)
    assert torch.count_nonzero(direction) == 0


def test_positive_alignment_can_overshoot_and_backtracking_checks_finite_loss():
    p = torch.nn.Parameter(torch.tensor([.1], dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=1.)
    closure = lambda: .5*p.square().sum()
    r = p.detach().clone()
    direction, _ = combine_gradients(r, r, 1.)
    m = actor_step([p], optimizer, r, direction, closure)
    assert m["actor_step_rejected_trials"] > 0
    assert 0 < m["actor_step_scale"] < 1
    assert m["actor_step_first_order"] < 0 and m["rl_surrogate_change"] <= 0
    assert optimizer.state[p]["step"] == 1


def test_adam_preconditioning_can_reverse_a_compatible_direction():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=.01, betas=(0., .999))
    optimizer.state[p] = dict(step=torch.tensor(10.), exp_avg=torch.zeros_like(p),
                              exp_avg_sq=torch.tensor([.01, 100.], dtype=p.dtype))
    r, j = torch.tensor([1., 1.], dtype=p.dtype), torch.tensor([-2., 2.], dtype=p.dtype)
    direction, m = combine_gradients(r, j, 1., max_aux_ratio=10.)
    assert m["projected_gradient_dot"] == 0 and r.dot(direction) > 0
    result = actor_step([p], optimizer, r, direction, lambda: p.sum())
    assert result["actor_step_rl_only"] == 1 and result["actor_step_rejected_trials"] > 0
    assert result["rl_surrogate_change"] < 0
    assert optimizer.state[p]["step"] == 11  # Rejected trial did not advance Adam.


def test_momentum_rejection_restores_parameters_and_all_optimizer_state():
    p = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=.1)
    optimizer.state[p] = dict(step=torch.tensor(5.), exp_avg=torch.tensor([-10.], dtype=p.dtype),
                              exp_avg_sq=torch.tensor([1.], dtype=p.dtype))
    before = deepcopy(optimizer.state_dict())
    result = actor_step([p], optimizer, torch.ones(1), torch.ones(1), lambda: p.sum())
    assert result["actor_step_skipped"] == 1 and result["rl_surrogate_change"] == 0
    assert p.item() == 0
    after = optimizer.state_dict()
    assert before["param_groups"] == after["param_groups"]
    for key in before["state"][0]:
        torch.testing.assert_close(before["state"][0][key], after["state"][0][key], rtol=0, atol=0)


def test_loss_exception_also_rolls_back_adam():
    p = torch.nn.Parameter(torch.tensor([0.]))
    optimizer = torch.optim.Adam([p])
    def closure():
        if p.item() != 0:
            raise RuntimeError("trial failure")
        return p.sum()
    with pytest.raises(RuntimeError, match="trial failure"):
        actor_step([p], optimizer, torch.ones(1), torch.ones(1), closure)
    assert p.item() == 0 and not optimizer.state


@pytest.mark.parametrize("integration", ["sum", "rl_priority"])
def test_training_reports_parameter_gradients_and_restores_mode(tmp_path, integration):
    from test_reach_hold import straight_env
    from test_mechanics_jacobian import model_args
    from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG
    torch.set_num_threads(1)
    env = straight_env()
    args = model_args(env)
    args["replay_buffer_kwargs"]["action_semantics"] = "proposal"
    model = JacobianDDPG(**args, physics_lengths=[.2], physics_integration=integration)
    model.learn(12)
    metrics = model.last_physics_metrics
    assert metrics["actor_step_checked"] == float(integration == "rl_priority")
    assert metrics["rl_gradient_norm"] > 0
    if integration == "rl_priority":
        assert metrics["actor_step_first_order"] <= 0
        assert metrics["rl_surrogate_change"] <= 0
        assert metrics["weighted_aux_norm_ratio_after"] <= 1.+1e-12
    path = tmp_path/"model.zip"
    model.save(path)
    evaluation = straight_env(compute_jacobian=False)
    loaded = JacobianDDPG.load(path, env=evaluation)
    assert loaded.physics_integration == integration
    assert loaded.actor_max_backtracks == 6
    obs, _ = evaluation.reset(seed=7101)
    torch.testing.assert_close(torch.tensor(model.predict(obs, deterministic=True)[0]),
                               torch.tensor(loaded.predict(obs, deterministic=True)[0]), rtol=0, atol=0)
    env.close(); evaluation.close()
