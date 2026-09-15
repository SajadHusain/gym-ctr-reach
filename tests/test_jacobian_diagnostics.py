from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_original_ivp import configuration
from ctr_reach_envs.ivp.config import make_env
from ctr_reach_envs.ivp.diagnostics import auxiliary_probe, run_diagnostics, main
from ctr_reach_envs.training.original import build_model, parse_args


def test_diagnostics_do_not_change_training_parameters_replay_or_rng():
    torch.set_num_threads(1)
    def train(enabled):
        cfg = configuration()
        cfg["physics"]["diagnostics"] = enabled
        env = make_env(cfg, compute_jacobian=True)
        try:
            model = build_model(env, cfg)
            model.learn(32)
            return ({k:v.clone() for k,v in model.policy.state_dict().items()},
                    model.replay_buffer.actions.copy(), model.last_physics_metrics,
                    np.random.get_state(), torch.random.get_rng_state())
        finally:
            env.close()
    before, after = train(False), train(True)
    for k in before[0]:
        torch.testing.assert_close(before[0][k], after[0][k], atol=0, rtol=0)
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(before[3][1], after[3][1])
    assert before[3][2:] == after[3][2:]
    torch.testing.assert_close(before[4], after[4], atol=0, rtol=0)
    m = after[2]
    assert np.isfinite(m["jacobian_loss_after"])
    assert m["jacobian_loss_change"] == pytest.approx(m["jacobian_loss_after"]-m["jacobian_loss_before"])
    assert m["nominal_weighted_loss"] == m["total_actor_loss"]
    assert "jacobian_loss_after" not in before[2]


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def set_training_mode(self, mode):
        self.train(mode)

    def forward(self, observations):
        return self.weight*observations["x"]


def test_auxiliary_probe_changes_only_copy_and_uses_fit_goal():
    actor = ToyActor()
    sample = SimpleNamespace(observations={"x":torch.ones(4)}, jacobian_valid=torch.ones(4, dtype=torch.bool))
    loss = lambda a, s: ((a-1.)**2).mean()
    candidate, report = auxiliary_probe(actor, loss, sample, relative_step=.1)
    assert report["accepted"]
    assert report["fit_loss_after"] < report["fit_loss_before"]
    assert actor.weight.item() == 0.
    assert candidate.weight.item() > 0.
    # Fitting toward +1 can worsen a different goal: acceptance is NOT selected
    # by held-out performance or claimed to guarantee generalization.
    assert float(((candidate(sample.observations)+1.)**2).mean().detach()) > 1.


def test_all_invalid_probe_is_explicit_noop():
    actor = ToyActor()
    sample = SimpleNamespace(observations={"x":torch.ones(2)}, jacobian_valid=torch.zeros(2, dtype=torch.bool))
    candidate, report = auxiliary_probe(actor, lambda a, s:a.sum()*0., sample)
    assert not report["accepted"] and report["valid_fit_samples"] == 0
    torch.testing.assert_close(actor.weight, candidate.weight)


def test_original_cli_diagnostics_are_opt_in():
    assert not parse_args(["--output-dir", "unused"]).physics_diagnostics
    assert parse_args(["--output-dir", "unused", "--physics-diagnostics"]).physics_diagnostics


@pytest.mark.parametrize("physics_mode", ["none", "jacobian_limits"])
def test_cli_loads_saved_her_checkpoint_and_preserves_files(tmp_path, capsys, physics_mode):
    """Exercise the ZIP-loading boundary missed by in-memory diagnostic tests."""
    import hashlib
    torch.set_num_threads(1)
    cfg = configuration()
    cfg["spec"]["physics_observation"]["mode"] = physics_mode
    env = make_env(cfg, compute_jacobian=True)
    model_path, config_path = tmp_path/"final_model.zip", tmp_path/"config.json"
    try:
        model = build_model(env, cfg)
        model.learn(16)
        model.save(model_path)
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
    finally:
        env.close()
    model_bytes, config_bytes = model_path.read_bytes(), config_path.read_bytes()
    output = tmp_path/"diagnostics"
    main([str(model_path), "--states", "1", "--rollout-steps", "0",
          "--action-fractions", "0.1", "1.0", "--output-dir", str(output)])
    report = json.loads((output/"summary.json").read_text())
    assert report["complete"] and report["checkpoint_timesteps"] == 16
    assert report["checkpoint_sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert report["prediction_rows"] > 0 and report["probe_rows"] > 0
    assert model_path.read_bytes() == model_bytes
    assert config_path.read_bytes() == config_bytes
    capsys.readouterr()


def test_offline_real_ivp_probe_preserves_model_and_records_failures(tmp_path):
    torch.set_num_threads(1)
    cfg = configuration()
    env = make_env(cfg)
    try:
        model = build_model(env, cfg)
        before = deepcopy(model.policy.state_dict())
        optimizer_before = deepcopy(model.actor.optimizer.state_dict())
        summary = run_diagnostics(model, cfg, tmp_path/"diagnostic", states=2,
                                  seed=910000, rollout_steps=0, action_fractions=(.1, 1.))
        assert summary["complete"] and summary["numerical_failures"] == 0
        assert summary["prediction_rows"] > 0 and summary["probe_rows"] > 0
        assert summary["prediction_by_action"]["zero:0"]["actual_progress_m"]["mean"] == pytest.approx(0, abs=1e-10)
        for result in summary["prediction_by_action"].values():
            assert result["projection_error_inf"]["p95"] < 1e-10
        for key in before:
            torch.testing.assert_close(before[key], model.policy.state_dict()[key], atol=0, rtol=0)
        assert optimizer_before == model.actor.optimizer.state_dict()
        assert model.num_timesteps == 0 and model.replay_buffer.size() == 0
        states = json.loads((tmp_path/"diagnostic"/"states.json").read_text())
        assert {s["fingerprint"] for s in states["fit"]}.isdisjoint(s["fingerprint"] for s in states["probe"])
        with pytest.raises(ValueError, match="empty"):
            run_diagnostics(model, cfg, tmp_path/"diagnostic", states=1)
        wrong = dict(cfg, profile="mechanics-comparison-v1")
        with pytest.raises(ValueError, match="original"):
            run_diagnostics(model, wrong, tmp_path/"wrong", states=1)
    finally:
        env.close()
