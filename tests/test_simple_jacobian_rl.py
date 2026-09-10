"""Joint-only plant, unblocked event crossings, and masked-Jacobian HER updates."""
from dataclasses import replace
import warnings

import numpy as np
import pytest
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

from ctr_reach_envs.mechanics.geometry import TubeParameters
from ctr_reach_envs.mechanics.solver import EquilibriumError
from ctr_reach_envs.mechanics.simple_rl_env import JointConstrainedReachEnv
from ctr_reach_envs.mechanics.rl_jacobian import (
    PhysicsSamples, ProjectedJacobianLoss, JacobianHerReplayBuffer, JacobianDDPG,
)
from ctr_reach_envs.mechanics.rl_replay import ExecutedActionHerReplayBuffer
from ctr_reach_envs.mechanics.rl_policy import EquilibriumStateExtractor
from ctr_reach_envs.paper_policy import PaperMlpPolicy


def make_env(jacobian=True, horizon=4):
    tube = TubeParameters(.2, 0., .001, .002, 50e9, 23e9, 0.)
    return JointConstrainedReachEnv(tubes=[tube], compute_jacobian=jacobian, max_episode_steps=horizon)


def options(goal=.19):
    return {"joints": [-.03, 0.], "goal": [0., 0., goal]}


def batch(obs):
    return {k: v[None].copy() for k, v in obs.items()}


def test_api_and_finite_observations_without_torsion_features():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = make_env()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, skip_render_check=True)
        sb3_check(env)
    obs, _ = env.reset(seed=3, options=options())
    assert obs["observation"].shape == (4,)
    assert env.observation_space.contains(obs)
    assert not hasattr(env, "tracker")
    assert not hasattr(env, "tracking_options")


def test_reset_and_step_never_use_tracking_stability_or_control(monkeypatch):
    from ctr_reach_envs.mechanics import tracking, stability, control
    def forbidden(*args, **kwargs):
        pytest.fail("Joint-only operation invoked branch/controller logic")
    monkeypatch.setattr(tracking.BranchTracker, "__init__", forbidden)
    monkeypatch.setattr(stability, "elastic_stability", forbidden)
    monkeypatch.setattr(control.GoalController, "step", forbidden)
    env = JointConstrainedReachEnv()
    obs, info = env.reset(seed=7006)
    assert info["reset_costs"]["equilibrium_calls"] == 2
    assert info["reset_costs"]["sensitivity_calls"] == 0
    env.step(np.full(6, .1))
    assert env.costs["stability_calls"] == 0
    assert env.costs["equilibrium_calls"] == 3
    assert env.costs["sensitivity_calls"] == 1


def test_goal_and_auxiliary_switch_do_not_change_the_plant():
    guided, base = make_env(True), make_env(False)
    guided.reset(seed=4, options=options(.19))
    base.reset(seed=4, options=options(.15))
    a = guided.step([.4, -.3]); b = base.step([.4, -.3])
    np.testing.assert_array_equal(a[0]["achieved_goal"], b[0]["achieved_goal"])
    np.testing.assert_array_equal(a[0]["observation"], b[0]["observation"])
    assert base.costs["sensitivity_calls"] == 0
    assert a[4]["physics"]["jacobian_valid"]
    assert not b[4]["physics"]["jacobian_valid"]


def test_geometry_event_crossing_is_executed():
    tubes = [TubeParameters(.3, .15, .001, .002, 50e9, 23e9, 2.),
             TubeParameters(.2, .1, .0021, .003, 50e9, 23e9, 1.)]
    env = JointConstrainedReachEnv(tubes=tubes)
    q = np.array([-.1496, -.07, 0., 0.])
    env.reset(options={"joints": q, "goal": [0., 0., .2]})
    before_curve_start = .3 + q[0] - .15
    _, _, _, _, info = env.step([-1., 0., 0., 0.])
    assert before_curve_start > 0 > .3+info["q_after"][0]-.15
    np.testing.assert_allclose(info["applied_delta"], [-.001, 0., 0., 0.], atol=1e-15)
    assert info["reason"] == "accepted"


def test_coincident_event_masks_jacobian_but_does_not_block_motion():
    tube = TubeParameters(.2, .1, .001, .002, 50e9, 23e9, 2.)
    env = JointConstrainedReachEnv(tubes=[tube])
    env.reset(options={"joints": [-.1, 0.], "goal": [0., 0., .15]})
    _, _, _, _, info = env.step([-.25, .1])
    assert not info["physics"]["jacobian_valid"]
    assert np.count_nonzero(info["physics"]["jacobian"]) == 0
    assert env.jacobian_failures == 1
    assert info["reason"] == "accepted"
    np.testing.assert_allclose(info["applied_delta"], [-.00025, .005], atol=1e-15)
    assert env.costs["stability_calls"] == 0


def test_plant_and_actor_use_identical_coupled_joint_projection():
    env = JointConstrainedReachEnv(compute_jacobian=False)
    q = np.array([-.0004, -.0003, -.0002, 0., 0., 0.])
    a = np.array([1., -1., .5, 1., -.5, .4])
    env.reset(options={"joints": q, "goal": [.1, 0., .1]})
    expected = ProjectedJacobianLoss(env.solver.lengths).projected_delta(
        torch.tensor(a[None]), torch.tensor(q[None]), torch.tensor(env.action_scales[None])).numpy()[0]
    _, _, _, _, info = env.step(a)
    np.testing.assert_allclose(info["applied_delta"], expected, atol=1e-13, rtol=0)
    assert env.solver.constraints.is_feasible(info["q_after"])
    assert np.max(np.abs(info["executed_action"])) <= 1


def test_failed_equilibrium_does_not_create_a_replay_transition(monkeypatch):
    env = make_env(False)
    env.reset(options=options())
    before = env.equilibrium.joints.copy()
    def fail(*args, **kwargs):
        raise EquilibriumError("synthetic failed solve")
    monkeypatch.setattr(env.solver, "solve", fail)
    with pytest.raises(EquilibriumError, match="synthetic"):
        env.step([1., 0.])
    np.testing.assert_array_equal(env.equilibrium.joints, before)
    assert env.transitions == 0
    assert env._finished
    assert env.costs["failed_calls"] == 1


def test_stateless_equilibrium_is_independent_of_previous_torsion():
    env = make_env(False)
    env.reset(options=options())
    env.equilibrium = replace(env.equilibrium, base_torsional_strain=np.array([999.]))
    env.step([.4, .3])
    fresh = env.solver.solve(env.equilibrium.joints)
    np.testing.assert_array_equal(env.equilibrium.tip, fresh.tip)


def test_invalid_jacobian_rows_have_zero_auxiliary_gradient():
    obs = {"achieved_goal": torch.tensor([[0.,0.,.17], [0.,0.,.17]]),
           "desired_goal": torch.tensor([[0.,0.,.18], [0.,0.,.18]])}
    sample = PhysicsSamples(obs, None, None, None, None,
                           torch.tensor([[[0.,0.],[0.,0.],[1.,0.]]]*2),
                           torch.tensor([[-.03,0.]]*2), torch.tensor([[.001,.05]]*2),
                           jacobian_valid=torch.tensor([[True],[False]]))
    action = torch.zeros(2,2,requires_grad=True)
    fn = ProjectedJacobianLoss([.2])
    value = fn(action,sample); value.backward()
    assert action.grad[0,0] < 0
    assert torch.count_nonzero(action.grad[1]) == 0
    action.grad.zero_()
    sample.jacobian_valid[:] = False
    value = fn(action,sample); value.backward()
    assert value == 0 and torch.count_nonzero(action.grad) == 0


def test_her_keeps_raw_proposals_and_the_matching_jacobian_mask(monkeypatch):
    env = make_env(horizon=4)
    vec = DummyVecEnv([make_env])
    replay = JacobianHerReplayBuffer(20, vec.observation_space, vec.action_space, env=vec,
                                    copy_info_dict=True, device="cpu", action_semantics="proposal")
    obs, _ = env.reset(options={"joints": [-.0002,0.], "goal": [0.,0.,.1]})
    for i in range(4):
        action = np.array([.4, .1])
        nxt, reward, term, trunc, info = env.step(action)
        info["physics"]["jacobian_valid"] = bool(i % 2)
        info["physics"]["jacobian"][0,0] = i
        info["TimeLimit.truncated"] = trunc
        replay.add(batch(obs), batch(nxt), action[None], np.array([reward]), np.array([term or trunc]), [info])
        obs = nxt
    assert not np.array_equal(replay.actions[0,0],replay.infos[0,0]["executed_action"])
    np.testing.assert_allclose(replay.actions[:4,0], np.tile([.4,.1],(4,1)))
    monkeypatch.setattr(env.solver, "solve", lambda *a,**k: pytest.fail("HER sampling invoked the solver"))
    sample = replay.sample(32)
    np.testing.assert_array_equal(sample.jacobian_valid.cpu().numpy().ravel(), sample.jacobian[:,0,0].cpu().numpy()%2 == 1)
    assert set(sample.observations) == {"observation","achieved_goal","desired_goal"}
    vec.close()


def model_args(env):
    return dict(policy=PaperMlpPolicy, env=env, seed=15, learning_starts=4, buffer_size=100,
                batch_size=8, train_freq=1, gradient_steps=1, device="cpu", learning_rate=.0005,
                policy_kwargs={"net_arch":[16,16], "features_extractor_class":EquilibriumStateExtractor,
                               "features_extractor_kwargs":{"length_scale":.2}},
                replay_buffer_class=JacobianHerReplayBuffer,
                replay_buffer_kwargs={"copy_info_dict":True, "action_semantics":"proposal"})


def test_zero_weight_matches_ddpg_and_guided_update_saves_loads(tmp_path):
    torch.set_num_threads(1)
    # Construct each immediately before learning to reset the shared RNG.
    plain = DDPG(**model_args(make_env())); plain.learn(12)
    zero = JacobianDDPG(**model_args(make_env()), physics_lengths=[.2],physics_weight=0,physics_final_weight=0)
    zero.learn(12)
    for key, value in plain.policy.state_dict().items():
        torch.testing.assert_close(value, zero.policy.state_dict()[key], rtol=0, atol=0)
    model = JacobianDDPG(**model_args(make_env()), physics_lengths=[.2],physics_weight=.1,physics_final_weight=.1)
    model.learn(12)
    assert model.physics_update_count == 8
    assert model.last_physics_metrics["jacobian_valid_fraction"] == 1
    obs, _ = model.get_env().envs[0].reset(seed=19)
    action = model.predict(obs, deterministic=True)[0]
    path = tmp_path/"model.zip"; model.save(path)
    loaded = JacobianDDPG.load(path, env=make_env(False), device="cpu")
    np.testing.assert_array_equal(action, loaded.predict(obs,deterministic=True)[0])
    for m in (plain, zero, model, loaded):
        m.get_env().close()
