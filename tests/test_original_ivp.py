from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ctr_reach_envs.envs.ctr_reach_env import CtrReachEnv
from ctr_reach_envs.ivp.config import load_spec, resolve, make_env
from ctr_reach_envs.ivp.sensitivity import tip_sensitivity, field_and_derivative, IVPSensitivityError
from ctr_reach_envs.ivp.rl import OriginalJacobianLoss
from ctr_reach_envs.training.original import build_model


def configuration(mode="continuous", weight=.1, system=0):
    s, e, p = load_spec()
    s.update(system_idx=system, max_steps_per_episode=8, batch_size=8, buffer_size=128,
             hidden_layers=[16, 16, 16], total_timesteps=64)
    s["legacy_defaults"].update(rollout_steps=8, gradient_steps=2)
    physics = dict(weight=weight, final_weight=weight, anneal_steps=1000,
                   integration="rl_priority", max_aux_ratio=1., gain=.5, scale_m=.002,
                   max_tip_step_m=.002, objective="hybrid")
    return resolve(s, e, segment_mode=mode, seed=7101, physics=physics, provenance=p)


def test_legacy_transition_parity_and_no_jacobian_at_evaluation():
    config = configuration("legacy")
    a, b = CtrReachEnv(**config["environment"]), make_env(config)
    try:
        oa, ia = a.reset(seed=7101); ob, ib = b.reset(seed=7101)
        for k in oa: np.testing.assert_array_equal(oa[k], ob[k])
        rng = np.random.default_rng(77)
        for _ in range(30):
            action = rng.uniform(-1, 1, 6).astype(np.float32)
            ra, rb = a.step(action), b.step(action)
            for k in ra[0]: np.testing.assert_array_equal(ra[0][k], rb[0][k])
            assert ra[1:4] == rb[1:4]
            np.testing.assert_array_equal(a.trig_obj.joints, b.trig_obj.joints)
            if ra[2] or ra[3]: a.reset(); b.reset()
        assert b.costs["sensitivity_calls"] == 0
    finally: a.close(); b.close()


def test_analytical_field_derivative():
    rng = np.random.default_rng(123)
    y = rng.normal(size=18)
    properties = (np.array([4., 8., 9.]), np.array([.2, .4, .7]), np.array([.2, .3, 0.]), np.array([.3, .4, 0.]))
    f, A = field_and_derivative(y, *properties)
    h = 1e-6
    numeric = np.column_stack([(field_and_derivative(y + h * v, *properties)[0] -
                                field_and_derivative(y - h * v, *properties)[0]) / (2 * h) for v in np.eye(18)])
    np.testing.assert_allclose(A, numeric, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("system", range(4))
def test_variational_jacobian_matches_independent_tip_perturbations(system):
    env = make_env(configuration(system=system))
    env.reset(seed=701 + system)
    q = env.trig_obj.joints.copy()
    before = env.model.r.copy()
    result = tip_sensitivity(env.model, q)
    np.testing.assert_array_equal(before, env.model.r)
    expected = env.model.forward_kinematics(q, 0)
    np.testing.assert_allclose(result.tip, expected, atol=2e-8, rtol=0)
    numeric = []
    for i in range(6):
        h = 1e-6 if i < 3 else 1e-5
        delta = np.eye(6)[i] * h
        numeric.append((env.model.forward_kinematics(q + delta, 0) - env.model.forward_kinematics(q - delta, 0)) / (2 * h))
    np.testing.assert_allclose(result.jacobian, np.array(numeric).T, atol=2e-5, rtol=2e-5)
    env.close()


def test_legacy_cell_derivative_is_not_continuous_geometry_derivative():
    env = make_env(configuration("legacy"))
    q = np.array([-.201234567, -.121345678, -.032456789, .3, -.6, 1.1])
    result = tip_sensitivity(env.model, q)
    np.testing.assert_allclose(result.jacobian[:, :3], [[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]], atol=1e-12)
    env.model.integration_options = dict(method="DOP853", rtol=1e-10, atol=1e-12)
    for i in range(3):
        d = np.eye(6)[i] * 1e-9
        numeric = (env.model.forward_kinematics(q + d, 0) - env.model.forward_kinematics(q - d, 0)) / (2e-9)
        np.testing.assert_allclose(result.jacobian[:, i], numeric, atol=2e-6)
    env.close()


@pytest.mark.parametrize("constrained", [False, True])
def test_differentiable_actions_match_original_ordered_constraints(constrained):
    config = configuration()
    config["spec"]["constrain_alpha"] = constrained
    config = resolve(config["spec"], config["environment"], segment_mode="continuous", seed=7101,
                     physics=config["physics"], provenance="test")
    env = make_env(config)
    loss = OriginalJacobianLoss(env.trig_obj.tube_lengths[0], env.n_substeps, constrained)
    rng = np.random.default_rng(93)
    for _ in range(30):
        q = env.trig_obj.sample_goal(0)
        q[3:] *= 1.5
        env.trig_obj.set_joints(q, 0)
        source = env.trig_obj.joints.copy()
        action = rng.uniform(-1, 1, 6).astype(np.float32)
        for _ in range(env.n_substeps): env.trig_obj.set_action(action.astype(float) * env.action_scale, 0)
        dq = loss.projected_delta(torch.tensor(action[None]), torch.tensor(source[None]), torch.tensor(env.action_scale[None]))
        np.testing.assert_allclose(dq.numpy()[0], env.trig_obj.joints - source, atol=2e-15, rtol=0)
    env.close()


def test_loss_action_gradient_and_invalid_mask():
    env = make_env(configuration())
    env.reset(seed=888)
    q = env.trig_obj.joints.copy()
    result = tip_sensitivity(env.model, q)
    loss = OriginalJacobianLoss(env.trig_obj.tube_lengths[0])
    sample = SimpleNamespace(joints=torch.tensor(q[None]), action_scales=torch.tensor(env.action_scale[None]),
        jacobian=torch.tensor(result.jacobian[None], requires_grad=True), jacobian_valid=torch.tensor([True]),
        observations={"desired_goal": torch.tensor([[.02, .03, .2]]), "achieved_goal": torch.tensor(result.tip[None])})
    action = torch.full((1, 6), .123, dtype=torch.float64, requires_grad=True)
    grad = torch.autograd.grad(loss(action, sample), action)[0].detach().numpy()[0]
    numeric = []
    for i in range(6):
        d = torch.eye(6, dtype=torch.float64)[i:i+1] * 1e-6
        numeric.append(float((loss(action+d, sample)-loss(action-d, sample)).detach())/(2e-6))
    np.testing.assert_allclose(grad, numeric, atol=1e-7, rtol=1e-6)
    assert sample.jacobian.grad is None
    sample.jacobian_valid[:] = False
    assert torch.autograd.grad(loss(action, sample), action)[0].abs().max() == 0
    env.close()


def test_zero_weight_guided_update_matches_paper_ddpg():
    torch.set_num_threads(1)
    cfg = configuration(weight=0.)
    def run(force):
        env = make_env(cfg)
        model = build_model(env, cfg, force_guided_class=force)
        model.learn(64)
        state = {k: v.clone() for k, v in model.policy.state_dict().items()}
        actions = model.replay_buffer.actions.copy()
        updates = model._n_updates
        env.close()
        return state, actions, updates
    a, b = run(False), run(True)
    assert a[2] == b[2] and a[2] > 0
    for key in a[0]: torch.testing.assert_close(a[0][key], b[0][key], rtol=0, atol=0)
    np.testing.assert_array_equal(a[1], b[1])


def test_saved_evaluation_space_and_seed_match():
    config = configuration()
    a, b = make_env(config), make_env(deepcopy(config), evaluation=True)
    assert a.observation_space == b.observation_space
    a.reset(seed=810000); b.reset(seed=810000)
    np.testing.assert_array_equal(a.trig_obj.joints, b.trig_obj.joints)
    np.testing.assert_array_equal(a.desired_goal, b.desired_goal)
    assert b.get_goal_tolerance() == .001
    a.close(); b.close()


def test_reported_shooting_failure_and_neighbors_use_no_shooting():
    config = configuration()
    env = make_env(config)
    q = np.array([-.32189071014609677, -.23722705892699403, -.079227058926994,
                   3.1434153506364657, .894513135415457, -1.7203025566390389])
    goal = env.model.forward_kinematics(q, 0)
    assert np.all(np.isfinite(goal))
    for delta in np.r_[np.eye(6), -np.eye(6)]:
        env.reset(seed=12, options={"initial_joints": q, "goal": goal + .01})
        _, _, _, _, info = env.step(delta)
        assert not info["solver_failure"]
    env.close()


def test_nonsmooth_sensitivity_masks_only_auxiliary_and_never_changes_action():
    cfg = configuration()
    guided, ordinary = make_env(cfg, compute_jacobian=True), make_env(cfg)
    q = np.array([-.25, -.15, -.15, .1, .3, .4])
    options = dict(initial_joints=q, goal=[.02, .03, .15])
    guided.reset(seed=10, options=options); ordinary.reset(seed=10, options=options)
    # beta_2 == beta_3 is a moving segmentation boundary.
    r = guided.step(np.ones(6) * .1)
    s = ordinary.step(np.ones(6) * .1)
    assert r[4]["physics"]["jacobian_valid"] is False
    assert not r[4]["solver_failure"]
    np.testing.assert_array_equal(r[4]["q_after"], s[4]["q_after"])
    for key in r[0]: np.testing.assert_array_equal(r[0][key], s[0][key])
    guided.close(); ordinary.close()


def test_her_relabels_goal_and_termination_but_retains_source_sensitivity():
    cfg = configuration()
    env = make_env(cfg, compute_jacobian=True)
    model = build_model(env, cfg)
    model.learn(16)
    replay = model.replay_buffer
    indices, env_indices = np.array([0]), np.array([0])
    old_goal = replay.observations["desired_goal"][0, 0].copy()
    new_goal = replay.next_observations["achieved_goal"][0:1, 0].copy()
    replay._sample_goals = lambda b, e: new_goal
    sample = replay._get_virtual_samples(indices, env_indices)
    np.testing.assert_array_equal(sample.observations["desired_goal"].cpu().numpy(), new_goal)
    np.testing.assert_array_equal(replay.observations["desired_goal"][0, 0], old_goal)
    np.testing.assert_array_equal(sample.observations["__physics_joints"].cpu().numpy()[0], replay.infos[0, 0]["q_before"])
    np.testing.assert_array_equal(sample.observations["__physics_jacobian"].cpu().numpy()[0], replay.infos[0, 0]["physics"]["jacobian"])
    assert sample.rewards.item() == 0. and sample.dones.item() == 1.
    env.close()


def test_original_gymnasium_contract():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = make_env(configuration())
    check_env(env, skip_render_check=True)
    sb3_check(env)
    env.close()
