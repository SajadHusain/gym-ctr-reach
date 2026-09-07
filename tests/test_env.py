import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env as check_sb3_env

from ctr_reach_envs.config import default_env_kwargs
from ctr_reach_envs.envs import CtrReachEnv
from ctr_reach_envs.her_replay_buffer import GoalTerminationHerReplayBuffer


def make_env(**overrides):
    kwargs = default_env_kwargs(evaluation=False)
    kwargs.update(overrides)
    return CtrReachEnv(**kwargs)


def assert_dict_obs_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_gymnasium_contract():
    env = make_env()
    check_env(env, skip_render_check=True)
    check_sb3_env(env, warn=True)
    observation, _ = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert all(value.dtype == np.float32 for value in observation.values())
    env.close()


def test_spaces_are_finite_and_actions_are_normalized():
    env = make_env()
    np.testing.assert_array_equal(env.action_space.low, -np.ones(6, dtype=np.float32))
    np.testing.assert_array_equal(env.action_space.high, np.ones(6, dtype=np.float32))
    for goal_key in ("achieved_goal", "desired_goal"):
        assert np.all(np.isfinite(env.observation_space[goal_key].low))
        assert np.all(np.isfinite(env.observation_space[goal_key].high))
    env.close()


def test_ddpg_her_can_update_save_and_load(tmp_path):
    env = make_env(max_steps_per_episode=3)
    model = DDPG(
        "MultiInputPolicy",
        env,
        replay_buffer_class=GoalTerminationHerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
        buffer_size=100,
        learning_starts=4,
        batch_size=16,
        policy_kwargs={"net_arch": [32, 32]},
        seed=9,
        verbose=0,
    )
    before = [parameter.detach().clone() for parameter in model.actor.parameters()]
    model.learn(total_timesteps=12)
    assert model.replay_buffer.size() == 12
    assert model._n_updates > 0
    assert any(not old.equal(new.detach()) for old, new in zip(before, model.actor.parameters()))
    observation, _ = env.reset(seed=11)
    expected, _ = model.predict(observation, deterministic=True)
    model.save(tmp_path / "model")
    loaded = DDPG.load(tmp_path / "model", env=env, device="cpu")
    actual, _ = loaded.predict(observation, deterministic=True)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    env.close()


def test_seeded_resets_are_reproducible():
    first = make_env()
    second = make_env()
    obs_a, _ = first.reset(seed=42)
    obs_b, _ = second.reset(seed=42)
    assert_dict_obs_equal(obs_a, obs_b)
    action = np.zeros(6, dtype=np.float32)
    transition_a = first.step(action)
    transition_b = second.step(action)
    assert_dict_obs_equal(transition_a[0], transition_b[0])
    assert transition_a[1:4] == transition_b[1:4]
    first.close()
    second.close()


def test_goal_relabel_does_not_leave_old_goal_in_state_vector():
    env = make_env()
    observation, _ = env.reset(seed=3)
    state_before = observation["observation"].copy()
    new_goal = observation["desired_goal"] + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    relabelled = env.set_goal(new_goal)
    np.testing.assert_array_equal(relabelled["observation"], state_before)
    np.testing.assert_array_equal(relabelled["desired_goal"], new_goal)
    env.close()


def test_vectorized_reward_uses_per_transition_tolerance():
    env = make_env()
    achieved = np.zeros((2, 3), dtype=np.float32)
    desired = np.array([[0.005, 0.0, 0.0], [0.005, 0.0, 0.0]], dtype=np.float32)
    infos = [{"position_tolerance": 0.01}, {"position_tolerance": 0.001}]
    np.testing.assert_array_equal(env.compute_reward(achieved, desired, infos), [0.0, -1.0])
    env.close()


def test_time_limit_is_truncation_not_task_termination():
    env = make_env(max_steps_per_episode=1)
    env.reset(seed=4, options={"goal": np.array([10.0, 10.0, 10.0])})
    _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
    assert not terminated
    assert truncated
    env.close()


def test_active_model_regression_at_zero_joints():
    env = make_env(resample_joints=False)
    observation, _ = env.reset(
        seed=5,
        options={"goal": np.zeros(3), "initial_joints": np.zeros(6)},
    )
    expected = np.array([0.0, -0.137678908, 0.208828318], dtype=np.float32)
    np.testing.assert_allclose(observation["achieved_goal"], expected, atol=2e-7)
    env.close()
