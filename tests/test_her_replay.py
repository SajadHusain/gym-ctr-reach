import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ctr_reach_envs.config import default_env_kwargs
from ctr_reach_envs.envs import CtrReachEnv
from ctr_reach_envs.her_replay_buffer import GoalTerminationHerReplayBuffer


@pytest.mark.parametrize("normalize", [False, True])
def test_relabelled_terminal_targets_and_historical_tolerances(monkeypatch, normalize):
    vec = DummyVecEnv([lambda: CtrReachEnv(**default_env_kwargs())])
    normalizer = VecNormalize(vec) if normalize else None
    try:
        replay = GoalTerminationHerReplayBuffer(
            20, vec.observation_space, vec.action_space, vec,
            device="cpu", copy_info_dict=True,
        )
        # Nonterminal -> success, terminal -> failure, timeout -> failure/success,
        # followed by the same 5 mm error at historical 10 mm and 1 mm tolerances.
        original_done = [False, True, True, True, False, False]
        timeouts = [False, False, True, True, False, False]
        tolerances = [0.001, 0.001, 0.001, 0.001, 0.01, 0.001]
        goals = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0, 0],
                          [0, 0, 0], [0.005, 0, 0], [0.005, 0, 0]], dtype=np.float32)
        for i in range(6):
            state = np.zeros((1, 10), dtype=np.float32)
            state[:, -1] = tolerances[i]
            obs = {
                "observation": state,
                "achieved_goal": np.zeros((1, 3), dtype=np.float32),
                "desired_goal": np.array([[0, 0, 0] if i == 1 else [0.1, 0, 0]], dtype=np.float32),
            }
            replay.add(obs, obs, np.zeros((1, 6), dtype=np.float32),
                       np.array([0 if i == 1 else -1], dtype=np.float32),
                       np.array([original_done[i]]),
                       [{"position_tolerance": tolerances[i], "TimeLimit.truncated": timeouts[i]}])
        original_goals = replay.observations["desired_goal"].copy()
        monkeypatch.setattr(replay, "_sample_goals", lambda *_: goals.copy())
        vec.env_method("update_goal_tolerance", 200_000)
        if normalizer is not None:
            # Strong/clipped normalization would hide goal differences if terminal
            # labels were incorrectly computed after observation normalization.
            for key in ("desired_goal", "achieved_goal"):
                normalizer.obs_rms[key].mean[:] = 0.1
                normalizer.obs_rms[key].var[:] = 1e-6
            normalizer.clip_obs = 1.0
        indices, env_indices = np.arange(6), np.zeros(6, dtype=int)
        samples = replay._get_virtual_samples(indices, env_indices, normalizer)
        np.testing.assert_array_equal(samples.dones.cpu().numpy().ravel(), [1, 0, 0, 1, 1, 0])
        rewards = samples.rewards.cpu().numpy()
        if normalizer is not None:
            rewards = normalizer.unnormalize_reward(rewards)
        np.testing.assert_allclose(rewards.ravel(), [0, -1, -1, 0, 0, -1], atol=1e-6)
        np.testing.assert_array_equal(replay.observations["desired_goal"], original_goals)
        real = replay._get_real_samples(indices, env_indices)
        np.testing.assert_array_equal(real.dones.cpu().numpy().ravel(), [0, 1, 0, 0, 0, 0])
    finally:
        (normalizer if normalizer is not None else vec).close()


def test_curriculum_replay_requires_stored_info():
    vec = DummyVecEnv([lambda: CtrReachEnv(**default_env_kwargs())])
    try:
        with pytest.raises(ValueError, match="copy_info_dict=True"):
            GoalTerminationHerReplayBuffer(20, vec.observation_space, vec.action_space, vec)
    finally:
        vec.close()
