"""Exploration must be identical for ordinary and physics-guided DDPG."""
import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

from ctr_reach_envs.mechanics.rl_exploration import ExplorationDDPG, exploration_settings
from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG


class ActionProbe(gym.Env):
    observation_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
    # Non-unit bounds test proposal-to-replay scaling as well as sampling.
    action_space = spaces.Box(-2., 2., (6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0., False, True, {}


def sample_actions(algorithm, probability, steps=1000, warmup=32):
    torch.set_num_threads(1)
    kwargs = {} if algorithm is DDPG else {"random_exploration": probability}
    if algorithm is JacobianDDPG:
        kwargs.update(physics_lengths=[.3, .2, .1], physics_weight=.1, physics_final_weight=.1)
    model = algorithm("MlpPolicy", ActionProbe(), seed=719, learning_starts=warmup,
                      policy_kwargs={"net_arch":[16, 16]}, buffer_size=10, device="cpu", **kwargs)
    model._last_obs = model.env.reset()
    settings = exploration_settings()
    noise = NormalActionNoise(np.zeros(6), np.asarray(settings["normalized_action_noise_std"]))
    actions, stored = [], []
    for step in range(steps):
        model.num_timesteps = step
        a, b = model._sample_action(warmup, noise)
        actions.append(a[0].copy()); stored.append(b[0].copy())
    model.env.close()
    return np.asarray(actions), np.asarray(stored), getattr(model, "exploration_counts", {})


def test_profile_matches_existing_paper_configuration_and_explicit_old_profile():
    paper = exploration_settings()
    assert paper["normalized_action_noise_std"] == [.0018]*3+[.025]*3
    assert paper["random_exploration"] == .294
    old = exploration_settings("gaussian")
    assert old["normalized_action_noise_std"] == [.05]*6 and old["random_exploration"] == 0.
    override = exploration_settings("paper", .07, .2)
    assert override["normalized_action_noise_std"] == [.07]*6 and override["random_exploration"] == .2


@pytest.mark.parametrize("kwargs", [{"noise_std": -1.}, {"random_exploration": 1.1},
                                   {"noise_std": float("nan")}, {"profile": "unknown"}])
def test_invalid_exploration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        exploration_settings(**kwargs)


def test_guided_and_ordinary_actors_receive_identical_exploration():
    base = sample_actions(ExplorationDDPG, .294)
    guided = sample_actions(JacobianDDPG, .294)
    np.testing.assert_array_equal(base[0], guided[0])
    np.testing.assert_array_equal(base[1], guided[1])
    assert base[2] == guided[2]
    assert base[2]["warmup_uniform"] == 32
    assert 220 < base[2]["mixture_uniform"] < 350
    assert sum(base[2].values()) == 1000


def test_zero_probability_preserves_sb3_random_stream_exactly():
    base = sample_actions(DDPG, 0., steps=100)
    compatible = sample_actions(ExplorationDDPG, 0., steps=100)
    np.testing.assert_array_equal(base[0], compatible[0])
    np.testing.assert_array_equal(base[1], compatible[1])


def test_uniform_sampling_spans_all_action_axes_and_preserves_replay_labels():
    actions, stored, counts = sample_actions(ExplorationDDPG, 1.)
    np.testing.assert_allclose(stored, actions/2, atol=1.2e-7)
    assert counts == {"warmup_uniform":32, "mixture_uniform":968, "policy_with_noise":0}
    assert np.all(actions <= 2.) and np.all(actions >= -2.)
    for axis in actions.T:
        assert np.all(np.histogram(axis, bins=8, range=(-2.,2.))[0] > 60)


def test_study_passes_identical_profile_and_overrides_to_both_arms(tmp_path, monkeypatch):
    import run_physics_study as study
    commands = []
    monkeypatch.setattr(study, "run_command", lambda command, out: commands.append(command) or True)
    config = {**study.DEFAULTS, "seeds": [10], "random_exploration": .3, "noise_std": .06}
    study.validate_config(config)
    study.train({"config":config}, tmp_path)
    assert len(commands) == 2
    for cmd in commands:
        assert cmd[cmd.index("--exploration-profile")+1] == "paper"
        assert cmd[cmd.index("--random-exploration")+1] == "0.3"
        assert cmd[cmd.index("--noise-std")+1] == "0.06"
