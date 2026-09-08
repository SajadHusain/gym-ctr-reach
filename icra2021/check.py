"""Reference comparisons for the archived CTR/Gymnasium compatibility port."""
import argparse
import copy
import importlib.util
import json
import sys
import numpy as np
from .bootstrap import ROOT, SOURCES, blob_sha, manifest
from .env import ArchivedCTREnv, LegacyLearnerInterface
from .policy import NumpyActor, read_checkpoint, flatten


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_sources():
    for item in manifest():
        assert blob_sha((SOURCES / item['path']).read_bytes()) == item['git_blob_sha1'], item['path']


def check_transitions():
    from gymnasium.utils.env_checker import check_env
    env = ArchivedCTREnv()
    check_env(env, skip_render_check=True)
    from ctm_envs.envs.ctm_env import CtmEnv
    reference = CtmEnv(**copy.deepcopy(env.configuration))
    rng = np.random.RandomState(12)
    for seed in range(3):
        env.seed(seed)
        q = env.core.rep_obj.sample_goal()
        goal = env.core.model.forward_kinematics(env.core.rep_obj.sample_goal())
        reference.rep_obj.set_q(q.copy())
        np.random.seed(seed)
        old_obs = reference.reset(goal=goal)
        new_obs, _ = env.reset(seed=seed, options={'q': q, 'goal': goal})
        for key in old_obs:
            np.testing.assert_array_equal(old_obs[key], new_obs[key])
        for _ in range(8):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            old_obs, old_reward, old_done, old_info = reference.step(action)
            new_obs, reward, terminated, truncated, info = env.step(action)
            for key in old_obs:
                np.testing.assert_array_equal(old_obs[key], new_obs[key])
            np.testing.assert_array_equal(reference.rep_obj.q, env.core.rep_obj.q)
            assert old_reward == reward and old_done == (terminated or truncated)
            assert old_info['error'] == info['error']
    # The learner sees timeouts as terminal, exactly as in the archived API.
    short = ArchivedCTREnv(max_episode_steps=1)
    legacy = LegacyLearnerInterface(short)
    legacy.reset()
    assert legacy.step(np.zeros(6))[2]
    # Runtime tolerance changes do not alter declared observation bounds.
    for function in ('constant', 'linear', 'decay'):
        e = ArchivedCTREnv(curriculum=function)
        e.update_goal_tolerance(200000)
        np.testing.assert_allclose(e.core.goal_tol_obj.get_tol(), .001, atol=1e-12)
    assert 'gym' not in sys.modules


def check_her():
    her = load_module('original_her', SOURCES / 'stable-baselines/stable_baselines/her/replay_buffer.py')

    class Memory:
        def __init__(self):
            self.rows = []
        def add(self, *row):
            self.rows.append(row)

    class Goals:
        def convert_obs_to_dict(self, obs):
            return dict(observation=obs[:13], achieved_goal=obs[13:16], desired_goal=obs[16:])
        def convert_dict_to_obs(self, obs):
            return flatten(obs)
        def compute_reward(self, a, b, info):
            return -float(np.linalg.norm(a - b) > .001)

    memory = Memory()
    wrapper = her.HindsightExperienceReplayWrapper(memory, 4, her.GoalSelectionStrategy.FUTURE, Goals())
    original = []
    for i in range(3):
        obs = np.r_[np.arange(13), [i, 0, 0], [9, 9, 9]].astype(float)
        nxt = obs.copy()
        nxt[13] += 1
        original.append(obs)
        wrapper.add(obs, np.zeros(6), -1., nxt, i == 2)
        if i < 2:
            assert len(memory.rows) == 0
    assert len(memory.rows) == 11  # 3 real + 4 synthetic for each of first 2.
    for i in range(2):
        for row in memory.rows[5*i+1:5*i+5]:
            np.testing.assert_array_equal(row[0][:13], original[i][:13])
            assert row[-1] is False
            assert row[0][16] in range(i+1, 3)


def check_original_solver():
    # Original scalar-array idioms require the pinned NumPy 1.x runtime.
    if int(np.__version__.split('.')[0]) >= 2:
        return 'not run: original scalar-array code requires pinned NumPy 1.x'
    env = ArchivedCTREnv()
    original = load_module('original_exact', SOURCES / 'gym-ctm-ros/ctm_envs/envs/exact_model.py')
    reference = original.ExactModel(env.core.tubes)
    for seed in range(10):
        env.seed(seed)
        q = env.core.rep_obj.sample_goal()
        np.testing.assert_allclose(env.core.model.forward_kinematics(q), reference.forward_kinematics(q), rtol=1e-10, atol=1e-12)
    return 'passed: 10 original-vs-port FK configurations'


def check_tensorflow():
    from .train import make_model
    env = ArchivedCTREnv(representation='proprioceptive', evaluation=True)
    model = make_model(env, seed=4)
    checkpoint = ROOT / '_checkpoints/author_500000.zip'
    metadata, weights = read_checkpoint(checkpoint)
    model.model.load_parameters(weights, exact_match=True)
    actor = NumpyActor(checkpoint, env.action_space)
    max_error = 0.
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        expected, _ = model.predict(obs, deterministic=True)
        actual = actor.predict(obs)
        error = float(np.max(np.abs(expected - actual)))
        max_error = max(error, max_error)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    model.model.sess.close()
    return max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensorflow', action='store_true')
    args = parser.parse_args()
    check_sources()
    check_transitions()
    check_her()
    results = dict(source_hashes='passed', gymnasium_transition_parity='passed',
                   archived_her_semantics='passed', original_solver=check_original_solver())
    if args.tensorflow:
        results['tensorflow_numpy_max_action_difference'] = check_tensorflow()
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
