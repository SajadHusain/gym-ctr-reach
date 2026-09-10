"""Audit checks use real equilibrium transitions and a recording fake actor."""
import unittest
import numpy as np

from audit_simple_reaching import (Task, check_transition, fixed_action,
    make_tasks, policy_observation, run_case)
from ctr_reach_envs.mechanics.simple_rl_env import JointConstrainedReachEnv


class RecordingActor:
    def __init__(self, action):
        self.action = action
        self.observations = []

    def predict(self, obs, deterministic=False):
        assert deterministic
        self.observations.append({k: v.copy() for k, v in obs.items()})
        return self.action.copy(), None


class TestReachingAudit(unittest.TestCase):
    def setUp(self):
        self.env = JointConstrainedReachEnv(compute_jacobian=False)

    def tearDown(self):
        self.env.close()

    def test_goal_corruption_does_not_modify_real_observation(self):
        obs, _ = self.env.reset(seed=810000)
        before = {k: v.copy() for k, v in obs.items()}
        wrong = obs['desired_goal'] + np.array([.001, 0, 0])
        zero = policy_observation(obs, 'goal_zero')
        shuffled = policy_observation(obs, 'goal_shuffled', wrong)
        np.testing.assert_array_equal(zero['desired_goal'], obs['achieved_goal'])
        np.testing.assert_array_equal(shuffled['desired_goal'], wrong.astype(np.float32))
        for key in obs:
            np.testing.assert_array_equal(obs[key], before[key])
            self.assertFalse(np.shares_memory(obs[key], zero[key]))

    def test_original_tasks_match_existing_evaluator_seeds(self):
        tasks = make_tasks(self.env, 2, 810000, reverse=True)
        for task in tasks:
            obs, info = self.env.reset(seed=task.seed)
            np.testing.assert_array_equal(task.joints, info['initial_q'])
            np.testing.assert_array_equal(task.goal, obs['desired_goal'])
            self.assertGreater(task.goal[0], 0)
            self.assertLess(task.reverse_goal[0], 0)

    def test_independent_check_catches_false_success(self):
        obs, info = self.env.reset(seed=810000)
        self.assertFalse(info['is_success'])
        with self.assertRaisesRegex(AssertionError, 'success/termination'):
            check_transition(obs, {**info, 'is_success': True}, True,
                             obs['desired_goal'].astype(float), self.env.tolerance_m)

    def test_independent_check_catches_changed_goal(self):
        obs, info = self.env.reset(seed=810000)
        original = obs['desired_goal'].astype(float)
        obs['desired_goal'] = obs['desired_goal'] + np.float32(.01)
        with self.assertRaisesRegex(AssertionError, 'real task goal'):
            check_transition(obs, info, False, original, self.env.tolerance_m)

    def test_corrupted_goal_is_only_seen_by_actor(self):
        task = make_tasks(self.env, 1, 810000)[0]
        actor = RecordingActor(np.zeros(6, dtype=np.float32))
        wrong = task.goal + np.array([.001, 0, 0])
        row, traces = run_case(self.env, task, 'goal_shuffled', actor,
                               wrong_goal=wrong, max_steps=2)
        self.assertEqual(row['failure'], '')
        for seen in actor.observations:
            np.testing.assert_array_equal(seen['desired_goal'], wrong.astype(np.float32))
        for trace in traces:
            np.testing.assert_array_equal([trace[f'goal_{i}'] for i in range(3)], task.goal)

    def nominal_task(self):
        obs, info = self.env.reset(seed=810000)
        q = info['initial_q'].copy()
        target = q.copy()
        for _ in range(4):
            target += self.env.projected_delta(target, fixed_action(self.env))
        goal = self.env.solver.solve(target).tip.astype(np.float32).astype(float)
        return Task(810000, q, goal)

    def test_fixed_control_has_no_actor_or_derivative_dependency(self):
        row, traces = run_case(self.env, self.nominal_task(), 'fixed', max_steps=8)
        self.assertEqual(row['failure'], '')
        self.assertTrue(row['success'])
        self.assertEqual(row['steps'], 4)
        self.assertEqual(self.env.costs['sensitivity_calls'], 0)
        self.assertEqual(self.env.costs['stability_calls'], 0)
        for trace in traces:
            np.testing.assert_array_equal([trace[f'proposed_{i}'] for i in range(6)], fixed_action(self.env))

    def test_hold_test_exposes_motion_after_first_hit(self):
        task = self.nominal_task()
        actor = RecordingActor(fixed_action(self.env))
        row, traces = run_case(self.env, task, 'actor', actor, max_steps=8, hold_steps=3)
        self.assertEqual(row['failure'], '')
        self.assertTrue(row['success'])
        self.assertEqual(row['steps'], 4)
        self.assertEqual(row['hold_steps'], 3)
        self.assertFalse(row['held_within_tolerance'])
        self.assertGreater(row['hold_max_error_m'], self.env.tolerance_m)
        self.assertEqual(sum(t['phase'] == 'hold' for t in traces), 3)
        self.assertLess(row['final_error_m'], self.env.tolerance_m)

    def test_actor_failure_is_recorded(self):
        class BrokenActor:
            def predict(self, *args, **kwargs):
                raise RuntimeError('failed prediction')
        task = make_tasks(self.env, 1, 810000)[0]
        row, traces = run_case(self.env, task, 'actor', BrokenActor())
        self.assertIn('failed prediction', row['failure'])
        self.assertFalse(row['success'])
        self.assertEqual(traces, [])


if __name__ == '__main__':
    unittest.main()
