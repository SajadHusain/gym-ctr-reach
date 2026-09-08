"""Gymnasium interface around the pinned, archived CTR transition code."""
import copy
import gymnasium as gym
import numpy as np
from .bootstrap import activate


class ArchivedCTREnv(gym.Env):
    metadata = {'render_modes': []}
    render_mode = None

    def __init__(self, representation='egocentric', curriculum='decay', evaluation=False,
                 noisy=False, max_episode_steps=150):
        activate()
        import ctm_envs  # Registration stores the author's actual tube parameters.
        from ctm_envs.envs.ctm_env import CtmEnv
        from .gym_compat import REGISTRY
        if representation not in ('egocentric', 'proprioceptive'):
            raise ValueError('Unknown representation')
        if curriculum not in ('constant', 'linear', 'decay'):
            raise ValueError('Unknown curriculum')
        args = copy.deepcopy(REGISTRY['CTR-Reach-Noisy-v0' if noisy else 'CTR-Reach-v0']['kwargs'])
        args.update(relative_q=representation == 'egocentric', resample_joints=evaluation,
                    render=False, max_episode_steps=int(max_episode_steps))
        args['goal_tolerance_parameters']['function'] = 'constant' if evaluation else curriculum
        self.configuration = args
        self.core = CtmEnv(**args)
        self.action_space = self.core.action_space
        # Float64 matches the archived observations; no observation-value casting.
        # Noise has unbounded Gaussian support, hence unbounded noisy state spaces.
        joint_low = np.array([-1., -1., -.215] * 3)
        joint_high = np.array([1., 1., .215] * 3)
        lo = np.r_[joint_low, [-1., -1., -1., 0.]]
        hi = np.r_[joint_high, [1., 1., 1., .02]]
        if noisy:
            lo[:-1], hi[:-1] = -np.inf, np.inf
        goal_bound = np.inf if noisy else 1.
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(lo, hi, dtype=np.float64),
            'achieved_goal': gym.spaces.Box(-goal_bound, goal_bound, (3,), dtype=np.float64),
            'desired_goal': gym.spaces.Box(-1., 1., (3,), dtype=np.float64)})

    def seed(self, seed=None):
        # Preserve the original process-wide stream for encoder/tracker noise.
        # Seed the two previously unseeded spaces for repeatable new experiments.
        self.core.seed(seed)
        self.core.rep_obj.q_space.seed(seed)
        self.action_space.seed(None if seed is None else seed + 1)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            self.core.rep_obj.set_q(np.array(self.configuration['initial_q'], dtype=float))
        options = options or {}
        if 'q' in options:
            self.core.rep_obj.set_q(np.asarray(options['q']).copy())
        obs = self.core.reset(goal=options.get('goal'))
        return obs, {'position_tolerance': self.core.goal_tol_obj.get_tol()}

    def step(self, action):
        obs, reward, done, info = self.core.step(action)
        terminated = bool(reward == 0)
        truncated = bool(done and not terminated)
        # Required by the author's pinned DDPG logger; no reward/state change.
        info['goal_tolerance'] = info['position_tolerance']
        return obs, float(reward), terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Intentionally uses the CURRENT tolerance, matching original HER.
        return self.core.compute_reward(achieved_goal, desired_goal, info)

    def update_goal_tolerance(self, step):
        self.core.update_goal_tolerance(step)

    def close(self):
        pass


class LegacyLearnerInterface(gym.Env):
    """Private four-return learner boundary; public environment remains Gymnasium."""
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def close(self):
        return self.env.close()
