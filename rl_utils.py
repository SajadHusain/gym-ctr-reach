from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from ctr_reach_envs.config import default_env_kwargs
from ctr_reach_envs.envs import CtrReachEnv


MONITOR_INFO = ("is_success", "error", "position_tolerance", "solver_failure")


def make_env(*, evaluation: bool, seed: int, render: bool = False, monitor_dir=None):
    kwargs = default_env_kwargs(evaluation=evaluation)
    kwargs["render_mode"] = "human" if render else None
    env = CtrReachEnv(**kwargs)
    env.reset(seed=seed)
    filename = None
    if monitor_dir is not None:
        monitor_dir = Path(monitor_dir)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        filename = str(monitor_dir / ("eval" if evaluation else "train"))
    return Monitor(env, filename=filename, info_keywords=MONITOR_INFO)

