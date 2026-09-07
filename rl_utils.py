from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor

from ctr_reach_envs.config import default_env_kwargs
from ctr_reach_envs.envs import CtrReachEnv


MONITOR_INFO = ("is_success", "error", "position_tolerance", "solver_failure")


def make_env(
    *,
    evaluation: bool,
    seed: int,
    render: bool = False,
    monitor_dir=None,
    position_tolerance: float | None = None,
):
    kwargs = default_env_kwargs(evaluation=evaluation)
    if position_tolerance is not None:
        position_tolerance = float(position_tolerance)
        if not np.isfinite(position_tolerance) or position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be finite and positive")
        kwargs["goal_tolerance_parameters"].update(
            {
                "initial_tol": position_tolerance,
                "final_tol": position_tolerance,
                "function": "constant",
                "set_tol": position_tolerance,
            }
        )
    kwargs["render_mode"] = "human" if render else None
    env = CtrReachEnv(**kwargs)
    env.reset(seed=seed)
    filename = None
    if monitor_dir is not None:
        monitor_dir = Path(monitor_dir)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        filename = str(monitor_dir / ("eval" if evaluation else "train"))
    return Monitor(env, filename=filename, info_keywords=MONITOR_INFO)
