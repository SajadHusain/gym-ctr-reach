#!/usr/bin/env python3
"""Follow Cartesian waypoints using one trained goal-conditioned policy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ctr_reach_envs.paper_config import env_kwargs_for_profile
from ctr_reach_envs.envs import CtrReachEnv
from rl_utils import load_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--profile", choices=["current", "paper-2024"], default="current")
    parser.add_argument("path", type=Path, help="CSV with x,y,z columns in metres")
    parser.add_argument("--seed", type=int, default=200_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/path_result.csv"))
    return parser.parse_args()


def load_waypoints(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"x", "y", "z"}.issubset(reader.fieldnames):
            raise ValueError("Path CSV must have x,y,z columns in metres")
        points = np.array(
            [[float(row["x"]), float(row["y"]), float(row["z"])] for row in reader],
            dtype=np.float64,
        )
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("Path CSV must contain at least one finite 3D waypoint")
    if not np.all(np.isfinite(points)):
        raise ValueError("All waypoint coordinates must be finite")
    return points


def main():
    args = parse_args()
    waypoints = load_waypoints(args.path)
    kwargs = env_kwargs_for_profile(args.profile, evaluation=True)
    if args.profile == "paper-2024":
        kwargs["max_steps_per_episode"] = 20
    kwargs["resample_joints"] = False
    kwargs["render_mode"] = "human" if args.render else None
    env = CtrReachEnv(**kwargs)
    model = load_policy(args.model, env, args.profile)
    rows = []
    observation, current_info = env.reset(
        seed=args.seed,
        options={"goal": waypoints[0], "initial_joints": np.zeros(6)},
    )
    try:
        for index, waypoint in enumerate(waypoints):
            if index > 0:
                observation = env.set_goal(waypoint, reset_step_count=True)
                current_info = env.get_info()
            final_info = current_info
            step = 0
            if not final_info["is_success"]:
                for step in range(1, env.max_steps_per_episode + 1):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    final_info = info
                    if terminated or truncated:
                        break
            current_info = final_info
            achieved = np.asarray(final_info["achieved_goal"])
            rows.append(
                {
                    "waypoint": index,
                    "goal_x_m": waypoint[0],
                    "goal_y_m": waypoint[1],
                    "goal_z_m": waypoint[2],
                    "achieved_x_m": achieved[0],
                    "achieved_y_m": achieved[1],
                    "achieved_z_m": achieved[2],
                    "error_m": final_info["error"],
                    "steps": step,
                    "success": int(final_info["is_success"]),
                    "solver_failure": int(final_info["solver_failure"]),
                }
            )
            if not final_info["is_success"] and not args.continue_on_failure:
                break
    finally:
        env.close()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    reached = sum(row["success"] for row in rows)
    print(f"Reached {reached}/{len(waypoints)} waypoints; evaluated {len(rows)}. Results: {output}")


if __name__ == "__main__":
    main()
