#!/usr/bin/env python3
"""Deterministically evaluate a trained CTR policy on held-out seeded episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG

from rl_utils import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return float(centre - radius), float(centre + radius)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(evaluation=True, seed=args.seed)
    model = DDPG.load(args.model, env=env)
    rows = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            final_info = None
            for step in range(1, env.unwrapped.max_steps_per_episode + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                final_info = info
                if terminated or truncated:
                    break
            rows.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "success": int(final_info["is_success"]),
                    "steps": step,
                    "return": episode_return,
                    "final_error_m": float(final_info["error"]),
                    "solver_failure": int(final_info["solver_failure"]),
                }
            )
    finally:
        env.close()

    csv_path = output_dir / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(row["success"] for row in rows)
    low, high = wilson_interval(successes, len(rows))
    errors = np.array([row["final_error_m"] for row in rows])
    steps = np.array([row["steps"] for row in rows])
    summary = {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_rate_wilson_95": [low, high],
        "mean_final_error_m": float(np.mean(errors)),
        "median_final_error_m": float(np.median(errors)),
        "p95_final_error_m": float(np.percentile(errors, 95)),
        "mean_steps": float(np.mean(steps)),
        "solver_failures": sum(row["solver_failure"] for row in rows),
        "evaluation_tolerance_m": 1e-3,
        "deterministic_policy": True,
        "first_seed": args.seed,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-episode results: {csv_path}")


if __name__ == "__main__":
    main()

