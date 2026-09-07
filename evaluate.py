#!/usr/bin/env python3
"""Deterministically evaluate a trained CTR policy on held-out seeded episodes."""

from __future__ import annotations

import argparse
import csv
import json
import time
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
    parser.add_argument("--tolerance-m", type=float, default=1e-3)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N episodes; 0 selects approximately 20 updates.",
    )
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - radius)
    high = 1.0 if successes == total else min(1.0, centre + radius)
    return float(low), float(high)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "episodes.csv"
    fieldnames = (
        "episode",
        "seed",
        "success",
        "steps",
        "return",
        "final_error_m",
        "solver_failure",
    )
    progress_every = args.progress_every or max(1, args.episodes // 20)
    env = make_env(
        evaluation=True,
        seed=args.seed,
        position_tolerance=args.tolerance_m,
    )
    model = DDPG.load(args.model, env=env)
    rows = []
    interrupted = False
    started = time.perf_counter()
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
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
                row = {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "success": int(final_info["is_success"]),
                    "steps": step,
                    "return": episode_return,
                    "final_error_m": float(final_info["error"]),
                    "solver_failure": int(final_info["solver_failure"]),
                }
                rows.append(row)
                writer.writerow(row)
                completed = episode + 1
                if completed % progress_every == 0 or completed == args.episodes:
                    handle.flush()
                    elapsed = time.perf_counter() - started
                    seconds_per_episode = elapsed / completed
                    eta = seconds_per_episode * (args.episodes - completed)
                    successes = sum(item["success"] for item in rows)
                    print(
                        f"Evaluation {completed}/{args.episodes}: "
                        f"success={successes / completed:.1%}, "
                        f"mean_error={np.mean([item['final_error_m'] for item in rows]):.4f} m, "
                        f"ETA={eta:.0f} s",
                        flush=True,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print(f"Evaluation interrupted after {len(rows)} episodes; partial CSV preserved.")
    finally:
        env.close()

    if not rows:
        raise RuntimeError("Evaluation ended before completing one episode")

    successes = sum(row["success"] for row in rows)
    low, high = wilson_interval(successes, len(rows))
    errors = np.array([row["final_error_m"] for row in rows])
    steps = np.array([row["steps"] for row in rows])
    summary = {
        "requested_episodes": args.episodes,
        "episodes": len(rows),
        "complete": not interrupted and len(rows) == args.episodes,
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_rate_wilson_95": [low, high],
        "mean_final_error_m": float(np.mean(errors)),
        "median_final_error_m": float(np.median(errors)),
        "p95_final_error_m": float(np.percentile(errors, 95)),
        "mean_steps": float(np.mean(steps)),
        "solver_failures": sum(row["solver_failure"] for row in rows),
        "evaluation_tolerance_m": args.tolerance_m,
        "deterministic_policy": True,
        "first_seed": args.seed,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-episode results: {csv_path}")


if __name__ == "__main__":
    main()
