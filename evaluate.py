#!/usr/bin/env python3
"""Deterministically evaluate a trained CTR policy on held-out seeded episodes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from rl_utils import make_env, load_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--profile", choices=["current", "paper-2024"], default="current")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--tolerance-m", type=float, default=1e-3)
    parser.add_argument(
        "--record-trajectories", action="store_true",
        help="Write per-step tip positions, normalized commands and applied joint changes.",
    )
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


def trajectory_row(episode, step, before, after, action, joints_before, joints_after, info):
    """A transition's actual motion; delta joints include clipping and constraints."""
    row = {
        "episode": episode, "step": step,
        "error_before_m": float(np.linalg.norm(before["desired_goal"] - before["achieved_goal"])),
        "error_m": float(info["error"]),
        "position_tolerance_m": float(info["position_tolerance"]),
        "success": int(info["is_success"]), "solver_failure": int(info["solver_failure"]),
    }
    for index, axis in enumerate("xyz"):
        row[f"tip_before_{axis}_m"] = float(before["achieved_goal"][index])
        row[f"tip_after_{axis}_m"] = float(after["achieved_goal"][index])
        row[f"goal_{axis}_m"] = float(after["desired_goal"][index])
    for index in range(6):
        joint = f"beta_{index}_m" if index < 3 else f"alpha_{index - 3}_rad"
        row[f"action_{index}_normalized"] = float(action[index])
        row[f"before_{joint}"] = float(joints_before[index])
        row[f"after_{joint}"] = float(joints_after[index])
        row[f"delta_{joint}"] = float(joints_after[index] - joints_before[index])
    return row


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
        profile=args.profile,
    )
    rows = []
    interrupted = False
    started = time.perf_counter()
    try:
        model = load_policy(args.model, env, args.profile)
        with ExitStack() as stack:
            handle = stack.enter_context(csv_path.open("w", newline="", encoding="utf-8"))
            trace_handle = None
            trace_writer = None
            if args.record_trajectories:
                trace_handle = stack.enter_context(
                    (output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8")
                )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=args.seed + episode)
                episode_return = 0.0
                final_info = None
                for step in range(1, env.unwrapped.max_steps_per_episode + 1):
                    action, _ = model.predict(observation, deterministic=True)
                    if trace_handle is not None:
                        before = observation
                        joints_before = env.unwrapped.trig_obj.joints.copy()
                    observation, reward, terminated, truncated, info = env.step(action)
                    if trace_handle is not None:
                        trace = trajectory_row(
                            episode, step, before, observation, action, joints_before,
                            env.unwrapped.trig_obj.joints.copy(), info,
                        )
                        if trace_writer is None:
                            trace_writer = csv.DictWriter(trace_handle, fieldnames=trace.keys())
                            trace_writer.writeheader()
                        trace_writer.writerow(trace)
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
                handle.flush()
                if trace_handle is not None:
                    trace_handle.flush()
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
        "checkpoint": str(args.model.resolve()),
        "profile": args.profile,
        "checkpoint_timesteps": int(model.num_timesteps),
        "trajectories_recorded": bool(args.record_trajectories),
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
