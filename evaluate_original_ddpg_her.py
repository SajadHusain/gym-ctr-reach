"""Frozen-actor evaluation of an original-IVP checkpoint; no Jacobian controller."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import DDPG
from ctr_reach_envs.ivp.config import make_env, fingerprint
from ctr_reach_envs.training.original import motion_metrics, write_json
from evaluate import wilson_interval


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=910000)
    p.add_argument("--tolerance-m", type=float)
    p.add_argument("--max-steps", type=int, help="Explicit evaluation-horizon override; default is saved training horizon")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--record-trajectories", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if min(a.episodes, a.progress_every) < 1 or (a.max_steps is not None and a.max_steps < 1):
        p.error("Episode count, horizon and progress interval must be positive")
    config = json.loads((a.model.parent / "config.json").read_text(encoding="utf-8"))
    env = make_env(config, evaluation=True, compute_jacobian=False, tolerance=a.tolerance_m)
    if a.max_steps is not None: env.max_steps_per_episode = a.max_steps
    torch.set_num_threads(1)
    model = DDPG.load(a.model, env=env, device="cpu")
    if fingerprint(getattr(model, "original_ivp_config", {})) != fingerprint(config):
        raise ValueError("Adjacent config.json does not match the checkpoint's embedded configuration")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new evaluation directory")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    trace = None
    if a.record_trajectories:
        trace = (a.output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8")
        tw = csv.writer(trace)
        tw.writerow(["episode", "step", "error_m"] + [f"q_before_{i}" for i in range(6)] +
                    [f"action_{i}" for i in range(6)] + [f"delta_q_{i}" for i in range(6)] +
                    [f"tip_{x}" for x in "xyz"] + [f"goal_{x}" for x in "xyz"])
    rows, started, interrupted = [], time.perf_counter(), False
    try:
        for episode in range(a.episodes):
            actions, deltas, failure = [], [], ""
            steps, success, trivial, error, task_hash = 0, False, False, None, ""
            try:
                obs, info = env.reset(seed=a.seed + episode)
                task_hash = hashlib.sha256(np.r_[env.trig_obj.joints, env.desired_goal].astype("<f8").tobytes()).hexdigest()
                trivial = bool(info["is_success"])
                while steps < env.max_steps_per_episode:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    if info["solver_failure"]:
                        raise RuntimeError("Original IVP solve failed")
                    actions.append(action.copy()); deltas.append(info["applied_delta_q"].copy())
                    error = float(info["error"])
                    if trace is not None:
                        tw.writerow([episode, steps, error, *info["q_before"], *action, *info["applied_delta_q"],
                                     *env.achieved_goal, *env.desired_goal])
                    if terminated or truncated:
                        success = bool(info["is_success"])
                        break
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                error = None
            rows.append(dict(episode=episode, seed=a.seed + episode, task_fingerprint=task_hash,
                success=success, steps=steps, final_error_m=error, initially_within_tolerance=trivial,
                failure=failure, **motion_metrics(actions, deltas)))
            if (episode + 1) % a.progress_every == 0:
                elapsed = time.perf_counter() - started
                successes = sum(r["success"] for r in rows)
                print(f"Evaluation {episode+1}/{a.episodes}: success={successes/len(rows):.1%}, "
                      f"ETA={elapsed*(a.episodes-len(rows))/len(rows):.0f} s", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if trace is not None: trace.close()
        with (a.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        successes = sum(r["success"] for r in rows)
        errors = [r["final_error_m"] for r in rows if r["final_error_m"] is not None]
        summary = dict(complete=len(rows) == a.episodes and not interrupted,
            requested_episodes=a.episodes, episodes=len(rows), successes=successes,
            success_rate=successes/len(rows) if rows else None,
            success_rate_wilson_95=wilson_interval(successes, len(rows)) if rows else None,
            failures=sum(bool(r["failure"]) for r in rows), trivial_goals=sum(r["initially_within_tolerance"] for r in rows),
            mean_final_error_m=float(np.mean(errors)) if errors else None,
            median_final_error_m=float(np.median(errors)) if errors else None,
            p95_final_error_m=float(np.percentile(errors, 95)) if errors else None,
            error_aggregation="final states of nonfailed episodes; failures remain in success-rate denominator",
            mean_steps=float(np.mean([r["steps"] for r in rows])) if rows else None,
            motion_metrics={k: float(np.mean([r[k] for r in rows])) for k in motion_metrics([], [])} if rows else {},
            motion_scope="mean episode RMS; compare with success and episode length; no physical timestep",
            checkpoint=str(a.model.resolve()), checkpoint_timesteps=model.num_timesteps,
            first_seed=a.seed, max_steps=env.max_steps_per_episode, tolerance_m=env.get_goal_tolerance(),
            segment_mode=config["segment_mode"], environment_fingerprint=config["environment_fingerprint"],
            deterministic_policy=True, actor_uses_jacobian=env.observation_uses_jacobian,
            physics_observation=env.physics_observation, costs=env.costs,
            elapsed_seconds=time.perf_counter()-started)
        write_json(a.output_dir / "summary.json", summary)
        env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
