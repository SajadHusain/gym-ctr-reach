"""Falsification checks for the simple reaching benchmark; no training updates.

This diagnostic preserves the training environment. Fixed controls and corrupted
policy goals are labelled separately from ordinary actor evaluation.
"""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ctr_reach_envs.mechanics.simple_rl_env import JointConstrainedReachEnv, make_reach_env


MODES = ("actor", "fixed", "goal_zero", "goal_shuffled", "reverse_actor")


@dataclass
class Task:
    seed: int
    joints: np.ndarray
    goal: np.ndarray
    reverse_goal: np.ndarray | None = None


def fixed_action(env):
    """Mean witness command, repeated four times; no target is consulted."""
    delta = np.r_[np.full(env.n, -.0005), np.full(env.n, .04)]
    return np.clip(delta / env.action_scales, -1., 1.).astype(np.float32)


def make_tasks(env, episodes, seed, reverse=False):
    tasks = []
    for i in range(episodes):
        obs, info = env.reset(seed=seed+i)
        q = info["initial_q"].copy()
        reverse_goal = None
        if reverse:
            # Feasible nearby target in the opposite rotation direction. The
            # initial joints are the same; no branch state or stability check.
            rng = np.random.default_rng(np.random.SeedSequence([seed+i, 719]))
            target = q.copy()
            if env.task_profile == "generalized_hold":
                for _ in range(info["goal_witness_steps"]):
                    target += env.projected_delta(target, -info["goal_witness_action"])
            else:
                for _ in range(4):
                    delta = np.r_[np.full(env.n, -.0005), -(.04+rng.uniform(-.015, .015, env.n))]
                    target += env.projected_delta(target, np.clip(delta/env.action_scales, -1., 1.))
            target_obs, _ = env.reset(options={"joints": target, "goal": obs["desired_goal"]})
            reverse_goal = target_obs["achieved_goal"].astype(float)
        tasks.append(Task(seed+i, q, obs["desired_goal"].astype(float), reverse_goal))
    return tasks


def policy_observation(obs, mode, wrong_goal=None):
    """Change only policy input; the environment keeps the true task goal."""
    result = {key: value.copy() for key, value in obs.items()}
    if mode == "goal_zero":
        result["desired_goal"] = result["achieved_goal"].copy()
    elif mode == "goal_shuffled":
        if wrong_goal is None:
            raise ValueError("A shuffled goal is required")
        result["desired_goal"] = np.asarray(wrong_goal, dtype=np.float32).copy()
    return result


def independent_error(obs, goal):
    return float(np.linalg.norm(obs["achieved_goal"].astype(float)-goal))


def check_transition(obs, info, terminated, goal, tolerance, terminate_on_success=True):
    if not np.array_equal(obs["desired_goal"], goal.astype(np.float32)):
        raise AssertionError("The real task goal changed during the audit")
    error = independent_error(obs, goal)
    expected = error <= tolerance
    if expected != bool(info["is_success"]) or (expected and terminate_on_success) != bool(terminated):
        raise AssertionError("Reported success/termination disagrees with Cartesian error")
    if abs(error-float(info["error"])) > 1e-10:
        raise AssertionError("Reported error disagrees with Cartesian coordinates")
    return error


def run_case(env, task, mode, model=None, wrong_goal=None, max_steps=8, hold_steps=5):
    goal = task.reverse_goal if mode == "reverse_actor" else task.goal
    row = dict(mode=mode, seed=task.seed, success=False, failure="", steps=0,
               initial_error_m=None, initially_within_tolerance=False,
               final_error_m=None, task_fingerprint=None,
               hold_steps=0, hold_max_error_m=None, held_within_tolerance=None)
    traces = []
    if goal is None:
        raise ValueError("The reverse target was not generated")
    try:
        obs, info = env.reset(seed=task.seed, options={"joints": task.joints, "goal": goal})
        goal = obs["desired_goal"].astype(float)
        row["initial_error_m"] = independent_error(obs, goal)
        row["initially_within_tolerance"] = row["initial_error_m"] <= env.tolerance_m
        row["task_fingerprint"] = hashlib.sha256(np.r_[task.joints, goal].astype("<f8").tobytes()).hexdigest()
        count = min(max_steps, 4) if mode == "fixed" else max_steps
        for step in range(count):
            action = fixed_action(env) if mode == "fixed" else model.predict(
                policy_observation(obs, mode, wrong_goal), deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            error = check_transition(obs, info, term, goal, env.tolerance_m, env.terminate_on_success)
            row.update(steps=step+1, final_error_m=error, success=error <= env.tolerance_m)
            traces.append(trace_row(mode, task.seed, "reach", step+1, error, obs, info))
            # This diagnostic deliberately measures first hit, then a separate
            # continuation. The main evaluator measures the full fixed horizon.
            if row["success"] or term or trunc:
                break
        if row["success"] and mode == "actor" and hold_steps:
            errors = []
            for step in range(hold_steps):
                # A successful Gym episode normally ends. Explicitly start an
                # audit continuation at the same q and same goal to test the
                # actor's subsequent commands; never alter training semantics.
                obs, _ = env.reset(options={"joints": env.equilibrium.joints.copy(), "goal": goal})
                action = model.predict(obs, deterministic=True)[0]
                obs, _, term, _, info = env.step(action)
                error = check_transition(obs, info, term, goal, env.tolerance_m, env.terminate_on_success)
                errors.append(error)
                row.update(hold_steps=step+1, hold_max_error_m=max(errors),
                           held_within_tolerance=bool(max(errors) <= env.tolerance_m))
                traces.append(trace_row(mode, task.seed, "hold", step+1, error, obs, info))
    except Exception as exc:
        row["failure"] = f"{type(exc).__name__}: {exc}"
        # Preserve an already observed reach if the separate hold phase fails.
        # The failure field still makes the overall audit incomplete.
    return row, traces


def trace_row(mode, seed, phase, step, error, obs, info):
    row = dict(mode=mode, seed=seed, phase=phase, step=step, error_m=error,
               reported_success=bool(info["is_success"]))
    for name, values in (("tip", obs["achieved_goal"]), ("goal", obs["desired_goal"]),
                         ("proposed", info["proposed_action"]), ("executed", info["executed_action"])):
        row.update({f"{name}_{i}": float(x) for i, x in enumerate(values)})
    return row


def load_actor(checkpoint, env):
    # Optional imports keep the fixed-command control usable without PyTorch.
    import torch
    from stable_baselines3 import DDPG
    from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG
    config = json.loads((checkpoint.parent/"config.json").read_text())
    if config.get("plant") != env.plant_version or config.get("observation_bounds_version") != 3:
        raise ValueError("This audit requires a simple-jacobian-rl checkpoint")
    if config.get("model_fingerprint") != env.solver.model_fingerprint:
        raise ValueError("Checkpoint and audit model parameters differ")
    torch.set_num_threads(1)
    algorithm = JacobianDDPG if config["algorithm"] == "JacobianDDPG" else DDPG
    return algorithm.load(checkpoint, env=env, device="cpu")


def policy_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=920000)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--hold-steps", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simple_reaching_audit"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or args.max_steps < 1 or args.hold_steps < 0:
        parser.error("Invalid counts or seed")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate modes")
    if "goal_shuffled" in args.modes and args.episodes < 2:
        parser.error("Shuffled-goal evaluation requires at least two tasks")
    if args.model is None and args.modes != ["fixed"]:
        parser.error("Supply a checkpoint or select --modes fixed")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a new, empty output directory")
    config = json.loads((args.model.parent/"config.json").read_text()) if args.model else {}
    if config and (config.get("plant") != JointConstrainedReachEnv.plant_version or config.get("observation_bounds_version") != 3):
        parser.error("This audit requires a simple-jacobian-rl checkpoint")
    env = make_reach_env(config, max_episode_steps=args.max_steps, compute_jacobian=False)
    model = load_actor(args.model, env) if any(m != "fixed" for m in args.modes) else None
    before_digest = None if model is None else policy_digest(model)
    before_updates = 0 if model is None else model._n_updates
    started = time.perf_counter()
    tasks = make_tasks(env, args.episodes, args.seed, reverse="reverse_actor" in args.modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_records = [dict(seed=t.seed, initial_q=t.joints.tolist(), goal_m=t.goal.tolist(),
                        reverse_goal_m=None if t.reverse_goal is None else t.reverse_goal.tolist()) for t in tasks]
    (args.output_dir/"tasks.json").write_text(json.dumps(task_records, indent=2)+"\n")
    rows = []
    with (args.output_dir/"episodes.csv").open("w", newline="") as f, (args.output_dir/"trajectories.csv").open("w", newline="") as tf:
        writer = trace_writer = None
        for mode in args.modes:
            for i, task in enumerate(tasks):
                row, traces = run_case(env, task, mode, model=model,
                    wrong_goal=tasks[(i+1) % len(tasks)].goal, max_steps=args.max_steps,
                    hold_steps=args.hold_steps)
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); f.flush()
                for trace in traces:
                    if trace_writer is None:
                        trace_writer = csv.DictWriter(tf, fieldnames=list(trace)); trace_writer.writeheader()
                    trace_writer.writerow(trace)
                tf.flush()
                print(f"{mode} {i+1}/{len(tasks)}: success={row['success']}, error={row['final_error_m']}, failure={row['failure']}", flush=True)
    unchanged = None if model is None else policy_digest(model) == before_digest
    updates = 0 if model is None else model._n_updates-before_updates
    summary = dict(complete=not any(r["failure"] for r in rows) and unchanged is not False and updates == 0,
        checkpoint=str(args.model) if args.model else None,
        checkpoint_timesteps=None if model is None else model.num_timesteps,
        episodes_per_mode=args.episodes, seed=args.seed, max_steps=args.max_steps,
        task_profile=env.task_profile, task_settings=env.task_settings,
        tolerance_m=env.tolerance_m, elapsed_seconds=time.perf_counter()-started,
        semantics="Diagnostic only; original and reverse targets separated. Hold phases continue from reached q.",
        physics_costs=dict(env.costs), modes={})
    summary.update(policy_weights_unchanged=unchanged, optimizer_updates_during_audit=updates,
                   policy_sha256=before_digest)
    for mode in args.modes:
        subset = [r for r in rows if r["mode"] == mode]
        errors = [r["final_error_m"] for r in subset if r["final_error_m"] is not None]
        held = [r for r in subset if r["hold_steps"] == args.hold_steps and args.hold_steps > 0]
        summary["modes"][mode] = dict(successes=sum(r["success"] for r in subset),
            success_rate=sum(r["success"] for r in subset)/len(subset),
            failures=sum(bool(r["failure"]) for r in subset),
            initially_within_tolerance=sum(r["initially_within_tolerance"] for r in subset),
            mean_final_error_m=float(np.mean(errors)) if errors else None,
            fully_observed_hold_trials=len(held),
            successful_hold_trials=sum(r["held_within_tolerance"] for r in held))
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    env.close()
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
