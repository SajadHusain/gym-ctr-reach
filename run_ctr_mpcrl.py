"""Train/evaluate MPC-as-policy RL. No demonstrations or neural actor are used."""
import argparse
from dataclasses import asdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from ctr_reach_envs.ivp.config import make_env, fingerprint
from ctr_reach_envs.mpc.learning import LearningOptions, MPCQLearner
from evaluate_ctr_mpc import motion_metrics, wilson_interval


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def checkpoint(source, learner, tolerance, max_steps, steps):
    payload = dict(format="ctr-mpcrl-v1", source_config=source,
        options=asdict(learner.options), parameters=learner.parameters(),
        tolerance_m=tolerance, max_steps=max_steps, timesteps=steps, updates=learner.updates,
        semantics="inference checkpoint; optimizer/replay/random state are not resumable")
    return dict(**payload, fingerprint=fingerprint(payload))


def read_checkpoint(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = data.pop("fingerprint", None)
    if data.get("format") != "ctr-mpcrl-v1" or digest != fingerprint(data):
        raise ValueError("Unrecognized or modified MPC-RL checkpoint")
    return data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--seed", type=int, default=10)
    train.add_argument("--tolerance-m", type=float, default=.0015)
    train.add_argument("--max-steps", type=int, default=200)
    train.add_argument("--horizon", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--max-parameter-change", type=float, default=.02)
    train.add_argument("--exploration-strength", type=float, default=.01)
    train.add_argument("--max-iterations", type=int, default=40)
    train.add_argument("--max-model-evaluations", type=int, default=1500)
    train.add_argument("--checkpoint-freq", type=int, default=100)
    train.add_argument("--freeze", action="store_true", help="Fixed-parameter control ablation")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=920000)
    evaluate.add_argument("--tolerance-m", type=float)
    evaluate.add_argument("--max-steps", type=int)
    for command in (train, evaluate):
        command.add_argument("--progress-every", type=int, default=10)
        command.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    training = args.mode == "train"
    if args.progress_every < 1: p.error("progress interval must be positive")
    if training:
        if args.steps < 1 or args.checkpoint_freq < 1: p.error("step counts must be positive")
        source = json.loads(args.config.read_text(encoding="utf-8"))
        options = LearningOptions(horizon=args.horizon, gamma=source["spec"]["gamma"],
            learning_rate=args.learning_rate, max_parameter_change=args.max_parameter_change,
            exploration_strength=args.exploration_strength, max_iterations=args.max_iterations,
            max_model_evaluations=args.max_model_evaluations)
        parameters = None
    else:
        if args.episodes < 1: p.error("episode count must be positive")
        saved = read_checkpoint(args.checkpoint)
        source, parameters = saved["source_config"], saved["parameters"]
        options = LearningOptions(**saved["options"])
        if args.tolerance_m is None: args.tolerance_m = saved["tolerance_m"]
        if args.max_steps is None: args.max_steps = saved["max_steps"]
    if args.max_steps < 1: p.error("max steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Output directory is not empty")
    # Explicit fixed tolerance for this pilot, also used for evaluation.
    env = make_env(source, evaluation=True, compute_jacobian=False, tolerance=args.tolerance_m)
    env.max_steps_per_episode = args.max_steps
    started = time.perf_counter()
    learner = MPCQLearner.from_env(env, source, options, parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = dict(format="ctr-mpcrl-run-v1", mode=args.mode, source_config=source,
        options=asdict(options), seed=args.seed, tolerance_m=args.tolerance_m, max_steps=args.max_steps,
        frozen_parameters=args.freeze if training else True,
        learning_cost="negative original environment reward; 1 outside tolerance, 0 inside",
        actor="parametric nonlinear MPC", learning="mpcrl first-order semi-gradient Q-learning",
        exploration="linear perturbation inside constrained MPC objective; zero in Q and TD target V",
        terminal_semantics="zero bootstrap on success; retain bootstrap on time-limit truncation",
        training_tolerance="constant", demonstrations=False, her=False,
        physical_stability_guarantee=False, robust_safety_guarantee=False,
        versions={name: version(name) for name in ("numpy", "scipy", "casadi", "csnlp", "mpcrl", "gymnasium")},
        python=platform.python_version())
    write_json(args.output_dir/"config.json", configuration)
    if training:
        write_json(args.output_dir/"checkpoint_initial.json", checkpoint(source, learner, args.tolerance_m, args.max_steps, 0))
    rng = np.random.default_rng(args.seed)
    rows, steps, episode, failure, interrupted = [], 0, 0, "", False
    with (args.output_dir/"updates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        try:
            while (steps < args.steps) if training else (episode < args.episodes):
                actions, deltas = [], []
                episode_steps, success, error, trivial, task_hash = 0, False, None, False, ""
                ep_failure, completed = "", False
                try:
                    _, info = env.reset(seed=args.seed+episode)
                    task_hash = hashlib.sha256(np.r_[env.trig_obj.joints, env.desired_goal].astype("<f8").tobytes()).hexdigest()
                    trivial = bool(info["is_success"])
                    while episode_steps < args.max_steps and (not training or steps < args.steps):
                        state = learner.state(env.trig_obj.joints, env.desired_goal)
                        perturbation = rng.normal(size=6)*options.exploration_strength if training else np.zeros(6)
                        step_started = time.perf_counter(); before_calls = learner.callback.calls
                        action, sol_policy, policy_metrics = learner.solve(state, perturbation=perturbation)
                        learner.check_executed_action(env, action, sol_policy)
                        learn = training and not args.freeze
                        sol_q = learner.solve(state, action=action)[1] if learn else None
                        q_before = env.trig_obj.joints.copy()
                        _, reward, terminated, truncated, info = env.step(action)
                        steps += 1; episode_steps += 1
                        if info["solver_failure"]: raise RuntimeError("Original IVP execution failed")
                        error = float(info["error"])
                        actions.append(action.copy()); deltas.append(env.trig_obj.joints-q_before)
                        metrics = dict(td_error=None, gradient_norm=None, parameter_change_norm=None)
                        next_value = None
                        if learn:
                            next_state = learner.state(env.trig_obj.joints, env.desired_goal)
                            sol_v = None if terminated else learner.solve(next_state)[1]
                            next_value = 0. if terminated else float(sol_v.f)
                            metrics = learner.learn_transition(-reward, sol_q, sol_v, terminated=terminated)
                        record = dict(timestep=steps, episode=episode, error_m=error, reward=float(reward),
                            terminated=bool(terminated), truncated=bool(truncated),
                            policy_status=policy_metrics["status"], q_value=float(sol_q.f) if learn else None,
                            next_value=next_value, **metrics, model_calls=learner.callback.calls-before_calls,
                            elapsed_step_s=time.perf_counter()-step_started,
                            parameters_json=json.dumps(learner.parameters()))
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(record)); writer.writeheader()
                        writer.writerow(record); stream.flush()
                        if training and steps % args.checkpoint_freq == 0:
                            write_json(args.output_dir/f"checkpoint_{steps:09d}.json",
                                checkpoint(source, learner, args.tolerance_m, args.max_steps, steps))
                        if steps % args.progress_every == 0:
                            print(f"Step {steps}: error={1000*error:.3f} mm, updates={learner.updates}, "
                                  f"model calls={learner.callback.calls}, elapsed={time.perf_counter()-started:.0f} s", flush=True)
                        if terminated or truncated:
                            success, completed = bool(info["is_success"]), True
                            break
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    ep_failure = f"{type(exc).__name__}: {exc}"; error = None
                    if training: failure = ep_failure
                rows.append(dict(episode=episode, seed=args.seed+episode, task_fingerprint=task_hash,
                    success=success, steps=episode_steps, final_error_m=error,
                    initially_within_tolerance=trivial, failure=ep_failure, episode_complete=completed,
                    **motion_metrics(actions, deltas)))
                with (args.output_dir/"episodes.csv").open("w", newline="", encoding="utf-8") as f:
                    ew = csv.DictWriter(f, fieldnames=list(rows[0])); ew.writeheader(); ew.writerows(rows)
                episode += 1
                if failure: break
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if training:
                write_json(args.output_dir/"checkpoint_final.json",
                    checkpoint(source, learner, args.tolerance_m, args.max_steps, steps))
            env.close()
    successes = sum(r["success"] for r in rows)
    errors = [r["final_error_m"] for r in rows if r["final_error_m"] is not None]
    summary = dict(complete=not failure and not interrupted and (steps == args.steps if training else episode == args.episodes),
        mode=args.mode, timesteps=steps, updates=learner.updates, episodes=len(rows), successes=successes,
        success_rate=successes/len(rows) if rows else None,
        success_rate_wilson_95=wilson_interval(successes, len(rows)) if rows else None,
        failures=sum(bool(r["failure"]) for r in rows), abort_reason=failure, interrupted=interrupted,
        partial_episodes=sum(not r["episode_complete"] and not r["failure"] for r in rows),
        metric_scope="training includes final partial episode; frozen evaluation uses full episodes",
        mean_final_error_m=float(np.mean(errors)) if errors else None,
        median_final_error_m=float(np.median(errors)) if errors else None,
        p95_final_error_m=float(np.percentile(errors, 95)) if errors else None,
        mean_steps=float(np.mean([r["steps"] for r in rows])) if rows else None,
        environment_fingerprint=source["environment_fingerprint"], first_seed=args.seed,
        tolerance_m=args.tolerance_m, max_steps=args.max_steps, horizon=options.horizon,
        parameters=learner.parameters(), optimizer_solves=learner.callback.solves,
        prediction_model_calls=learner.callback.calls, prediction_seconds=learner.callback.seconds,
        environment_costs=env.costs, elapsed_seconds=time.perf_counter()-started,
        physical_stability_guarantee=False, robust_safety_guarantee=False)
    write_json(args.output_dir/"summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
