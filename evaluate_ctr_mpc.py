"""Evaluate nonlinear MPC on the original IVP plant or paper free-tip BVP plant."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ctr_reach_envs.ivp.config import make_env, fingerprint
from ctr_reach_envs.envs.model import Model
from ctr_reach_envs.mechanics.geometry import TubeParameters, JointConstraints
from ctr_reach_envs.mechanics.solver import EquilibriumSolver, SolverOptions
from ctr_reach_envs.mpc import NonlinearMPC, MPCOptions, OriginalIVPModel, PaperIVPModel
from ctr_reach_envs.mpc.models import original_joint_step


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def motion_metrics(actions, deltas):
    # Same mean-episode RMS definitions as evaluate_original_ddpg_her.py.
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else 0.
    a, dq = np.asarray(actions).reshape(-1, 6), np.asarray(deltas).reshape(-1, 6)
    da, ddq = np.diff(a, axis=0), np.diff(dq, axis=0)
    return dict(action_change_rms=rms(da), extension_increment_rms_m=rms(dq[:, :3]),
        rotation_increment_rms_rad=rms(dq[:, 3:]), extension_increment_change_rms_m=rms(ddq[:, :3]),
        rotation_increment_change_rms_rad=rms(ddq[:, 3:]))


def wilson_interval(successes, count):
    z = 1.959963984540054
    p = successes/count
    den = 1+z*z/count
    center = (p+z*z/(2*count))/den
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [float(center-half), float(center+half)]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="An original-IVP run/checkpoint config.json")
    p.add_argument("--plant", choices=("original_ivp", "paper_bvp"), default="original_ivp")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=920000)
    p.add_argument("--tolerance-m", type=float, default=.0015)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--max-model-evaluations", type=int, default=400)
    p.add_argument("--tracking-scale-m", type=float, default=.01)
    p.add_argument("--move-weight", type=float, default=.001)
    p.add_argument("--terminal-weight", type=float, default=1.)
    p.add_argument("--finite-difference-step", type=float, default=1e-4)
    p.add_argument("--boundary-tolerance", type=float, default=1e-5, help="Max |L * distal strain| in the paper NLP")
    p.add_argument("--base-separation-m", type=float, help="Paper Eq. 14 base clearance; default 1e-5 for paper_bvp, 0 for original_ivp")
    p.add_argument("--branch-tip-tolerance-m", type=float, default=1e-4,
                   help="Paper mode: reject disagreement with independent continuation BVP")
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    if min(a.episodes, a.max_steps, a.progress_every) < 1:
        p.error("Counts must be positive")
    if not np.isfinite(a.branch_tip_tolerance_m) or a.branch_tip_tolerance_m <= 0:
        p.error("Branch tip tolerance must be finite and positive")
    separation = a.base_separation_m
    if separation is None: separation = 1e-5 if a.plant == "paper_bvp" else 0.
    if a.plant == "original_ivp" and separation:
        p.error("Keep --base-separation-m 0 for matched original-IVP tasks")
    options = MPCOptions(horizon=a.horizon, max_iterations=a.max_iterations,
        max_model_evaluations=a.max_model_evaluations, tracking_scale_m=a.tracking_scale_m,
        move_weight=a.move_weight, terminal_weight=a.terminal_weight,
        finite_difference_step=a.finite_difference_step, boundary_tolerance=a.boundary_tolerance,
        base_separation_m=separation)
    config = json.loads(a.config.read_text(encoding="utf-8"))
    if config.get("segment_mode") != "continuous":
        p.error("MPC requires an existing continuous original-IVP config; do not silently change a legacy run")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error("Output directory is not empty; choose a new directory")
    env = make_env(config, evaluation=True, compute_jacobian=False, tolerance=a.tolerance_m)
    if env.observation_uses_jacobian:
        env.close()
        p.error("Use a physics-observation=none config for this actor-free MPC comparison")
    env.max_steps_per_episode = a.max_steps
    lengths = env.trig_obj.tube_lengths[0]
    constraints = JointConstraints(lengths, constrain_alpha=env.trig_obj.constrain_alpha)
    caps = env.n_substeps*env.action_scale
    equilibrium = None
    solver = None
    truth_calls, truth_seconds = 0, 0.
    if a.plant == "paper_bvp":
        solver = EquilibriumSolver([TubeParameters.from_legacy(t) for t in env.ctr_system_parameters[0]],
            SolverOptions(shooting_strategy="hybr_restarts", max_shooting_evaluations=500,
                          max_restart_evaluations=300))
        model = PaperIVPModel(solver)
    else:
        model = OriginalIVPModel(Model(env.ctr_system_parameters, **config["environment"]["model_options"]))
    controller = NonlinearMPC(model, constraints, caps, options)

    def solve_truth(q, base=None):
        nonlocal truth_calls, truth_seconds
        started = time.perf_counter(); truth_calls += 1
        try:
            return solver.solve(q, initial_torsion=base)
        finally:
            truth_seconds += time.perf_counter()-started

    plant_contract = dict(plant=a.plant, source_environment=config["environment"],
        solver_options=asdict(solver.options) if solver else None,
        base_separation_m=separation, branch_tip_tolerance_m=a.branch_tip_tolerance_m,
        branch_selection="previous measured base torsion; independent BVP check" if solver else None)
    environment_hash = (fingerprint(plant_contract) if solver else config["environment_fingerprint"])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(controller="nonlinear_mpc", version=1, source_config=config,
        source_config_path=str(a.config), plant_contract=plant_contract,
        environment_fingerprint=environment_hash, optimizer=asdict(options),
        max_prediction_rhs=getattr(model, "max_prediction_rhs", None),
        evaluation=dict(episodes=a.episodes, first_seed=a.seed, max_steps=a.max_steps, tolerance_m=a.tolerance_m),
        joint_step_caps=caps.tolist(), action_semantics="normalized original proposal repeated n_substeps",
        paper="https://doi.org/10.1109/TRO.2020.2991651",
        reference_code="https://github.com/RViMLab/TRO2020-CTR-Model-Predictive-Control/tree/a9cf2a880a338a7141acf11d4419997a4c5f235b",
        goal_sampling="original seeded joint sampler; endpoints mapped through selected plant",
        terminal_critic=False, trained_actor=False, physical_stability_guarantee=False)
    write_json(a.output_dir/"config.json", run_config)
    rows, times, holds, decisions = [], [], 0, 0
    started, interrupted = time.perf_counter(), False
    with (a.output_dir/"controller.csv").open("w", newline="", encoding="utf-8") as trace:
        writer = None
        try:
            for episode in range(a.episodes):
                controller.reset()
                actions, deltas, failure = [], [], ""
                steps, success, trivial, error, task_hash = 0, False, False, None, ""
                try:
                    _, info = env.reset(seed=a.seed+episode)
                    q, goal, tip = env.trig_obj.joints.copy(), env.desired_goal.copy(), env.achieved_goal.copy()
                    if solver:
                        equilibrium = solve_truth(q)
                        goal = solve_truth(env.desired_joints).tip
                        tip = equilibrium.tip
                    task_hash = hashlib.sha256(np.r_[q, goal].astype("<f8").tobytes()).hexdigest()
                    trivial = bool(np.linalg.norm(tip-goal) <= a.tolerance_m)
                    while steps < a.max_steps:
                        z = equilibrium.base_torsional_strain*solver.scale if solver else None
                        result = controller.plan(q, tip, goal, torsion_scaled=z)
                        action = np.clip((result.joints-q)/caps, -1., 1.).astype(np.float32)
                        actual_q = original_joint_step(q, action, env.action_scale, env.n_substeps,
                                                       lengths, constraints.constrain_alpha)
                        if np.max(abs((actual_q-result.joints)/caps)) > 1e-6:
                            raise RuntimeError("Planned first joint command cannot be executed by the plant")
                        steps += 1
                        decisions += 1
                        holds += int(result.diagnostics["hold"])
                        times.append(result.diagnostics["planning_seconds"])
                        record = dict(episode=episode, step=steps, seed=a.seed+episode,
                            error_before_m=float(np.linalg.norm(tip-goal)), **result.diagnostics,
                            prediction_error_m=None, error_after_m=None, execution_failure="")
                        try:
                            if solver:
                                # The optimizer's torsion is NEVER used to initialize the plant root.
                                next_equilibrium = solve_truth(actual_q, equilibrium.base_torsional_strain)
                                next_tip = next_equilibrium.tip
                                disagreement = float(np.linalg.norm(next_tip-result.predicted_tip))
                                if disagreement > a.branch_tip_tolerance_m:
                                    raise RuntimeError(f"MPC/continuation BVP branch mismatch: {disagreement:.6g} m")
                                equilibrium = next_equilibrium
                                terminated = np.linalg.norm(next_tip-goal) <= a.tolerance_m
                                truncated = steps >= a.max_steps
                            else:
                                _, _, terminated, truncated, info = env.step(action)
                                if info["solver_failure"]:
                                    raise RuntimeError("Original IVP execution failed")
                                actual_q, next_tip = env.trig_obj.joints.copy(), env.achieved_goal.copy()
                            record["prediction_error_m"] = float(np.linalg.norm(next_tip-result.predicted_tip))
                            deltas.append(actual_q-q); actions.append(action.copy())
                            q, tip = actual_q, next_tip
                            error = float(np.linalg.norm(tip-goal))
                            record["error_after_m"] = error
                        except (RuntimeError, ValueError, FloatingPointError) as exc:
                            record["execution_failure"] = f"{type(exc).__name__}: {exc}"
                            raise
                        finally:
                            if writer is None:
                                writer = csv.DictWriter(trace, fieldnames=list(record)); writer.writeheader()
                            writer.writerow(record); trace.flush()
                        if terminated or truncated:
                            success = bool(terminated)
                            break
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"; error = None
                rows.append(dict(episode=episode, seed=a.seed+episode, task_fingerprint=task_hash,
                    success=success, steps=steps, final_error_m=error, initially_within_tolerance=trivial,
                    failure=failure, **motion_metrics(actions, deltas)))
                with (a.output_dir/"episodes.csv").open("w", newline="", encoding="utf-8") as f:
                    ew = csv.DictWriter(f, fieldnames=list(rows[0])); ew.writeheader(); ew.writerows(rows)
                if (episode+1) % a.progress_every == 0:
                    elapsed = time.perf_counter()-started
                    print(f"MPC {episode+1}/{a.episodes}: success={np.mean([r['success'] for r in rows]):.1%}, "
                          f"holds={holds}/{decisions}, elapsed={elapsed:.0f} s", flush=True)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            env.close()
    successes = sum(r["success"] for r in rows)
    errors = [r["final_error_m"] for r in rows if r["final_error_m"] is not None]
    summary = dict(complete=len(rows) == a.episodes and not interrupted, requested_episodes=a.episodes,
        episodes=len(rows), successes=successes, success_rate=successes/len(rows) if rows else None,
        success_rate_wilson_95=wilson_interval(successes, len(rows)) if rows else None,
        failures=sum(bool(r["failure"]) for r in rows), trivial_goals=sum(r["initially_within_tolerance"] for r in rows),
        mean_final_error_m=float(np.mean(errors)) if errors else None,
        median_final_error_m=float(np.median(errors)) if errors else None,
        p95_final_error_m=float(np.percentile(errors, 95)) if errors else None,
        error_aggregation="final states of nonfailed episodes; failures remain in success-rate denominator",
        mean_steps=float(np.mean([r["steps"] for r in rows])) if rows else None,
        motion_metrics={k: float(np.mean([r[k] for r in rows])) for k in motion_metrics([], [])} if rows else {},
        motion_scope="mean episode RMS; compare with success and episode length; no physical timestep",
        first_seed=a.seed, max_steps=a.max_steps, tolerance_m=a.tolerance_m, plant=a.plant,
        environment_fingerprint=environment_hash, source_environment_fingerprint=config["environment_fingerprint"],
        deterministic_controller=True, controller="nonlinear_mpc", horizon=a.horizon,
        decisions=decisions, holds=holds, hold_fraction=holds/decisions if decisions else None,
        planning_seconds_median=float(np.median(times)) if times else None,
        planning_seconds_p95=float(np.percentile(times, 95)) if times else None,
        prediction_costs=dict(ivp_calls=model.calls, seconds=model.seconds,
                              rhs_evaluations=getattr(model, "rhs_evaluations", None)),
        bvp_plant_costs=dict(shooting_calls=truth_calls, seconds=truth_seconds),
        original_environment_costs=env.costs, elapsed_seconds=time.perf_counter()-started,
        physical_stability_guarantee=False)
    write_json(a.output_dir/"summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
