"""Local reaching regression for goal control, numerical decrease, and fallback.

Goals are endpoints of four declared feasible-branch witness commands from each
aligned start. This is a local controller test, not a trained-policy evaluation
or a whole-workspace success estimate. Every requested case is retained.
"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from ctr_reach_envs.mechanics import (TubeParameters, EquilibriumSolver, SolverOptions, EquilibriumError,
                                    BranchTracker, TrackingOptions, GoalController, ControlOptions, elastic_stability)
from validate_mechanics import sample_joints


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-system", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7004)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--tolerance-m", type=float, default=.001)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/step4_validation"))
    args = parser.parse_args()
    if args.episodes_per_system < 1 or args.seed < 0 or args.max_steps < 1:
        parser.error("Use positive episode/step counts and nonnegative seed")
    if not np.isfinite(args.tolerance_m) or args.tolerance_m <= 0:
        parser.error("Tolerance must be finite and positive")
    return args


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tracking_options, control_options = TrackingOptions(), ControlOptions(goal_tolerance_m=args.tolerance_m)
    refined_options = SolverOptions(rtol=1e-11, atol=1e-13, max_step=.002)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    rows, details = [], []
    started = time.perf_counter()

    def save(complete):
        with (args.output_dir/"episodes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (args.output_dir/"details.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
        nontrivial = [r for r in rows if r["initial_error_m"] is not None and r["initial_error_m"] > args.tolerance_m]
        summary = {"complete": complete, "episodes": len(rows), "contract_passed": sum(r["contract_passed"] for r in rows),
                   "contract_failed": sum(not r["contract_passed"] for r in rows),
                   "reached_goals": sum(r["success"] for r in rows), "nontrivial_goals": len(nontrivial),
                   "nontrivial_goals_reached": sum(r["success"] for r in nontrivial),
                   "status_counts": dict(Counter(r["status"] for r in rows)),
                   "accepted_moves": sum(r["accepted_moves"] for r in rows),
                   "fallback_moves": sum(r["fallback_moves"] for r in rows),
                   "witness_held_commands": sum(r["witness_held_commands"] for r in rows),
                   "maximum_positive_v_change_m2": max(r["maximum_positive_v_change_m2"] for r in rows),
                   "control_equilibrium_calls": sum(r["control_equilibrium_calls"] for r in rows),
                   "control_sensitivity_calls": sum(r["control_sensitivity_calls"] for r in rows),
                   "control_stability_calls": sum(r["control_stability_calls"] for r in rows),
                   "witness_equilibrium_calls": sum(r["witness_equilibrium_calls"] for r in rows),
                   "witness_sensitivity_calls": sum(r["witness_sensitivity_calls"] for r in rows),
                   "witness_stability_calls": sum(r["witness_stability_calls"] for r in rows),
                   "external_equilibrium_calls": sum(r["external_equilibrium_calls"] for r in rows),
                   "external_stability_calls": sum(r["external_stability_calls"] for r in rows),
                   "total_equilibrium_calls": sum(r["control_equilibrium_calls"]+r["witness_equilibrium_calls"]+r["external_equilibrium_calls"] for r in rows),
                   "failed_calls_without_rhs_counts": sum(r["failed_calls_without_rhs_counts"] for r in rows),
                   "seed": args.seed, "episodes_per_system": args.episodes_per_system,
                   "max_steps": args.max_steps, "tolerance_m": args.tolerance_m,
                   "tracking_options": asdict(tracking_options), "control_options": asdict(control_options),
                   "refined_options": asdict(refined_options), "git_commit": commit,
                   "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "total_elapsed_seconds": time.perf_counter()-started,
                   "goal_distribution": "four fixed witness commands from each seeded aligned start; held commands retained",
                   "policy_test": "alternating pure Jacobian and synthetic uphill proposals; no learned policy",
                   "decrease_scope": "computed V=0.5*tip_error^2 for a fixed goal, on accepted samples; holds preserve state",
                   "global_convergence_certified": False, "controller_stability_certified": False,
                   "elastic_stability_certified": False, "rl_training_performed": False,
                   "success_required_for_contract_pass": False}
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary

    with (args.output_dir/"transitions.jsonl").open("w", encoding="utf-8") as stream:
        for name, system in CTR_SYSTEMS_PARAMETERS.items():
            tubes = [TubeParameters(**t) for t in system.values()]
            solver = EquilibriumSolver(tubes)
            for episode in range(args.episodes_per_system):
                row = {"system": name, "episode": episode, "contract_passed": False, "failure": "",
                       "proposal_mode": "jacobian" if episode % 2 == 0 else "synthetic_uphill",
                       "status": "not_started", "success": False, "initial_error_m": None, "final_error_m": None,
                       "steps": 0, "accepted_moves": 0, "fallback_moves": 0, "witness_held_commands": 0,
                       "control_equilibrium_calls": 0, "control_sensitivity_calls": 0, "control_stability_calls": 0,
                       "witness_equilibrium_calls": 0, "witness_sensitivity_calls": 0, "witness_stability_calls": 0,
                       "external_equilibrium_calls": 0, "external_stability_calls": 0, "failed_calls_without_rhs_counts": 0,
                       "maximum_positive_v_change_m2": 0., "refinement_tip_error_m": None,
                       "refined_goal_error_m": None,
                       "final_minimum_eigenvalue": None, "seconds": 0.}
                detail = {"system": name, "episode": episode, "witness_commands": [], "witness_transitions": []}
                controller, witness = None, None
                t = time.perf_counter()
                try:
                    q = sample_joints(solver, rng); q[solver.n:] = 0.
                    row["external_equilibrium_calls"] += 1
                    initial = solver.solve(q)
                    detail["initial_q"] = q.tolist()
                    detail["initial_solver"] = initial.diagnostics
                    witness = BranchTracker(solver, initial, tracking_options)
                    for _ in range(4):
                        command = np.r_[np.full(solver.n, -.0005), .04+rng.uniform(-.015, .015, solver.n)]
                        detail["witness_commands"].append(command.tolist())
                        move = witness.step(command)
                        detail["witness_transitions"].append({"status": move.status, "applied_delta": move.applied_delta.tolist(),
                                                             "diagnostics": move.diagnostics})
                        row["witness_held_commands"] += int(move.status != "accepted")
                    goal = witness.state.equilibrium.tip
                    detail["goal_m"] = goal.tolist(); detail["goal_witness_q"] = witness.state.equilibrium.joints.tolist()
                    row["initial_error_m"] = float(np.linalg.norm(initial.tip-goal))
                    controller = GoalController(BranchTracker(solver, initial, tracking_options), control_options)
                    for step in range(args.max_steps):
                        before = controller.tracker.state
                        proposal = None
                        if row["proposal_mode"] == "synthetic_uphill":
                            scales = controller.joint_scales
                            uphill = (before.sensitivity.tip_jacobian*scales).T@(before.equilibrium.tip-goal)
                            proposal = scales*uphill/max(np.max(abs(uphill)), np.finfo(float).tiny)
                        result = controller.step(goal, policy_delta=proposal)
                        old_error, new_error = before.equilibrium.tip-goal, result.state.equilibrium.tip-goal
                        old_v, new_v = .5*float(old_error@old_error), .5*float(new_error@new_error)
                        row["maximum_positive_v_change_m2"] = max(row["maximum_positive_v_change_m2"], new_v-old_v)
                        if result.transition is None:
                            np.testing.assert_array_equal(before.equilibrium.position, result.state.equilibrium.position)
                            np.testing.assert_array_equal(before.equilibrium.base_torsional_strain, result.state.equilibrium.base_torsional_strain)
                        else:
                            slope = float(old_error@before.sensitivity.tip_jacobian@result.applied_delta)
                            if slope >= 0 or not new_v < old_v or new_v > old_v+control_options.armijo_fraction*slope:
                                raise AssertionError("Independent fixed-goal decrease check failed")
                            for point in result.transition.diagnostics["attempts"][-1]["checkpoints"]:
                                if point["minimum_eigenvalue"] <= tracking_options.minimum_elastic_eigenvalue:
                                    raise AssertionError("Accepted goal move failed the elastic margin")
                        np.testing.assert_allclose(result.state.equilibrium.joints-before.equilibrium.joints,
                                                   result.applied_delta, rtol=0, atol=1e-12)
                        row["steps"] = step+1
                        row["accepted_moves"] += int(result.transition is not None)
                        row["fallback_moves"] += int(result.diagnostics["fallback_used"])
                        row["status"] = result.status
                        row["final_error_m"] = float(np.linalg.norm(new_error))
                        record = {"system": name, "episode": episode, "step": step, "status": result.status,
                                  "action_source": result.action_source, "goal_m": goal.tolist(),
                                  "tip_before_m": before.equilibrium.tip.tolist(), "tip_after_m": result.state.equilibrium.tip.tolist(),
                                  "q_before": before.equilibrium.joints.tolist(), "q_after": result.state.equilibrium.joints.tolist(),
                                  "applied_delta": result.applied_delta.tolist(), "diagnostics": result.diagnostics}
                        stream.write(json.dumps(record, allow_nan=False)+"\n"); stream.flush()
                        if result.status != "moving":
                            break
                        if (step+1) % 10 == 0:
                            print(f"{name} episode {episode+1}: {step+1} moves, error={row['final_error_m']*1000:.3f} mm", flush=True)
                    if row["status"] == "moving":
                        row["status"] = "step_budget_exhausted"
                    row["success"] = row["status"] == "goal_reached"
                    final = controller.tracker.state
                    row["final_minimum_eigenvalue"] = float(final.stability.minimum_eigenvalues[-1])
                    reference = EquilibriumSolver(tubes, refined_options)
                    row["external_equilibrium_calls"] += 1
                    refined = reference.solve(final.equilibrium.joints, initial_torsion=final.equilibrium.base_torsional_strain)
                    detail["refined_solver"] = refined.diagnostics
                    row["external_stability_calls"] += 1
                    stable = elastic_stability(reference, refined)
                    detail["refined_stability"] = stable.diagnostics
                    row["refinement_tip_error_m"] = float(np.linalg.norm(refined.tip-final.equilibrium.tip))
                    row["refined_goal_error_m"] = float(np.linalg.norm(refined.tip-goal))
                    if row["refinement_tip_error_m"] > 2e-6 or stable.status != "positive_second_variation_on_tested_meshes":
                        raise AssertionError("Refined final equilibrium check failed")
                    if row["success"] and row["refined_goal_error_m"] > args.tolerance_m:
                        raise AssertionError("Claimed goal success failed tighter-solver verification")
                    detail["final_q"] = final.equilibrium.joints.tolist()
                    row["contract_passed"] = True
                except (EquilibriumError, ValueError, AssertionError, np.linalg.LinAlgError) as exc:
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                finally:
                    row["seconds"] = time.perf_counter()-t
                    if witness is not None:
                        detail["witness_totals"] = witness.totals
                        for kind in ("equilibrium", "sensitivity", "stability"):
                            row["witness_"+kind+"_calls"] = witness.totals[kind+"_calls"]
                        row["failed_calls_without_rhs_counts"] += witness.totals["failed_calls_without_rhs_counts"]
                    if controller is not None:
                        detail["control_totals"] = controller.tracker.totals
                        for kind in ("equilibrium", "sensitivity", "stability"):
                            row["control_"+kind+"_calls"] = controller.tracker.totals[kind+"_calls"]
                        row["failed_calls_without_rhs_counts"] += controller.tracker.totals["failed_calls_without_rhs_counts"]
                rows.append(row); details.append(detail); save(False)
                print(f"{name} episode {episode+1}/{args.episodes_per_system}: {'PASS' if row['contract_passed'] else 'FAIL'}; "
                      f"{row['status']}; error={row['final_error_m']} m; {row['failure']}", flush=True)
    summary = save(True)
    print(json.dumps(summary, indent=2))
    if summary["contract_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
