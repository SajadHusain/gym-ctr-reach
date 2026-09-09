"""Audit explicit branch continuation without training an RL policy.

Known negative roots are rejected at initialization, then their joint targets
are approached from aligned references. A recorded stop is distinct from target
completion and is never replaced with an easier case. Seeded finite commands
also test new configurations in all four repository systems.
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
                                    BranchTracker, BranchInitializationError, TrackingOptions, elastic_stability)
from validate_mechanics import sample_joints


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-system", type=int, default=2)
    parser.add_argument("--negative-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7003)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/step3_validation"))
    args = parser.parse_args()
    if args.samples_per_system < 1 or args.seed < 0 or args.max_steps < 1 or not 1 <= args.negative_cases <= 5:
        parser.error("Use positive sample/step counts, nonnegative seed, and 1 to 5 negative cases")
    return args


def solver_for(system, options=None):
    return EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS[system].values()], options)


def assert_transition(solver, before, result, options):
    """Check the public transition contract independently of tracker mutation."""
    after = result.state
    np.testing.assert_allclose(after.equilibrium.joints-before.equilibrium.joints, result.applied_delta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.applied_delta, result.accepted_fraction*result.projected_delta, rtol=0, atol=1e-12)
    if not solver.constraints.is_feasible(after.equilibrium.joints):
        raise AssertionError("Returned state is infeasible")
    if result.status != "accepted":
        for field in ("joints", "base_torsional_strain", "position", "angles"):
            np.testing.assert_array_equal(getattr(before.equilibrium, field), getattr(after.equilibrium, field))
        np.testing.assert_array_equal(before.sensitivity.tip_jacobian, after.sensitivity.tip_jacobian)
        if before.accepted_steps != after.accepted_steps or before.stability.minimum_eigenvalues != after.stability.minimum_eigenvalues:
            raise AssertionError("Rejected move changed the cached state")
        return
    caps = np.r_[np.full(solver.n, options.max_translation_step_m), np.full(solver.n, options.max_rotation_step_rad)]
    if np.max(abs(result.applied_delta)/caps) > 1+1e-10:
        raise AssertionError("Accepted increment exceeds cap")
    attempt = result.diagnostics["attempts"][-1]
    if not attempt["accepted"] or len(attempt["checkpoints"]) != 2:
        raise AssertionError("Missing accepted midpoint or endpoint")
    if after.accepted_steps != before.accepted_steps+1:
        raise AssertionError("Accepted-step counter mismatch")
    for checkpoint in attempt["checkpoints"]:
        if (checkpoint["elastic_status"] != "positive_second_variation_on_tested_meshes"
                or checkpoint["minimum_eigenvalue"] <= options.minimum_elastic_eigenvalue):
            raise AssertionError("Accepted checkpoint lacks the elastic margin")
        for name in ("local_prediction", "overall_prediction"):
            values = checkpoint[name]
            if (values["tip_error_m"] > values["tip_limit_m"]
                    or values["twist_prediction_error"] > values["twist_prediction_limit"]
                    or values["twist_change"] > options.max_twist_change):
                raise AssertionError("Accepted checkpoint exceeds a prediction bound")
    # Recompute the full-increment prediction from the actual before/after data.
    tip_error = np.linalg.norm(after.equilibrium.tip-before.equilibrium.tip-before.sensitivity.tip_jacobian@result.applied_delta)
    tip_limit = options.tip_absolute_tolerance_m+options.tip_relative_tolerance*np.linalg.norm(after.equilibrium.tip-before.equilibrium.tip)
    if tip_error > tip_limit:
        raise AssertionError("Independent tip prediction check failed")
    if (attempt["reverse_tip_error_m"] > options.reverse_tip_tolerance_m
            or attempt["reverse_twist_error"] > options.reverse_twist_tolerance):
        raise AssertionError("Accepted reverse check exceeds tolerance")


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    options = TrackingOptions()
    refined_options = SolverOptions(rtol=1e-11, atol=1e-13, max_step=.002)
    baseline = json.loads((Path(__file__).resolve().parent/"docs/validation/step2_seed7002.json").read_text())
    negatives = [c for c in baseline["cases"] if c["stability"]["eigenvalues"][-1] < 0][:args.negative_cases]
    cases = [{"kind": "negative_regression", "system": c["system"], "source_sample": c["sample"],
              "target_q": c["q"]} for c in negatives]
    for name in CTR_SYSTEMS_PARAMETERS:
        solver = solver_for(name)
        for i in range(args.samples_per_system):
            q = sample_joints(solver, rng)
            q[solver.n:] = 0.
            delta = rng.uniform(-1., 1., 2*solver.n)*np.r_[np.full(solver.n, .004), np.full(solver.n, .15)]
            cases.append({"kind": "seeded_command", "system": name, "source_sample": i,
                          "initial_q": q.tolist(), "command": delta.tolist()})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    rows, details = [], []
    started = time.perf_counter()

    def save(complete):
        with (args.output_dir/"cases.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (args.output_dir/"details.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
        neg = [r for r in rows if r["kind"] == "negative_regression"]
        summary = {"complete": complete, "cases": len(rows), "contract_passed": sum(r["contract_passed"] for r in rows),
                   "contract_failed": sum(not r["contract_passed"] for r in rows),
                   "negative_roots_rejected": sum(r["negative_root_rejected"] is True for r in neg),
                   "negative_targets_reached": sum(r["target_reached"] for r in neg),
                   "negative_targets_tested": len(neg),
                   "path_status_counts": dict(Counter(r["path_status"] for r in rows)),
                   "accepted_moves": sum(r["accepted_moves"] for r in rows),
                   "rejected_trials": sum(r["rejected_trials"] for r in rows),
                   "tracking_equilibrium_calls": sum(r["tracking_equilibrium_calls"] for r in rows),
                   "tracking_sensitivity_calls": sum(r["tracking_sensitivity_calls"] for r in rows),
                   "tracking_stability_calls": sum(r["tracking_stability_calls"] for r in rows),
                   "external_equilibrium_calls": sum(r["external_equilibrium_calls"] for r in rows),
                   "external_sensitivity_calls": sum(r["external_sensitivity_calls"] for r in rows),
                   "external_stability_calls": sum(r["external_stability_calls"] for r in rows),
                   "total_equilibrium_calls": sum(r["external_equilibrium_calls"]+r["tracking_equilibrium_calls"] for r in rows),
                   "failed_calls_without_rhs_counts": sum(r["failed_calls_without_rhs_counts"] for r in rows),
                   "seed": args.seed, "samples_per_system": args.samples_per_system, "max_steps": args.max_steps,
                   "tracking_options": asdict(options), "solver_options": asdict(SolverOptions()),
                   "refined_options": asdict(refined_options), "total_elapsed_seconds": time.perf_counter()-started,
                   "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "git_commit": commit, "elastic_stability_certified": False,
                   "continuous_path_certified": False, "controller_stability_certified": False,
                   "global_uniqueness_certified": False,
                   "contract_pass_definition": "accepted samples satisfy numerical checks; rejected moves preserve state; known negative roots refused; refined final equilibrium agrees",
                   "target_completion_required_for_contract_pass": False}
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary

    with (args.output_dir/"transitions.jsonl").open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases):
            solver = solver_for(case["system"])
            row = {"case": index, "kind": case["kind"], "system": case["system"], "source_sample": case["source_sample"],
                   "contract_passed": False, "failure": "", "negative_root_rejected": None,
                   "target_reached": False, "path_status": "not_started", "stop_reason": "",
                   "accepted_moves": 0, "rejected_trials": 0, "tracking_equilibrium_calls": 0,
                   "tracking_sensitivity_calls": 0, "tracking_stability_calls": 0,
                   "external_equilibrium_calls": 0, "failed_calls_without_rhs_counts": 0,
                   "external_sensitivity_calls": 0, "external_stability_calls": 0,
                   "maximum_accepted_tip_error_m": 0., "final_minimum_eigenvalue": None,
                   "final_joint_target_error_scaled": None, "refinement_tip_error_m": None,
                   "final_twist_difference_from_cold": None, "seconds": 0.}
            detail = {**case, "external_calls": []}
            tracker, cold = None, None
            t = time.perf_counter()
            try:
                caps = np.r_[np.full(solver.n, options.max_translation_step_m), np.full(solver.n, options.max_rotation_step_rad)]
                if case["kind"] == "negative_regression":
                    target = np.array(case["target_q"])
                    row["external_equilibrium_calls"] += 1
                    cold = solver.solve(target)
                    detail["external_calls"].append({"kind": "cold_negative_root", **cold.diagnostics})
                    try:
                        unexpected = BranchTracker(solver, cold, options)
                        row["external_stability_calls"] += unexpected.totals["stability_calls"]
                        row["external_sensitivity_calls"] += unexpected.totals["sensitivity_calls"]
                    except BranchInitializationError as exc:
                        detail["negative_initialization"] = exc.diagnostics
                        row["external_stability_calls"] += exc.diagnostics["stability_calls"]
                        row["external_sensitivity_calls"] += exc.diagnostics["sensitivity_calls"]
                        row["failed_calls_without_rhs_counts"] += exc.diagnostics["failed_calls_without_rhs_counts"]
                        row["negative_root_rejected"] = "negative_second_variation" in str(exc)
                    if row["negative_root_rejected"] is not True:
                        raise AssertionError("Known negative equilibrium was not explicitly rejected")
                    initial = target.copy(); initial[solver.n:] = 0.
                else:
                    initial = np.array(case["initial_q"])
                    target = solver.constraints.project(initial+case["command"])
                row["external_equilibrium_calls"] += 1
                root = solver.solve(initial)
                detail["initial_q"] = initial.tolist()
                detail["external_calls"].append({"kind": "aligned_reference", **root.diagnostics})
                try:
                    tracker = BranchTracker(solver, root, options)
                except BranchInitializationError as exc:
                    detail["failed_initialization"] = exc.diagnostics
                    row["external_stability_calls"] += exc.diagnostics["stability_calls"]
                    row["external_sensitivity_calls"] += exc.diagnostics["sensitivity_calls"]
                    row["failed_calls_without_rhs_counts"] += exc.diagnostics["failed_calls_without_rhs_counts"]
                    raise
                detail["initialization"] = tracker.initialization_diagnostics
                steps = args.max_steps if case["kind"] == "negative_regression" else 1
                for step in range(steps):
                    before = tracker.state
                    command = target-before.equilibrium.joints if case["kind"] == "negative_regression" else np.array(case["command"])
                    result = tracker.step(command)
                    assert_transition(solver, before, result, options)
                    row["accepted_moves"] += int(result.status == "accepted")
                    row["rejected_trials"] += sum(not a["accepted"] for a in result.diagnostics["attempts"])
                    if result.status == "accepted":
                        values = result.diagnostics["attempts"][-1]["checkpoints"]
                        row["maximum_accepted_tip_error_m"] = max(row["maximum_accepted_tip_error_m"],
                                                                  *(v["overall_prediction"]["tip_error_m"] for v in values))
                    record = {"case": index, "step": step, "status": result.status,
                              "requested_delta": result.requested_delta.tolist(), "applied_delta": result.applied_delta.tolist(),
                              "accepted_fraction": result.accepted_fraction, "q": result.state.equilibrium.joints.tolist(),
                              "base_torsion": result.state.equilibrium.base_torsional_strain.tolist(),
                              "diagnostics": result.diagnostics}
                    stream.write(json.dumps(record, allow_nan=False)+"\n"); stream.flush()
                    row["final_joint_target_error_scaled"] = float(np.max(abs(target-result.state.equilibrium.joints)/caps))
                    row["target_reached"] = row["final_joint_target_error_scaled"] < 1e-8
                    if row["target_reached"]:
                        row["path_status"] = "target_reached"
                        break
                    if result.status != "accepted":
                        row.update(path_status="held", stop_reason=result.diagnostics["reason"])
                        break
                    if case["kind"] == "seeded_command":
                        row["path_status"] = "partial_command_accepted"
                    else:
                        row["path_status"] = "step_budget_exhausted"
                    if (step+1) % 10 == 0:
                        print(f"Case {index+1}/{len(cases)} {case['system']}: {step+1} accepted moves, remaining={row['final_joint_target_error_scaled']:.3g} capped steps", flush=True)
                final = tracker.state
                row["final_minimum_eigenvalue"] = float(final.stability.minimum_eigenvalues[-1])
                detail["final_q"] = final.equilibrium.joints.tolist()
                detail["final_base_torsion"] = final.equilibrium.base_torsional_strain.tolist()
                detail["final_elastic_energy_j"] = final.equilibrium.elastic_energy_j
                if cold is not None:
                    row["final_twist_difference_from_cold"] = float(solver.scale*np.linalg.norm(final.equilibrium.base_torsional_strain-cold.base_torsional_strain))
                    detail["cold_elastic_energy_j"] = cold.elastic_energy_j
                reference = solver_for(case["system"], refined_options)
                row["external_equilibrium_calls"] += 1
                refined = reference.solve(final.equilibrium.joints, initial_torsion=final.equilibrium.base_torsional_strain)
                detail["external_calls"].append({"kind": "refined_final_root", **refined.diagnostics})
                row["external_stability_calls"] += 1
                refined_stability = elastic_stability(reference, refined)
                detail["refined_stability"] = refined_stability.diagnostics
                row["refinement_tip_error_m"] = float(np.linalg.norm(refined.tip-final.equilibrium.tip))
                if row["refinement_tip_error_m"] > 2e-6 or refined_stability.status != "positive_second_variation_on_tested_meshes":
                    raise AssertionError("Final equilibrium failed tighter-solver validation")
                row["contract_passed"] = True
            except (EquilibriumError, ValueError, AssertionError, np.linalg.LinAlgError) as exc:
                row["failure"] = f"{type(exc).__name__}: {exc}"
            finally:
                row["seconds"] = time.perf_counter()-t
                if tracker is not None:
                    counters = tracker.totals
                    for kind in ("equilibrium", "sensitivity", "stability"):
                        row["tracking_"+kind+"_calls"] = counters[kind+"_calls"]
                    row["failed_calls_without_rhs_counts"] += counters["failed_calls_without_rhs_counts"]
                    detail["tracking_totals"] = counters
            rows.append(row); details.append(detail)
            save(False)
            print(f"Case {index+1}/{len(cases)} {case['system']}: {'PASS' if row['contract_passed'] else 'FAIL'}; "
                  f"{row['path_status']}; {row['stop_reason'] or row['failure']}", flush=True)
    summary = save(True)
    print(json.dumps(summary, indent=2))
    if summary["contract_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
