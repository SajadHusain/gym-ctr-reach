"""Seeded Step 2 audit: second variation, implicit Jacobian, and local prediction.

Run tests/test_mechanics_step2.py for the independent analytical stability cases.
This audit includes every sampled configuration, including failed or unstable
ones. It never replaces failures with easier samples. No RL policy is trained.
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
from ctr_reach_envs.mechanics import (TubeParameters, EquilibriumSolver, SolverOptions,
                                    EquilibriumError, equilibrium_sensitivity, elastic_stability)
from validate_mechanics import sample_joints


def finite_difference_reference(solver, equilibrium, sensitivity, normalized_step):
    """Full perturbed BVPs with an explicit first-order branch predictor.

    The branch-distance check is a numerical safeguard, not a uniqueness proof.
    Smaller h is used near constraints. No perturbed joints are projected.
    """
    n = solver.n
    q = equilibrium.joints
    qscale = np.r_[np.full(n, solver.scale), np.ones(n)]
    jacobian = np.zeros((3, 2*n))
    counters = {"equilibrium_solves": 0, "rhs_evaluations": 0, "shooting_evaluations": 0}
    steps, distances = [], []
    for i in range(2*n):
        h = normalized_step*qscale[i]
        if i < n:
            a = abs(solver.constraints.A[:, i])
            mask = a > 0
            slack = solver.constraints.b-solver.constraints.A@q[:n]
            h = min(h, .2*np.min(slack[mask]/a[mask]))
        ends = solver.lengths+q[:n]
        starts = ends-np.array([t.length_curved for t in solver.tubes])
        events = np.unique(np.r_[0., ends, starts[starts > 0]])
        if i < n:
            h = min(h, .1*np.min(np.diff(events)))
        steps.append(float(h))
        if h < 1e-10*qscale[i]:
            raise EquilibriumError("Insufficient feasible interior or event separation for a reliable central finite difference")
        tips = []
        for sign in [1., -1.]:
            dq = np.eye(2*n)[i]*(sign*h)
            guess = equilibrium.base_torsional_strain+sensitivity.base_torsion_jacobian@dq
            probe = solver.solve(q+dq, initial_torsion=guess)
            branch_distance = float(np.linalg.norm((probe.base_torsional_strain-guess)*solver.scale))
            distances.append(branch_distance)
            if branch_distance > .05:
                raise EquilibriumError("Perturbed BVP departed from the local branch predictor")
            tips.append(probe.tip)
            counters["equilibrium_solves"] += 1
            for key in ("rhs_evaluations", "shooting_evaluations"):
                counters[key] += probe.diagnostics[key]
        jacobian[:, i] = (tips[0]-tips[1])/(2*h)
    return jacobian, {**counters, "steps": steps, "maximum_branch_predictor_distance": max(distances)}


def local_predictions(solver, equilibrium, sensitivity, rng):
    q = equilibrium.joints
    n = solver.n
    direction = rng.normal(size=2*n)
    direction /= np.linalg.norm(direction)
    direction *= np.r_[np.full(n, solver.scale), np.ones(n)]*1e-3
    change = solver.constraints.A@direction[:n]
    slack = solver.constraints.b-solver.constraints.A@q[:n]
    toward_boundary = change > 0
    if np.any(toward_boundary):
        direction *= min(1., .2*np.min(slack[toward_boundary]/change[toward_boundary]))
    rows = []
    for fraction in [1., .5, .25]:
        dq = fraction*direction
        guess = equilibrium.base_torsional_strain+sensitivity.base_torsion_jacobian@dq
        probe = solver.solve(q+dq, initial_torsion=guess)
        branch_distance = float(np.linalg.norm((probe.base_torsional_strain-guess)*solver.scale))
        if branch_distance > .05:
            raise EquilibriumError("Local prediction probe departed from the branch predictor")
        actual = probe.tip-equilibrium.tip
        error = float(np.linalg.norm(actual-sensitivity.tip_jacobian@dq))
        rows.append({"fraction": fraction, "applied_dq": dq.tolist(), "error_m": error,
                     "actual_displacement_m": float(np.linalg.norm(actual)),
                     "rhs_evaluations": probe.diagnostics["rhs_evaluations"],
                     "shooting_evaluations": probe.diagnostics["shooting_evaluations"],
                     "branch_predictor_distance": branch_distance})
    return rows


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-system", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7002)
    parser.add_argument("--relative-jacobian-tolerance", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/step2_validation"))
    args = parser.parse_args()
    if args.samples_per_system < 1 or args.seed < 0:
        parser.error("Use a positive sample count and nonnegative seed")
    if not np.isfinite(args.relative_jacobian_tolerance) or args.relative_jacobian_tolerance <= 0:
        parser.error("Jacobian tolerance must be finite and positive")
    return args


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_rng = np.random.default_rng(args.seed)
    probe_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    standard = SolverOptions()
    refined = SolverOptions(rtol=1e-11, atol=1e-13, max_step=.002)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    rows, details = [], []
    started = time.perf_counter()
    def save(complete=False):
        with (args.output_dir/"cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output_dir/"details.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        errors = [r["relative_jacobian_error"] for r in rows if r["relative_jacobian_error"] is not None]
        column_errors = [r["maximum_column_relative_error"] for r in rows if r["maximum_column_relative_error"] is not None]
        summary = {"complete": complete, "samples": len(rows), "passed": sum(r["passed"] for r in rows),
                   "failed": sum(not r["passed"] for r in rows), "seed": args.seed,
                   "stability_status_counts": dict(Counter(r["stability_status"] for r in rows)),
                   "maximum_relative_jacobian_error": max(errors, default=None),
                   "maximum_column_relative_error": max(column_errors, default=None),
                   "relative_jacobian_tolerance": args.relative_jacobian_tolerance,
                   "mean_base_solve_seconds": float(np.mean([r["base_seconds"] for r in rows])),
                   "mean_sensitivity_seconds": float(np.mean([r["sensitivity_seconds"] for r in rows if r["sensitivity_seconds"] is not None])) if any(r["sensitivity_seconds"] is not None for r in rows) else None,
                   "mean_stability_seconds": float(np.mean([r["stability_seconds"] for r in rows if r["stability_seconds"] is not None])) if any(r["stability_seconds"] is not None for r in rows) else None,
                   "total_elapsed_seconds": time.perf_counter()-started,
                   "standard_solver": asdict(standard), "reference_solver": asdict(refined),
                   "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "git_commit": commit, "elastic_stability_certified": False,
                   "global_uniqueness_certified": False, "controller_stability_certified": False,
                   "sampled_prediction_errors_are_rigorous_bounds": False,
                   "pass_definition": "Jacobian agrees with full BVP finite differences at two step sizes; stability diagnostic completed"}
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return summary
    for name, system in CTR_SYSTEMS_PARAMETERS.items():
        tubes = [TubeParameters(**t) for t in system.values()]
        solver, reference = EquilibriumSolver(tubes, standard), EquilibriumSolver(tubes, refined)
        qscale = np.r_[np.full(solver.n, solver.scale), np.ones(solver.n)]
        for index in range(args.samples_per_system):
            q = sample_joints(solver, sample_rng)
            row = {"system": name, "sample": index, "passed": False, "failure": "",
                   "stability_status": "not_evaluated", "minimum_eigenvalue": None,
                   "shooting_condition_number": None, "relative_jacobian_error": None,
                   "maximum_column_relative_error": None,
                   "finite_difference_refinement_error": None, "base_seconds": 0.,
                   "sensitivity_seconds": None, "stability_seconds": None,
                   "reference_seconds": None, "largest_probe_error_m": None,
                   **{f"q{i}": float(v) for i, v in enumerate(q)}}
            detail = {"system": name, "sample": index, "q": q.tolist()}
            t = time.perf_counter()
            try:
                equilibrium = solver.solve(q)
                row["base_seconds"] = time.perf_counter()-t
                detail["base_solver"] = equilibrium.diagnostics
                stable = elastic_stability(solver, equilibrium)
                row.update(stability_status=stable.status, minimum_eigenvalue=stable.minimum_eigenvalues[-1],
                           stability_seconds=stable.diagnostics["seconds"])
                detail["stability"] = {**stable.diagnostics, "eigenvalues": stable.minimum_eigenvalues,
                                       "mesh_elements_per_segment": stable.mesh_elements_per_segment}
                sensitivity = equilibrium_sensitivity(solver, equilibrium)
                row.update(shooting_condition_number=sensitivity.diagnostics["shooting_condition_number"],
                           sensitivity_seconds=sensitivity.diagnostics["seconds"])
                detail["sensitivity"] = sensitivity.diagnostics
                detail["tip_jacobian"] = sensitivity.tip_jacobian.tolist()
                t = time.perf_counter()
                j1, d1 = finite_difference_reference(reference, equilibrium, sensitivity, 1e-5)
                j2, d2 = finite_difference_reference(reference, equilibrium, sensitivity, 5e-6)
                row["reference_seconds"] = time.perf_counter()-t
                denominator = max(1e-8, float(np.linalg.norm(j2*qscale)))
                row["relative_jacobian_error"] = float(np.linalg.norm((sensitivity.tip_jacobian-j2)*qscale)/denominator)
                row["maximum_column_relative_error"] = float(np.max(np.linalg.norm((sensitivity.tip_jacobian-j2)*qscale, axis=0)
                                                                                    / np.maximum(1e-6, np.linalg.norm(j2*qscale, axis=0))))
                row["finite_difference_refinement_error"] = float(np.linalg.norm((j1-j2)*qscale)/denominator)
                detail["finite_difference"] = [d1, d2]
                detail["finite_difference_jacobian"] = j2.tolist()
                predictions = local_predictions(reference, equilibrium, sensitivity, probe_rng)
                detail["local_prediction"] = predictions
                row["largest_probe_error_m"] = max(p["error_m"] for p in predictions)
                row["passed"] = (row["relative_jacobian_error"] <= args.relative_jacobian_tolerance
                                 and row["maximum_column_relative_error"] <= args.relative_jacobian_tolerance
                                 and row["finite_difference_refinement_error"] <= args.relative_jacobian_tolerance)
                if not row["passed"]:
                    row["failure"] = "Jacobian or finite-difference refinement tolerance exceeded"
            except (EquilibriumError, ValueError, np.linalg.LinAlgError) as exc:
                row["failure"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            details.append(detail)
            save()
            print(f"{name} {index+1}/{args.samples_per_system}: {'PASS' if row['passed'] else 'FAIL'}; "
                  f"elastic={row['stability_status']}; relative_J_error={row['relative_jacobian_error']}"
                  + (f"; {row['failure']}" if row["failure"] else ""), flush=True)
    summary = save(complete=True)
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
