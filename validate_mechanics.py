"""Seeded numerical audit of the experimental unloaded equilibrium backend.

Run tests/test_mechanics.py separately for the independent analytical checks.
This command checks numerical consistency; it does not certify elastic stability.
"""
import argparse
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
from ctr_reach_envs.mechanics import TubeParameters, EquilibriumSolver, SolverOptions, EquilibriumError


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-system", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--tip-threshold-m", type=float, default=20e-6)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mechanics_validation"))
    args = parser.parse_args()
    if args.samples_per_system < 1:
        parser.error("--samples-per-system must be positive")
    if not np.isfinite(args.tip_threshold_m) or args.tip_threshold_m <= 0:
        parser.error("--tip-threshold-m must be finite and positive")
    return args


def sample_joints(solver, rng):
    # Rejection sampling for joint feasibility only; failed mechanics solves are
    # recorded, never discarded or replaced with easier configurations.
    for _ in range(10000):
        q = np.r_[rng.uniform(-solver.lengths+solver.constraints.minimum_deployed, 0),
                  rng.uniform(-np.pi, np.pi, solver.n)]
        if solver.constraints.is_feasible(q):
            return q
    raise RuntimeError("Unable to sample a feasible joint configuration")


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    standard = SolverOptions()
    refined = SolverOptions(rtol=1e-11, atol=1e-13, max_step=.002)
    rows = []
    started = time.perf_counter()
    for system_name, system in CTR_SYSTEMS_PARAMETERS.items():
        tubes = [TubeParameters(**t) for t in system.values()]
        solver = EquilibriumSolver(tubes, standard)
        reference = EquilibriumSolver(tubes, refined)
        for index in range(args.samples_per_system):
            q = sample_joints(solver, rng)
            row = {"system": system_name, "sample": index, "passed": False,
                   "failure": "", "tip_discrepancy_m": None,
                   "boundary_residual_scaled": None, "base_torque_sum_nm": None,
                   "solve_seconds": None, "shooting_evaluations": None,
                   "rhs_evaluations": None, "reference_rhs_evaluations": None,
                   "used_curvature_continuation": None,
                   **{f"q{i}": float(v) for i, v in enumerate(q)}}
            t = time.perf_counter()
            try:
                result = solver.solve(q)
                row["solve_seconds"] = time.perf_counter()-t
                d = result.diagnostics
                row.update({k: d[k] for k in ["boundary_residual_scaled", "base_torque_sum_nm",
                                             "shooting_evaluations", "rhs_evaluations", "used_curvature_continuation"]})
                # Explicitly reuse the root estimate to compare the same local
                # equilibrium. This is not a proof of global branch uniqueness.
                ref = reference.solve(q, initial_torsion=result.base_torsional_strain)
                row["reference_rhs_evaluations"] = ref.diagnostics["rhs_evaluations"]
                row["tip_discrepancy_m"] = float(np.linalg.norm(result.tip-ref.tip))
                row["passed"] = row["tip_discrepancy_m"] <= args.tip_threshold_m
                if not row["passed"]:
                    row["failure"] = "tip_refinement_threshold_exceeded"
            except (EquilibriumError, ValueError) as exc:
                row["failure"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"{system_name} {index+1}/{args.samples_per_system}: "
                  f"{'PASS' if row['passed'] else 'FAIL'}", flush=True)
            # Preserve completed results if a longer audit is interrupted.
            with (args.output_dir/"cases.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    differences = [r["tip_discrepancy_m"] for r in rows if r["tip_discrepancy_m"] is not None]
    times = [r["solve_seconds"] for r in rows if r["solve_seconds"] is not None]
    summary = {"complete": True, "samples": len(rows), "passed": sum(r["passed"] for r in rows),
               "failed": sum(not r["passed"] for r in rows), "seed": args.seed,
               "tip_threshold_m": args.tip_threshold_m,
               "max_tip_discrepancy_m": max(differences, default=None),
               "mean_standard_solve_seconds": float(np.mean(times)) if times else None,
               "total_elapsed_seconds": time.perf_counter()-started,
               "standard_solver": asdict(standard), "reference_solver": asdict(refined),
               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "git_commit": commit, "mechanics_backend": "unloaded_equilibrium_bvp_v1",
               "elastic_stability_certified": False, "global_uniqueness_certified": False,
               "refinement_is_rigorous_error_bound": False}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
