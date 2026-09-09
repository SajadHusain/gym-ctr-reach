# Step 3: explicit branch continuation and finite-action validation

`BranchTracker` is a separate mechanics component. It does not change DDPG,
the training scripts, rewards, saved-policy interfaces, or the registered Gym
environments. It advances a supplied equilibrium through checked joint
increments. It does not search for a new root when a move fails.

## What is accepted

Initialization requires an explicit equilibrium from the same tube model. The
tracker reintegrates and analyzes that root. Its elastic diagnostic must be
positive on the tested meshes, its minimum eigenvalue must exceed 0.05, and
the shooting derivative must pass the regularity checks. A negative or
inconclusive initial root is rejected with `BranchInitializationError`.

For each requested joint increment:

1. Project the requested joint target onto the complete feasible extension
   polytope. Preserve the requested and projected increments separately.
2. Limit the proposed increment to at most 1 mm per translation and 0.05 rad
   per rotation. The entire joint vector is scaled together.
3. Refuse a change in segment-event ordering. The current Jacobian does not
   support continuation through coincident moving events. This is an explicit
   limitation of this implementation, including some feasible translation paths.
4. Solve at the midpoint, then endpoint. Each solve uses an explicit shooting
   guess predicted from the preceding checked state. At each point, check tip
   prediction error, base-twist prediction error and change, elastic classification
   and margin, and shooting regularity. Also check the full increment against
   the original state's predictor.
5. Solve backward from the endpoint to the original joint configuration and
   compare the recovered tip and base torsion with the original equilibrium.
6. Commit the endpoint only if all checks pass. Otherwise halve the proposed
   increment and retry, up to eight backtracks. If all attempts fail, retain the
   original equilibrium and both of its cached analyses exactly.

The midpoint is a numerical probe: it is not committed if the endpoint or
reverse check fails. Accepted transitions can be partial commands. Rotation
angles remain unwrapped; the tracker does not silently change the requested
rotation path to another equivalent angle.

## Numerical thresholds

Let L be the longest total tube length, z = L * eta_base, and dq the actual
candidate joint change. For each local prediction and the full-increment
prediction, require

\[
\|\Delta p-J\Delta q\|_2
\leq 2\,\mu\mathrm{m}+0.05\|\Delta p\|_2,
\]

\[
\|\Delta z-z_q\Delta q\|_\infty
\leq 0.001+0.05\|z_q\Delta q\|_\infty,
\qquad \|\Delta z\|_\infty\leq 0.25.
\]

The reverse comparison allows 2 micrometres tip discrepancy and 0.001
dimensionless base-twist discrepancy. The shooting condition limit is 1e5;
the minimum singular-value guard and full BVP residual checks also apply.
Event clearance is 1e-8 m. The second-variation meshes remain (8, 16, 32)
elements per mechanical segment, with the Step 2 convergence and sign checks.

These are declared numerical acceptance tolerances. They have not been tuned
to demonstrate a policy advantage and are not rigorous physical uncertainty
bounds. The complete configuration is included in each transition diagnostic.

## API

```python
import numpy as np
from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from ctr_reach_envs.mechanics import (
    TubeParameters, EquilibriumSolver, BranchTracker, TrackingOptions,
)

solver = EquilibriumSolver([
    TubeParameters(**t)
    for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()
])
q0 = np.array([-.1, -.05, -.02, 0., 0., 0.])
initial = solver.solve(q0)
tracker = BranchTracker(solver, initial, TrackingOptions())
transition = tracker.step([-.0002, .0001, 0., .02, -.01, .01])

print(transition.status, transition.applied_delta)
print(transition.diagnostics["reason"])
snapshot = tracker.state
q = snapshot.equilibrium.joints
eta_base = snapshot.equilibrium.base_torsional_strain
```

`status` is `accepted`, `held`, or `no_motion`. `accepted_fraction` multiplies
`projected_delta`, not the original requested command. A request projected
entirely to zero causes no solve. All result and state snapshots are detached
copies; caller mutation cannot change the accepted tracker state. The input
solver is copied too. A rejection does not erase its computation cost.

The branch state includes base torsional strain, not just q. A future RL
interface must expose or otherwise explicitly manage this history. Do not
replace a memoryless forward-kinematics call with this tracker while leaving
the state definition and replay semantics unexamined.

## Validation and meaning of a pass

The combined test suite contains 51 tests. New checks cover accepted midpoint,
endpoint and reverse solves; independent tighter-solver quarter-point
continuation; rejected endpoints that must not commit a valid midpoint; wrong
twist branches even when the tip agrees; failed solves; inconclusive stability;
event crossings; projection and caps; initialization rejection; immutable
snapshots; and malformed inputs.

The audit uses the five negative cases recorded in the Step 2 reference report.
It first verifies that each cold-solved negative root is refused. It then starts
from aligned clamp rotations at the same insertions and approaches the target
rotations through explicit continuation. There is no random replacement of a
failed case and no jump to a selected alternative root. Eight additional seeded
finite commands exercise new configurations across the four tube systems.

Every committed state and sampled checkpoint is checked. A final tighter-solver
equilibrium and elastic diagnostic cross-check each case. Full attempted-step
records, including rejections, go to `transitions.jsonl`. An explicit stop or a
step-budget limit is reported separately from target completion. A contract
pass means that acceptance/rejection behavior and the refined final state pass
their tests; it does not require every requested target to be reachable.

The first local audit completed all 13 contracts. All five negative roots were
refused at initialization, and all five corresponding joint targets were
subsequently reached on different roots with positive numerical second
variation. This demonstrates why one should not label a joint configuration
unusable merely because a cold solve returns an unstable equilibrium.

The eight additional requests produced partial accepted increments, as expected
from the declared caps. This is not full completion of those eight commands.
See `docs/validation/step3_seed7003.json` for the reproducible report.

## Cost accounting

A first-attempt accepted move uses three equilibrium solves (midpoint,
endpoint, reverse), two sensitivity integrations and two elastic diagnostics.
Initialization reuses the caller's supplied equilibrium but adds its two
analyses. Rejected attempts increase the cost. The report separates tracking
calls, external initialization/reference calls, accepted moves, and rejections.

All attempted equilibrium, sensitivity and stability function calls are counted.
Successful calls expose RHS counts. The existing Step 1/2 exception APIs do not
expose partial RHS counts on failure, so `failed_calls_without_rhs_counts` reports
that limitation explicitly. `rhs_evaluations_known` must not be described as
the complete total if that count is nonzero. Failed trial solves are retained
in the log and can be followed by an accepted smaller increment.

This implementation is a validation-oriented reference. It does not yet show
fewer RL environment interactions or lower training time. Internal physics
queries must be counted when making either claim.

## Scope of stability statements

`elastic_stability_certified`, `continuous_path_certified`,
`controller_stability_certified` and `global_uniqueness_certified` remain false.
Finite-element positivity and midpoint/endpoint checks do not prove that every
point on the continuum path is elastically stable. Small prediction errors and
reverse consistency are numerical branch checks, not a proof of uniqueness.
The tracker does not simulate dynamic snapping or contact, and it does not
enforce Cartesian goal progress or a Lyapunov decrease condition.

## Windows commands

Remain on `physics-equilibrium-step1`:

```powershell
git pull --ff-only
python -m pytest tests/test_mechanics.py tests/test_mechanics_step2.py tests/test_mechanics_tracking.py -q
python validate_mechanics_step3.py --samples-per-system 2 --negative-cases 5 --seed 7003 --output-dir runs/step3_validation
```

Outputs: `summary.json`, `cases.csv`, `details.json`, and `transitions.jsonl`.
The audit prints progress every ten continuation moves and after each case.
No new dependencies are needed. Your local training-file edits are untouched.
No retraining is required to validate this stage.
