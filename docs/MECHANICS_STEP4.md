> Historical implementation notes. For current training commands and defaults, use [clean_experiment.md](clean_experiment.md). The old root training entry points have been consolidated.

# Step 4: goal-directed equilibrium control

Step 4 adds a Jacobian controller and a check on proposed policy actions to the
unloaded equilibrium backend. Every accepted move must pass Step 3's branch,
joint, sensitivity, prediction, reverse-solve and elastic-margin checks, plus
a decrease test using the newly solved tip position. No DDPG training script,
baseline configuration, Gymnasium registration or HER replay buffer changes in
this step.

## Evidence carried forward

The supplied Windows Step 3 audit at commit
`3352f80f40b2903411286d990e6f508d5fc93cc2` passed all 13 contracts. Its 207
accepted transition records reconcile with the summary and satisfy the recorded
bounds. All five rejected negative-second-variation initial roots were rejected;
their joint targets were reached on different admissible branches. There were
nine rejected trials. Retry counts can differ across numerical platforms.
These checks support continuation to control experiments, not a proof of global
root uniqueness or continuum elastic stability.

## How physics selects an action

The state includes the current equilibrium and branch information (including
base torsional strain), the tip Jacobian and the sampled elastic assessment.
Joint positions alone do not identify an equilibrium branch.

For a fixed Cartesian goal `g`, define `e = p - g`, where `p` is the solved tip.
The candidate order is:

1. An optional policy proposal, expressed as physical joint displacement:
   translations in metres followed by rotations in radians.
2. A normalized damped least-squares Jacobian displacement.
3. A normalized negative-gradient displacement.

For the Jacobian candidate, let `S` be the diagonal matrix of the tracker's
per-joint displacement caps, `d` the maximum Cartesian displacement, and
`A = J S / d`. The desired tip displacement is `-k e`, norm-limited to `d`.
Compute

```text
u = A.T @ solve(A @ A.T + damping**2 * I, desired_tip_delta / d)
u = u / max(1, max(abs(u)))
candidate_delta_q = S @ u
```

Thus metres and radians are scaled before damping. Default values are
`k=0.5`, `d=0.002 m`, dimensionless damping `0.05`, and goal tolerance `0.001 m`.
These are displacement commands per controller call, not velocities or a
physical time integration model. The third candidate uses the similarly scaled
negative gradient. Joint projection, step caps and backtracking remain the
responsibility of `BranchTracker`.

## The decrease check and its exact scope

Use `V = 0.5 * ||p - g||^2`. For the displacement **after** joint projection,
step limiting and backtracking, compute `s = e.T @ J @ applied_delta_q`.
A candidate must satisfy

```text
s < 0
V_next <= V + c * s < V,       c = 0.1 by default
```

`V_next` uses a new nonlinear equilibrium solve. A favorable linear prediction
alone never accepts a move. The midpoint and endpoint each satisfy both a local
decrease test and a test relative to the original state. All checks occur before
the single state commit. Failed trials preserve the complete cached state; the
controller tries another candidate or returns `stalled`. Once inside the goal
tolerance, it holds without further solver calls.

This is an Armijo sufficient-decrease check, not a strong-Wolfe search or a QP.
See the [SciPy line-search documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html)
for the descent-direction and Armijo terminology. This implementation uses its
own backtracking together with the equilibrium checks.

The resulting property is **runtime-checked monotonic decrease of computed
squared tip error for an unchanged goal**, at accepted samples; holds preserve
it. There is no numerical slack added to the acceptance inequality. This does
not establish:

- convergence to every reachable goal, a uniform contraction rate or finite-time
  arrival (constraints, branch restrictions or local stationary points can stall);
- error monotonicity between sampled equilibria or across changes of waypoint;
- dynamic Lyapunov stability, stability of all redundant joint coordinates, or
  a continuum elastic-stability certificate;
- a rigorous bound on numerical error or agreement with a physical robot.

The fixed-goal error function is a useful Lyapunov candidate, but strict decrease
on accepted samples alone is insufficient for global convergence. There is no
proved uniform lower bound on accepted progress, and a held state can remain
outside tolerance. The diagnostics therefore retain
`controller_stability_certified=False` and `global_convergence_certified=False`.
Changing the goal starts a new error function.

## API example

```python
import numpy as np
from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
from ctr_reach_envs.mechanics import (
    TubeParameters, EquilibriumSolver, BranchTracker, GoalController,
)

tubes = [TubeParameters(**t)
         for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()]
solver = EquilibriumSolver(tubes)
initial = solver.solve(np.array([-.1, -.05, -.02, .4, -.7, 1.2]))
controller = GoalController(BranchTracker(solver, initial))
goal = initial.tip + np.array([0.001, 0.001, 0.0])

for _ in range(60):
    result = controller.step(goal)  # optionally: policy_delta=physical_delta_q
    print(result.status, result.action_source,
          result.diagnostics["error_after_m"])
    if result.status != "moving":
        break
```

The example's arbitrary goal is not guaranteed reachable. Outcomes are
`moving`, `goal_reached`, or `stalled`; a caller must also enforce its own budget.
`result.applied_delta` reports the displacement actually committed.
`candidate_attempts` retains rejection reasons and physics-call costs.

## Preserve HER transition semantics in the next stage

This goal-dependent safeguard is an **action-selection layer**. It is not
silently inserted into the existing HER environment. If replay stored the
original proposed action while the environment filtered it using the original
goal, relabelling the goal could imply a different executed displacement. That
would make the stored transition inconsistent.

The next RL integration must explicitly define critic/plant actions in terms of
executed physical displacements, retain the original proposal and branch state
for diagnostics and learning, and handle goal-dependent reward and termination
when relabelling. A regression test here verifies that replaying an accepted
physical displacement through the goal-independent tracker reproduces the
filtered transition in the tested case. This is not a complete HER integration.

No learned policy was trained or evaluated in Step 4. Damped least squares and
Armijo backtracking are established ingredients; this implementation alone is
not evidence of algorithmic novelty or fewer environment interactions. A later
matched RL comparison must count solver probes and rejected candidates as well
as committed transitions and wall time.

## Run the validation in Windows PowerShell

From the repository on `physics-equilibrium-step1`, with `.venv` activated:

```powershell
git pull --ff-only
python -m pytest tests/test_mechanics.py tests/test_mechanics_step2.py tests/test_mechanics_tracking.py tests/test_mechanics_control.py -q
python validate_mechanics_step4.py --episodes-per-system 2 --seed 7004 --output-dir runs/step4_validation
```

The combined suite contains 64 tests. The audit writes `summary.json`,
`episodes.csv`, `details.json` and `transitions.jsonl`; progress prints per case
and every ten controller calls. Keep these outputs together.

This is an eight-case **local reaching regression**, not a workspace-wide
success estimate or the 1,000-episode paper benchmark. Each seeded aligned start
gets a goal generated by four predeclared witness commands: all translations
`-0.0005 m`, and each rotation `0.04 + Uniform(-0.015, 0.015) rad`.
Held witness commands are retained, not resampled. Trivial goals are counted
separately. Each system tests pure Jacobian control and a synthetic uphill
proposal that exercises rejection and fallback; the latter is not a trained
actor. Stalling can satisfy the safety contracts without counting as reaching
success, so inspect both fields in the summary.

The audit independently recomputes accepted error decrease from the logged tips
and applied joint displacement. A tighter final equilibrium solve checks tip
agreement, elastic assessment and every claimed goal success. Costs distinguish
controller, witness generation and external verification calls, with sensitivity
and stability calls reported separately. RHS counts unavailable for failed calls
are explicitly flagged rather than treated as free computation.

The checked local run is recorded in
[`validation/step4_seed7004.json`](validation/step4_seed7004.json), together with
source hashes and per-episode outcomes. Linux and Windows CI run the same tests
and audit; platform-dependent retry counts and timings need not match exactly.
