> Historical implementation notes. For current training commands and defaults, use [clean_experiment.md](clean_experiment.md). The old root training entry points have been consolidated.

# Step 7: matched experiments and command smoothness

This stage tests hypotheses. It does not establish better sample efficiency,
smoother actions, or a training-stability theorem.

## What the Step 6 smoke runs establish

The submitted Windows runs each contain only 64 transitions and 48 updates.
Actor-only evaluation succeeded on 0/4 tasks in both cases. The mean final
error was 159.28 mm without the actor loss and 28.59 mm with it. This is a useful
directional observation, but four goals and one training seed are insufficient
for a performance claim. Errors should also be compared with initial errors;
a smaller final error than another policy does not itself establish progress.

Both training runs used the collection safeguard. The zero-loss control replaced
56/64 proposals, including 50 Jacobian fallbacks. The nonzero-loss run replaced
29/64 proposals, including 19 fallbacks. Consequently the 6/6 versus 2/4 rollout
success counts are not unassisted-policy results. On evaluation, the zero-loss
safeguard and Jacobian-only controller had identical endpoint errors and cost.
The nonzero-loss safeguarded policy took 373 equilibrium calls versus 178 for
the independent Jacobian controller. Less intervention cannot be equated with
less total compute or faster reaching.

## DOP853 numerical warning

The warning is the same SciPy numerical warning observed in Step 6, not another
Box warning. Promoting it to an error locates it inside a shooting residual for
the reverse-branch check of `ctr_2`, reset seed 16. Stage derivatives were around
1e-159, with a scale near 1e-10. Squaring the embedded error estimates produces
subnormal numbers; multiplying the third-order term by 0.01 can underflow to
zero. SciPy then evaluates 0/0. See the
[SciPy DOP853 implementation](https://github.com/scipy/scipy/blob/v1.17.0/scipy/integrate/_ivp/rk.py).

`ScaleSafeDOP853` uses the same norm formula with a common error scale factored
out before squaring in extreme ranges. In the ordinary range it calls SciPy's
unchanged calculation. There is no warning filter, state clipping, or tolerance
relaxation. It is used by both the equilibrium and sensitivity integrations.
The ODE equations, method coefficients, shooting boundary conditions, and branch
acceptance checks are unchanged. The protected SciPy error-norm extension point
has regression tests for small/large values and equality to ordinary DOP853.
New invalid numerical states still fail explicitly.

The arithmetic fix can alter the root-finding path of affected trials. Re-run
both comparison arms on this version; do not pair a new arm with the old smoke
control. A passed numerical test does not certify elastic or global stability.

## Experimental arms

| Arm | DDPG/HER actor update | Goal-dependent collection safeguard |
| --- | --- | --- |
| `ddpg` | ordinary DDPG | off |
| `jacobian` | DDPG plus local Jacobian penalty | off |
| `safeguard` | ordinary DDPG | on |
| `both` | DDPG plus local Jacobian penalty | on |

All arms share the equilibrium plant, feasibility constraints, branch tracking,
executed-action replay convention, observations, reward, termination relabelling,
network, optimizer, noise scale, update ratio, goal distribution, and fixed 1 mm
tolerance. Plant rejection/holding still exists when the goal safeguard is off.
Thus `ddpg` means ordinary actor/critic optimization inside the common mechanical
environment. It is not the old fast-simulator paper baseline.

The default study compares `ddpg` with `jacobian`. Use all four arms to separate
the loss effect, safeguard effect, and their interaction. Two safeguarded arms
alone would only measure the incremental effect of the loss under guidance.

Default network: three 256-wide layers with the existing paper actor/late-action
critic. Learning rate 5e-4, gamma .95, tau .001, one gradient update per collected
step, future HER with four sampled goals, normalized Gaussian noise std .05.
The pilot holds the tolerance fixed and linearly anneals the Jacobian coefficient
from .1 to .01 over 2,000 steps. This is an explicit pilot setting, not a tuned
optimum. Architecture and optimization are identical between the paired arms.

## Why the Jacobian loss is not a smoothness penalty

The loss compares `J(q) delta_q` with a desired tip displacement at one state.
It has no term comparing successive actions. A component in the null space of
the 3-by-6 tip Jacobian may change sign without changing this loss. Jacobian
variation, projection-face changes and safeguard switching can also create
action changes. Smoothness improvement must therefore be measured; it is not a
consequence of a small Jacobian loss.

For normalized proposed actions `u_t` and executed actions `a_t`, the primary
variation statistic is computed separately for each:

```text
RMS change = sqrt(sum_t ||a_t-a_(t-1)||^2 / ((T-1) * number_of_joints))
```

`episodes.csv` also includes action magnitude RMS, total variation (sum of
Euclidean differences), second-difference RMS, physical translation-increment
change RMS in metres, rotation-increment change RMS in radians, held-action
fraction, tip path length, episode steps, and initial/final error. Raw rotations
use actual unwrapped increments; metres and radians are never added as physical
quantities. Normalized six-joint metrics use the fixed common action scales.

Metrics exclude transitions across resets and do not invent a prior zero action.
Episodes with fewer than two actions have missing first-difference metrics;
fewer than three have missing second differences. These are never converted to
zero. The model has no physical control timestep, so these are per-command
variation measures, not physical acceleration, jerk, or actuator energy.

Evaluate success and smoothness together: a motionless failed robot is trivially
smooth. Reports retain all-episode and successful-episode metrics, and paired
contrasts additionally compare tasks where both actors succeeded. The count of
such episodes per training seed is recorded. Even normalized metrics are affected
by different reaching times, so retain steps, total variation and trajectories.
This stage measures goal-reaching command smoothness. Continuous path/waypoint
smoothness needs a separate evaluation with no reset at a waypoint.

No temporal regularizer is added in this stage: that would change the hypothesis
being tested. A later ablation can add one explicitly, with previous action/state
semantics and HER handled consistently.

## Frozen plans, checkpoints and statistics

`run_physics_study.py` freezes options, relevant source hashes and dependency
versions in `study.json`. Later stages read that plan. Changed settings, source,
or runtime versions require another study folder. Identical seeds match network
initialization and episode-index reset streams, not identical learning
trajectories. Different policies may end episodes at different times. Arm order
rotates between seeds to reduce systematic ordering effects on wall time.

The trainer saves step zero, every specified checkpoint interval and the final
fixed-budget checkpoint. Snapshots occur after the preceding optimizer update;
each has `config.json`, `budget.json`, and `model.zip`. Physics costs include
reset/witness construction, branch, sensitivity and stability work. Known RHS
counts and failed calls lacking partial RHS counts remain separately reported.
Learning curves use environment interactions and additionally expose solver work
and time. Fewer interactions alone need not mean lower compute cost.

Development evaluation uses identical held-out seed blocks at every checkpoint,
actor-only actions, and the fixed tolerance. Initial-state/goal fingerprints must
agree across policies. Evaluations occur in separate processes after training,
so they do not consume the learner's RNG or change its budget. Final evaluation
uses 1,000 episodes per mode on a separate seed block, the final fixed-budget
checkpoint, and both actor-only and safeguarded modes. A Jacobian-only reference
is evaluated once per seed block, because it does not use the trained network.

The report supplies per-seed learning curves, normalized success-curve AUC over
the fixed budget, final success/precision/motion/cost differences, seed standard
deviations and paired seed-bootstrap intervals. Training seed is the uncertainty
unit; 1,000 episodes are not 1,000 independent training runs. With only three or
five training seeds, these intervals are descriptive and can be unstable. Missing
jobs and evaluation failures stay visible; no improvement flag is inferred from
a p-value or a smoke comparison. Raw CSV/JSON outputs support subsequent analysis.

Existing run directories are retained. Re-running a stage skips existing jobs;
an incomplete job is not resumed from weights without its replay and RNG state.
Choose a new study folder for a clean rerun. Checkpoints support evaluation, not
bit-exact training resumption. An incomplete study must not be called complete.

## Windows commands

From the activated `.venv` on `physics-equilibrium-step1`:

```powershell
git pull --ff-only
python -m pip install -e ".[physics-train,test]"
python -m pytest tests/test_mechanics_study.py tests/test_mechanics_rl.py tests/test_mechanics_jacobian.py -q -W error::RuntimeWarning
```

This targeted command runs 37 tests. Locally, those and the 64 core mechanics
tests passed with runtime warnings promoted to errors. A two-seed/two-arm tiny
study verified identical initialization within each seed, checkpoint update
counts, and smoothness metrics reconstructed from 12 recorded trajectories.
The separate final-evaluation stage was also exercised with one episode per
mode. These are software checks, not the 1,000-episode scientific experiment;
see [validation evidence](validation/step7_integration.json).

First create a budgeted pilot (three seeds, two arms, 12,000 training transitions
in total). This command only writes the plan:

```powershell
python run_physics_study.py --stage plan --seeds 7100 7101 7102 --total-timesteps 2000 --output-dir runs/step7_pilot
```

Run training and then development evaluation/reporting. Expect hours, not smoke
test duration; reset/witness and branch solves dominate cost. Development uses
20 episodes at each of five checkpoints per run, plus the independent reference:

```powershell
python run_physics_study.py --stage train --output-dir runs/step7_pilot
python run_physics_study.py --stage evaluate --output-dir runs/step7_pilot
python run_physics_study.py --stage report --output-dir runs/step7_pilot
```

Inspect `development_report.json`, `development_learning_curves.csv`, training
`summary.json`/`updates.csv`, and evaluation trajectories. A 2,000-step pilot may
still have low success: it profiles learning direction and compute before fixing
a larger study. It is not chosen as a sufficient convergence budget.

After fixing the larger protocol using only development results, a new plan can
include five or more training seeds and all four arms. Choose its interaction
budget and annealing horizon explicitly. Run `plan`, `train`, `evaluate`, and
`report` as above on that new folder. Only once that protocol is frozen, run its
planned 1,000-episode final test:

```powershell
python run_physics_study.py --stage final-evaluate --output-dir runs/YOUR_FROZEN_STUDY
```

Final held-out results must not be reused repeatedly to select weights, budgets
or checkpoints while still being described as an untouched test set.
