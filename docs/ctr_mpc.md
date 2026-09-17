# CTR nonlinear model predictive control

This controller implements the finite-horizon nonlinear program in Khadem,
O'Neill, Mitros, da Cruz and Bergeles, **Autonomous Steering of Concentric Tube
Robots via Nonlinear Model Predictive Control**, IEEE T-RO 36(5), 2020,
[DOI 10.1109/TRO.2020.2991651](https://doi.org/10.1109/TRO.2020.2991651).
It runs directly; there is no actor training phase or learned critic.

The authors' [reference repository](https://github.com/RViMLab/TRO2020-CTR-Model-Predictive-Control/tree/a9cf2a880a338a7141acf11d4419997a4c5f235b)
was inspected at commit `a9cf2a880a338a7141acf11d4419997a4c5f235b`.
Its public `Example.py` calls a two-tube, single-tip SLSQP optimizer. That
execution path does not implement the paper's complete multi-node, unknown-base-
torsion constrained NLP. This is an independent implementation of the published
equations, using this repository's mechanics; no upstream source was copied.

## Two explicitly different plants

| `--plant` | Prediction and execution | Valid comparison |
|---|---|---|
| `original_ivp` | Original zero-initial-torsion forward model, original repeated action increments, original sampled goals | Direct comparison with the existing continuous-IVP RL evaluations |
| `paper_bvp` | Base torsion optimized in the MPC IVPs, zero free-tip torsion enforced; execution checked with an independent BVP solve | A separate experiment with the paper's mechanical boundary conditions |

Do not compare their success rates as if they used identical physics. In paper
mode the same seeded joint sampler is used, but initial and goal positions are
recomputed with the equilibrium BVP. The environment fingerprint changes.
In original mode both environment and per-episode task fingerprints are retained.
The controller sees the current joints/tip and Cartesian goal, never the sampled
joint configuration that generated the goal.

## Mathematical implementation

At each decision the measured current joint configuration `q0` and tip `x0` are
fixed. For `H` future configurations, optimize

\[
\min_{q_{1:H},z_{1:H}}\quad
\sum_{k=1}^{H} w_k \left\|\frac{p(q_k,z_k)-g_k}{\ell}\right\|^2
+\rho\sum_{k=1}^{H}\left\|D^{-1}(q_k-q_{k-1})\right\|^2.
\]

Here `z = L psi`, `L` is the longest tube length, `psi` is base torsional strain,
`D` contains the physical joint increments available per environment decision,
and `ell=0.01 m` scales the tracking cost. By default `w_k=1`, including the
terminal stage. The small optional motion penalty defaults to `rho=0.001`;
`--move-weight 0` removes it to recover pure tip tracking. The API also accepts an
`H x 3` target preview; the reaching CLI repeats the fixed goal at every node.

For paper mode each prediction integrates the full unloaded nonlinear rod ODE:

\[
\theta_i(0)=\alpha_i-\beta_i\psi_i,\qquad
u_{i3}(0)=\psi_i,\qquad
L u_{i3}(L_i+\beta_i)=0.
\]

The last expression is an equality constraint at **each tube's own tip at every
prediction node**. The Bishop-frame representation is equivalent to the paper's
material-frame equations. Base torsion is a decision variable; there is no nested
BVP root search in the optimizer. A separate root search is used for the simulated
plant and initialization. Original-IVP mode has no torsion decision variables or
free-tip equalities and uses exactly the existing forward model.

Both modes enforce extension bounds, base ordering, deployed-tip ordering,
optional rotation limits, and `abs(q_k-q_(k-1)) <= D`. Paper mode also imposes the
base spacing and outer-base clearance of Eq. (14), default `epsilon=1e-5 m`.
Maximum retraction is set by the repository's per-tube minimum deployment bounds,
rather than a separate paper-specific scalar `phi_max`.

The current tip must agree with the current prediction before optimization.
Fixing the measured current joints is a stronger initialization than matching
only the measured tip in the paper's Eq. (14a). `H` counts future commands here;
the paper's horizon notation includes its initial node, so do not identify these
integers without checking the indexing.

Only the first command is applied. The unused sequence is shifted for the next
solve, and the controller replans from the new measured state. Each candidate
node uses a new nonlinear IVP; this does not propagate one fixed local Jacobian
over a distant goal. Derivatives of this NLP use cached stage-local finite
differences. SLSQP replaces the paper's custom interior-point solver; this is not
a reproduction of its reported timing or exact optimizer iterates.

## Run a matched IVP pilot first

From the activated virtual environment in the repository:

```powershell
python evaluate_ctr_mpc.py `
  --config runs/ivp_short_h2_seed10_v1/config.json `
  --plant original_ivp `
  --episodes 10 --seed 920000 `
  --tolerance-m 0.0015 --max-steps 200 `
  --horizon 2 --max-iterations 25 --max-model-evaluations 400 `
  --output-dir runs/mpc_ivp_h2_pilot
```

Use the same saved configuration as the RL comparison. The adjacent checkpoint
`config.json` is also accepted. This command loads no checkpoint weights and
requires only the base NumPy/SciPy/Gymnasium installation. It rejects legacy
segment quantization and Jacobian observation configurations rather than changing
the evaluation environment silently. Use `physics_observation.mode=none` runs.

MPC can consume substantially more wall time than an actor. Inspect the pilot's
`planning_seconds_median`, `planning_seconds_p95`, `hold_fraction`, numerical
`failures`, and `controller.csv` before starting 500 episodes. Planning cost and
simulated interactions must be reported separately. If the pilot is usable, run
the same command with `--episodes 500` and a new output directory. Compare it with
the RL evaluations using identical seeds, tolerance, horizon, configuration and
per-episode task hashes. A 10-episode pilot is not evidence of a reliable success
rate. Investigate solver statuses before increasing the prediction horizon.

## Run the paper's free-tip formulation separately

```powershell
python evaluate_ctr_mpc.py `
  --config runs/ivp_short_h2_seed10_v1/config.json `
  --plant paper_bvp `
  --episodes 2 --seed 920000 `
  --tolerance-m 0.0015 --max-steps 200 `
  --horizon 2 --max-iterations 25 --max-model-evaluations 400 `
  --base-separation-m 0.00001 `
  --output-dir runs/mpc_paper_bvp_h2_pilot
```

This uses your saved three-tube dimensions, not the reference demo's two tubes.
Tests separately exercise the reference demo's two-tube parameters. The controller
supports one to three tubes; the evaluation CLI follows the existing three-tube
experiment format. Try `--horizon 5` only after measuring horizon-2 cost and
feasibility. The finite-difference solve may need a larger evaluation budget.

The independent BVP execution solve starts from the previous measured base
torsion, **not the optimizer's proposed torsion**. If its tip differs from the
MPC prediction by more than `--branch-tip-tolerance-m` (default 0.1 mm), the
episode is reported as a numerical/model-consistency failure. This avoids
accepting an optimizer-selected branch as its own ground truth. It is not a proof
of elastic stability or of continuous branch following between commands. These
unloaded, quasi-static models do not include obstacles, tissue loads, actuator
dynamics or certified snap avoidance.

## Outputs and failure semantics

- `config.json`: source environment, controller settings, mechanical contract,
  attribution, evaluation settings and fingerprint.
- `episodes.csv`: original evaluator's task hash, success, final error, steps,
  failures and motion RMS metrics.
- `controller.csv`: every decision's optimizer status, convergence, hold flag,
  objective before/after, independent boundary residual check, prediction error,
  nonlinear model call count, planning time and execution failures.
- `summary.json`: success with Wilson interval, errors, steps, hold fraction,
  numerical failures, planning-time quantiles and separate planning/plant costs.

A solver's success flag alone never authorizes a command. A candidate must satisfy
all constraints and improve the horizon objective relative to holding. Feasible
improved candidates may be used at the iteration/evaluation limit; otherwise the
controller holds and records why. An accepted horizon improvement is not a
guarantee that the first step reduces distance, or that global reaching succeeds.
All failed episodes stay in the success-rate denominator. Goal misses and planner
holds are distinct from numerical execution failures. No trajectory is silently
resampled to remove a hard case. Output directories must be new.

## Validation and next RL integration

Run:

```powershell
python -m pip install -e ".[test]"
python -m pytest tests/test_ctr_mpc.py tests/test_mechanics.py -q
```

Tests check preview-dependent first commands, unknown torsion optimization,
boundary residuals against an independent BVP, a real two-tube nonlinear control
step, action/substep parity, budget and mismatch fallbacks, identical original
task hashes, and failure accounting. These establish implementation behavior,
not a whole-workspace success rate or real-time performance.

First establish whether this MPC solves the held-out tasks. Once it does, use its
successful trajectories as demonstrations or policy warm starts and measure
whether an actor can reproduce that performance at lower execution cost. This
change deliberately supplies the requested MPC controller and benchmark; it does
not claim an untested MPC+RL training improvement.
