# Generalized reaching and sustained precision

The `generalized_hold` task addresses two limitations exposed by the goal-input
audit: restricted goal directions and terminating collection on the first
successful step. It is now the default for **new training runs and new study
plans**. Existing checkpoints without `task_profile` still evaluate on the
legacy task. Use `--task-profile legacy` to explicitly reproduce old training.

These changes provide a learning task and measurements for broader reaching
and holding. They do not demonstrate improved performance before retraining
and matched evaluation, and they do not provide a stability guarantee.

## What changes and what remains fixed

| Component | Legacy | Generalized hold |
|---|---|---|
| Initial translations | Feasible random translations | Same sampling rule |
| Initial rotations | All zero | Common rotation uniform in [-pi, pi], plus independent tube offsets uniform in [-0.15, 0.15] rad |
| Goal witness | Four commands with negative translations and positive rotations | A six-dimensional command sampled uniformly in [-1, 1], repeated for a uniformly sampled count from 2 through 8 |
| Goal endpoint | Forward equilibrium of projected witness endpoint | Same forward model; resample endpoints initially closer than 2 mm at the default 1 mm tolerance |
| Success termination | Stop at first hit | Never terminate because of reaching the goal |
| Collection horizon | At most 60 steps | Exactly 60 valid transitions, unless a numerical failure stops the run |
| Sparse reward | 0 inside tolerance, -1 outside | Unchanged, evaluated on every transition including after reaching |
| Primary evaluation success | First hit | Every one of the final 10 steps inside tolerance |

The witness command is used only to generate the desired endpoint; it is not a
demonstration or a command given to the policy. The observation still contains
egocentric joints, tolerance, achieved goal and desired goal. Neither the witness
joint target nor success history is added to the actor's input.

Each goal applies the actual joint projection and step caps at every witness
increment. Only the initial and endpoint equilibria need solving. The endpoint
is reachable by construction in the joint-constrained forward map; this does
not certify a continuous physical branch or all intermediate solves. Common
rotations broaden azimuth coverage while bounded independent offsets introduce
relative tube rotation. This is still a **local reachable-goal distribution**,
not uniform sampling over the full workspace or all torsional configurations.

Trivial endpoints are rejected with a bounded 32-candidate budget, and the
rejection count and extra solve work are recorded. Numerical equilibrium failures
are raised and counted rather than silently resampled. Failed equilibrium joint
coordinates are included in training/evaluation diagnostics when available.
New training uses an explicit shooting budget of 500 evaluations, increased
from 250 after a recorded generalized-task configuration required 280. The
budget is stored in checkpoint configuration and reconstructed by evaluation;
both comparison arms use the same value. This removes that confirmed budget
failure but does not guarantee that every possible configuration will converge.

The paper-form analytical Jacobian, actor loss, six-joint action limits,
egocentric encoding, actor/critic architecture, HER future-goal sampling,
optimizer settings and exploration profile remain unchanged. The baseline and
guided runs differ in the Jacobian actor-loss weight and the associated
sensitivity work. No branch tracker, stability test, controller wrapper,
near-goal action override or temporal smoothness penalty is introduced.

## Reward, holding and HER consistency

Let e be the next-state Euclidean tip error and epsilon the fixed tolerance.
The implemented reward is r = 0 for e <= epsilon and r = -1 otherwise.
Consequently, leaving the tolerance region during continued collection incurs
negative rewards again. The goal remains fixed for the entire collection window.

For this continuing task, `compute_terminated()` is false even for a successful
real or hindsight-relabeled transition. At the collection horizon Gymnasium
returns `terminated=False, truncated=True`, including when the last transition
is successful. The critic bootstraps through these time-limit truncations. It
therefore learns a continuing task observed through finite collection windows,
not a time-to-go-dependent finite-horizon task. No remaining-time feature is
required for that definition.

`compute_success()`, `compute_reward()` and `compute_terminated()` are deliberately
separate. This avoids the incorrect reward of -1 everywhere when success stops
being a terminal event. The existing HER buffer recomputes reward and termination
for each new goal. Source-state joints and Jacobians remain paired with their
original transition. Proposal-action replay and differentiable joint projection
are unchanged.

The Jacobian loss still asks the current actor to predict a displacement toward
the goal. Its desired displacement tends to zero as the goal error tends to
zero. Collection now supplies the real subsequent transitions on which this
behaviour can be learned and corrected. This is not a Lyapunov constraint and
does not bound second-order tip errors or null-space motion.

## Evaluation definitions

For horizon H and holding window K (default H=60, K=10):

- `reaching_success_rate`: at least one recorded transition has error <= epsilon.
- `sustained_success_rate`: the episode completes and all errors at steps
  H-K+1 through H are <= epsilon.
- `success_rate`: equals sustained success for `generalized_hold`, and first-hit
  success for `legacy`. Always compare the same `success_definition`.
- `first_success_step`: first hit, or null if none. Initial trivial goals remain
  separately flagged; sampled generalized tasks exclude them.
- `hold_window_max_error_m` and `hold_window_rms_error_m`: maximum and RMS error
  in the final K observed steps. Short windows are null. Interrupted episodes
  never count as sustained successes.
- `within_tolerance_fraction`: fraction of executed steps inside tolerance.
- `post_first_hit_max_error_m`: largest error from first hit through the end,
  including any departures and recoveries.

A late recovery can pass the final-window criterion. This is intentionally not
the stronger condition of never leaving after the first hit. The latter can be
inspected using `post_first_hit_max_error_m` and the recorded trajectories.
None of these finite tests establishes indefinite holding or dynamic stability.

The evaluator runs the deterministic frozen actor with joint constraints only.
It does not compute Jacobians or stop/zero the policy's action at the goal.
Existing motion metrics remain per-command differences, not acceleration or jerk.
The saved training config and explicit evaluation profile accompany the results.

## Windows / PowerShell workflow

Keep the previous checkpoints and results. Use new output directories and fresh
training runs; mixing old terminal-semantics replay with the new task is invalid.
Dependency requirements have not changed.

```powershell
git pull --ff-only
python -m pytest tests/test_reach_hold.py tests/test_simple_jacobian_rl.py tests/test_simple_reaching_audit.py tests/test_mechanics_study.py -q
```

Run a matched 10,000-transition pilot. This is a harder distribution and a stricter
success criterion; the old 95-96% first-hit result is not a target guaranteed at
the same budget. The default network remains three hidden layers of 256 units.

```powershell
python train_physics_ddpg_her.py --task-profile generalized_hold --total-timesteps 10000 --physics-weight 0 --exploration-profile paper --seed 7101 --checkpoint-freq 1000 --output-dir runs/reach_hold_ddpg7101

python train_physics_ddpg_her.py --task-profile generalized_hold --total-timesteps 10000 --physics-weight 0.1 --exploration-profile paper --seed 7101 --checkpoint-freq 1000 --output-dir runs/reach_hold_guided7101

python evaluate_physics_ddpg_her.py runs/reach_hold_ddpg7101/final_model.zip --episodes 100 --max-steps 60 --hold-steps 10 --seed 820000 --record-trajectories --output-dir runs/reach_hold_ddpg7101/evaluation_100

python evaluate_physics_ddpg_her.py runs/reach_hold_guided7101/final_model.zip --episodes 100 --max-steps 60 --hold-steps 10 --seed 820000 --record-trajectories --output-dir runs/reach_hold_guided7101/evaluation_100
```

Evaluation restores the profile and sampling parameters from each checkpoint.
An explicit `--task-profile legacy` or `--task-profile generalized_hold` override
is available for a labeled cross-distribution diagnostic; it does not alter the
saved training configuration or policy weights. Old aligned-goal checkpoints
remain loadable with no override. For generalized checkpoints, the separate
goal-dependence audit reverses the sampled witness command for `reverse_actor`;
for old checkpoints it retains the old negative-rotation diagnostic.

Inspect both reaching and sustained success, error tails, numerical failures,
motion metrics and solver cost before extending the budget. Training rollout
metrics include exploration noise and uniform random proposals even near the
goal; they are not directly comparable to deterministic evaluation success.

For the subsequent multi-seed experiment, the runner freezes both arms' task,
holding definition and budgets. Old frozen plans cannot be resumed after code
changes; use a new folder.

```powershell
python run_physics_study.py --stage plan --task-profile generalized_hold --seeds 7100 7101 7102 --total-timesteps 10000 --checkpoint-freq 5000 --eval-episodes 20 --hold-steps 10 --eval-seed 820000 --final-seed 930000 --output-dir runs/reach_hold_study
python run_physics_study.py --stage train --output-dir runs/reach_hold_study
python run_physics_study.py --stage evaluate --output-dir runs/reach_hold_study
python run_physics_study.py --stage report --output-dir runs/reach_hold_study
```

Once the protocol and final-budget checkpoint selection are fixed, run the
separate `--stage final-evaluate` on that study folder. It uses the predeclared
1,000 final evaluation episodes per arm and training seed. Do not repeatedly
tune against this final-test seed block.

## Regression coverage

Tests cover signed joint-goal diversity, varied starting rotations, exact seeded
reproducibility, projected witness endpoints, bounded trivial-goal rejection,
explicit numerical failures, nonterminal successful transitions, truncation
bootstrapping for real and HER transitions, continued motion after first hit,
holding-window edge cases, checkpoint reload without a Jacobian, legacy task
compatibility and rejection of mixed task semantics in study reports. Short
training/evaluation smoke tests check execution, not learning superiority.
