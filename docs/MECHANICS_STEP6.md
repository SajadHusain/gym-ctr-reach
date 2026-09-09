# Step 6: finite observation bounds and a Jacobian actor objective

The submitted Step 5 run passed its integration checks: 64 transitions, 48
gradient updates, finite parameters and exact executed-action replay. All seven
completed episodes succeeded, but 57/64 actions used Jacobian fallback. That
89.1% fallback rate prevents attributing the success to the learned actor. The
three failed physics calls are reported, including unavailable partial RHS
counts; there were no failed resets. No training-performance claim follows from
this smoke run.

## Observation warnings: fixed without changing the state encoding

The warnings concerned declared infinite Box bounds, not observed infinities.
Step 6 retains every observation value and feature ordering. It neither clips
torsion nor applies a saturating transformation that could hide branch state.

For this specific unloaded, isotropic-cross-section equilibrium model, let
`k_i` be the norm of tube i's intrinsic planar curvature, `k_max = max(k_i)`,
`Lc_i` its curved length, and `L` the longest tube length. The common curvature
is a convex stiffness-weighted sum of rotated intrinsic curvatures, hence its
norm is at most `k_max`. The implemented torsion equation gives

```text
|d eta_i / ds| <= (EI_i / GJ_i) * k_i * k_max
```

Its right-hand side vanishes in the straight portion and outside the deployed
tube. The guide has constant torsional strain. Integrating backward from the
free-tip condition therefore bounds the exact model's base state:

```text
|L * eta_i(base)| <= B_i
B_i = L * Lc_i * (EI_i / GJ_i) * k_i * k_max
```

The declared numerical envelope is `1.01 * B_i + 100 * boundary_tolerance`,
rounded outward to float32. The fixed tolerance feature also receives a finite
upper bound. An observation outside its declared bounds raises an error and
ends further stepping until reset; it is not clipped. The padding is an
engineering allowance, not a rigorous integration-error estimate. This bound
uses the unloaded equations in `solver.py`; it must be reconsidered if external
torques, forces, different constitutive laws or dynamics are added. It is not a
stability or uniqueness theorem.

New checkpoints record `observation_bounds_version=2`. The evaluator recognizes
old Step 5 configuration files and restores their legacy Box declarations when
loading those models. Network inputs retain the same values, so no weight
conversion is necessary. New training uses finite bounds by default.

## How the Jacobian now guides policy optimization

Previously, physics changed behavior during collection while optimization was
ordinary DDPG. Step 6 can additionally optimize

```text
L_actor = -mean Q(s, mu_theta(s)) + lambda(t) * L_J
L_J = mean || [J(q) delta_q_theta - d_goal] / d_scale ||^2
d_goal = norm_clip(0.5 * (goal - tip), d_scale)
d_scale = 0.002 m
```

`mu_theta` is the current differentiable actor output. `delta_q_theta` is formed
by applying the fixed metre/radian action scales, projecting extensions onto
the complete joint-feasibility polytope and imposing the same uniform step cap
as the mechanical tracker. The projection enumerates the same active faces as
`JointConstraints.project()`, with double precision for narrow feasible faces.
Gradients pass through the selected affine face and the step cap. The map is
piecewise differentiable; at face changes, a unique classical derivative need
not exist. Finite-difference tests cover an active face away from such changes.

The Cartesian normalization avoids directly comparing radians with metres.
The regularizer encourages the current actor's predicted feasible motion to
approach the goal. It can provide directional gradients even where the sparse
reaching reward has not provided a useful critic gradient. This is the reason
to test it; improved sample efficiency is an experimental hypothesis.

`J(q)` comes from the analytic variational ODE and implicit shooting derivative
on the stored equilibrium branch. The branch tracker already computed and
checked it during collection. The source joints, Jacobian and action scales are
copied into transition information and matched to real and hindsight samples.
They are detached constants during optimization. No finite-difference forward
solves or new equilibrium solves occur inside actor updates.

HER changes the desired goal, so `d_goal` is recomputed from each sampled goal.
The source state and Jacobian stay unchanged. The implementation never reuses a
goal-directed target computed for the original goal. Physics context is removed
from the temporary sampling dictionary before it reaches the policy; the actor
does not receive a Jacobian as an extra observation.

The critic update and unfiltered-target-actor convention remain DDPG. The actor
update includes the new term; this is not an auxiliary prediction head, a
differentiable ODE solver or a change to HER's executed-action convention.
The implementation extends the [SB3 DDPG/TD3 update](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/td3/td3.html)
and [HER sampling](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/her/her_replay_buffer.html).
The new replay adapter retains the existing reward-and-termination relabelling.

## Loss weight, diagnostics and limits

The default remains `--physics-weight 0`, preserving Step 5 training. Supplying
`--physics-weight 0.1` selects `JacobianDDPG`; by default its weight decreases
linearly to `0.01` over 100,000 collection steps. Explicit flags are
`--physics-final-weight` and `--physics-anneal-steps`. A zero initial and final
weight delegates directly to the original SB3 update; a regression test checks
bit-for-bit identical network parameters after matching runs.

The pilot starts at 0.1 and retains the critic objective. Loss scales and gradient
magnitudes determine its influence; this coefficient is not a tuned optimum. The
weight and normalization are recorded in `config.json`; `updates.csv` records
the component losses, total actor loss and the norm of the Jacobian loss's
gradient with respect to the current actor output. Nonfinite losses or gradients
abort the corresponding optimizer update. This is failure detection, not a
proof that neural optimization converges.

The Jacobian is local. The differentiable proposal includes joint projection
and the step cap, but not the equilibrium tracker's nonlinear rejection,
branch checks or backtracking. Therefore the loss does not guarantee that its
predicted displacement will be executed unchanged. A singular tip Jacobian can
offer no gradient in an unreachable direction. A small loss is neither a
physical-accuracy certificate nor a global reaching guarantee.

The online safeguard remains a separate option. Its measured error-decrease
checks apply to accepted original-goal transitions. They do not certify the
actor weights, the hindsight goal, or unfiltered actor evaluation. The current
physics replay assumes one fixed tube system and unnormalized replay
coordinates; VecNormalize is explicitly rejected rather than silently changing
the units of the loss.

## Validation and Windows commands

From the activated environment on `physics-equilibrium-step1`:

```powershell
git pull --ff-only
python -m pytest tests/test_mechanics_rl.py tests/test_mechanics_jacobian.py -q
```

This runs 24 tests. The observation Box warnings should be gone. A SciPy
RuntimeWarning can still occur during a rejected numerical trial; it is a
different issue and is not suppressed. The test suite verifies the accepted
results and the audit records failed-call costs.

Start with matching short runs using the same seed, network, episode budget and
collection safeguard. These are integration checks, not success-rate evidence:

```powershell
python train_physics_ddpg_her.py --physics-weight 0 --seed 7006 --output-dir runs/step6_control
python train_physics_ddpg_her.py --physics-weight 0.1 --seed 7006 --output-dir runs/step6_jacobian
```

Both commands default to 64 transitions, 16 warmup steps and a 16-step episode
budget. The nonzero-weight run should show positive `physics_actor_updates`,
finite parameters, zero executed-action replay error, and finite loss/gradient
diagnostics. Its `extra_equilibrium_calls` should be zero: this counts additional
optimizer queries only, not the considerable physics cost during collection.
Training directories are never overwritten.

Evaluate the learned actor separately from fallback assistance on matching
held-out seeds:

```powershell
python evaluate_physics_ddpg_her.py runs/step6_control/final_model.zip --episodes 4 --max-steps 30 --seed 800000 --output-dir runs/step6_control_eval
python evaluate_physics_ddpg_her.py runs/step6_jacobian/final_model.zip --episodes 4 --max-steps 30 --seed 800000 --output-dir runs/step6_jacobian_eval
```

Each command reports actor-only, safeguarded-actor and Jacobian-only results.
The last mode is an independent controller benchmark and should agree across
the paired configurations. After this executable check, a performance study
needs multiple training seeds, longer matched budgets and the planned 1,000
held-out episodes per evaluated policy. Compare actor-only precision and success,
fallback usage, environment transitions, all solver/sensitivity/stability work
and wall time. Both training arms must use this same equilibrium model and local
goal distribution. The old paper baseline is not a matched control for this
changed simulator/task.

The direct Jacobian loss is an experimental baseline component, not evidence by
itself of research novelty. A publication claim requires its ablations and a
comparison with the standalone Jacobian controller and relevant learning
methods. No improvement or stability theorem is claimed from the smoke runs.

The checked source hashes and local results are recorded in
[`validation/step6_seed7005.json`](validation/step6_seed7005.json).
