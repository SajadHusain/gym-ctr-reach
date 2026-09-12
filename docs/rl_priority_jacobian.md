> Historical implementation notes. For current training commands and defaults, use [clean_experiment.md](clean_experiment.md). The old root training entry points have been consolidated.

# RL-priority Jacobian actor updates

This implementation tests whether local mechanics can assist reward-based
learning without overriding the current critic's descent direction. It does
not establish better return, fewer interactions, physical stability or novelty
by construction. Projection is an asymmetric adaptation of the gradient-conflict
idea in [Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).
The Adam displacement check below is an additional engineering choice in this
repository, not a theorem from that paper.

## The two objectives and their gradients

The environment still gives reward 0 within the tip tolerance and -1 outside.
The critic still minimizes its reward-based Bellman error, with rewards and
termination recomputed for HER goals. The auxiliary term does not change reward.

For a replay minibatch, freeze the stored source configurations, their analytical
tip Jacobians, goals and the just-updated critic parameters. Let

\[
L_R(\theta)=-\frac1B\sum_i Q_\phi(o_i,\mu_\theta(o_i)),\qquad
L_J(\theta)=\frac1{N_v}\sum_{i\text{ valid}}
\left\|\frac{J_p(q_i)\Delta q_i(\mu_\theta(o_i))-d_i}{c}\right\|^2.
\]

Here `ProjectedJacobianLoss` computes the actual joint projection and uniform
increment cap differentiably within the selected constraint face. The desired
local displacement is

\[
d_i=\frac{k(p_{g,i}-p_i)}{\max(1,\|k(p_{g,i}-p_i)\|/c)},\quad
k=0.5,\quad c=0.002\;\mathrm m.
\]

These existing local-target hyperparameters are unchanged by this patch. They
are not material constants or stability certificates. Zero valid Jacobians give
zero auxiliary loss. HER changes the goal and therefore recomputes this target;
the source-state Jacobian stays paired with its transition. No differentiation
through the BVP and no additional equilibrium solves occur during optimization.

`torch.autograd.grad` calculates *actor parameter* gradients

\[
g_R=\nabla_\theta L_R,\qquad g_J=\nabla_\theta L_J.
\]

The old action-gradient norm is insufficient to diagnose interference, since
the actor's parameterization changes the inner product. The new logs use the
flattened trainable actor parameters for both gradients.

## Protecting the RL direction

For nonzero `g_R`, remove only the conflicting component:

\[
\bar g_J=g_J-
\frac{\min(0,g_R^Tg_J)}{\|g_R\|^2}g_R.
\]

If the dot product is positive, this operation leaves the auxiliary gradient
unchanged. If negative, it projects onto the plane perpendicular to `g_R`.
Then limit the *weighted* auxiliary norm using an explicit optimization
hyperparameter `rho = physics_max_aux_ratio`, default 1:

\[
\tilde g_J=\bar g_J\min\left(1,
\frac{\rho\|g_R\|}{\lambda\|\bar g_J\|}\right),\qquad
d=g_R+\lambda\tilde g_J.
\]

The zero-denominator case leaves a zero auxiliary contribution. If `g_R` is
zero, the checked actor update is skipped. The norm cap ensures the auxiliary
part cannot exceed `rho` times the RL gradient norm even when aligned. `rho`
is not a physics-derived number and must be included in configuration/ablations.

In exact arithmetic,

\[
g_R^Td=\|g_R\|^2+\lambda g_R^T\tilde g_J\geq\|g_R\|^2.
\]

Therefore an infinitesimal plain-gradient step `theta - eta*d` descends the
current RL surrogate to first order. This is a statement about a fixed learned
critic and a sampled batch, not true expected return. This projected update
generally is **not** the gradient of the original scalar `L_R + lambda L_J`.
The CSV `total_actor_loss` remains that scalar's diagnostic value only.

## Adam and finite updates

Adam rescales coordinates and uses past gradients. The inequality above alone
does not ensure descent for an Adam step. The `rl_priority` mode consequently:

1. Saves actor parameters and Adam moments/step counters.
2. Makes an Adam trial using `d`.
3. Computes the actual parameter displacement `Delta theta` and re-evaluates
   `L_R` on the same minibatch with the critic held fixed.
4. Accepts only finite trials satisfying `g_R^T Delta theta < 0` and
   `L_R(theta + Delta theta) <= L_R(theta)`.
5. Otherwise tries displacement fractions 1/2, 1/4, ..., 1/64 by default.
6. If no fraction passes, restores parameters and Adam state, then tries an
   RL-only Adam update with the same acceptance rules.
7. If both candidates fail, restores everything and skips this actor update.

For accepted fractional steps, the candidate's moments are retained and only
the parameter displacement is shortened. This is an explicit line-search
variant of Adam. Rejected trials never advance its saved counters. Critic and
target-network updates continue normally. The default separate, deterministic
MLPs are supported; batch normalization, dropout and shared actor/critic
parameters are rejected in checked mode to preserve the check's meaning.

This prevents observed finite-step increases of the specified training
surrogate for accepted updates, subject to numerical precision. It cannot
correct a wrong critic. A high rejection/skip rate is an experimental failure
signal to investigate, not evidence of better training. It can slow learning.
There are no branch controllers, joint-action overrides or Lyapunov constraints.

## Configuration and diagnostics

New trainer/study CLI runs default to `--physics-integration rl_priority`.
Use `--physics-integration sum` for the former weighted-sum ablation. With
`--physics-weight 0 --physics-final-weight 0`, ordinary DDPG+HER is unchanged:
it computes no Jacobians and does not use the actor-step checks. Old checkpoints
and old saved study configurations lacking the new option retain `sum`.
Checkpoint configurations preserve the integration mode, ratio and backtrack
budget. Original exploration settings are unchanged in all arms.

`updates.csv` records:

| Field | Interpretation |
|---|---|
| `gradient_dot`, `gradient_cosine`, `gradient_conflict` | Raw actor parameter gradient agreement before correction |
| `rl_gradient_norm`, `jacobian_parameter_gradient_norm` | Separate magnitudes; compare weighted gradients, not scalar loss magnitudes |
| `weighted_aux_norm_ratio_before/after` | Auxiliary influence before/after projection and capping |
| `projected_gradient_dot` | Dot product after correction, before Adam |
| `actor_step_first_order` | `g_R` dotted with the accepted, actual parameter displacement |
| `rl_surrogate_before/after/change` | Same-batch critic surrogate before/after the retained actor step |
| `actor_step_scale` | Accepted fraction of the Adam trial; zero if skipped |
| `actor_step_rl_only` | Accepted RL-only fallback indicator |
| `actor_step_skipped`, `actor_step_rejected_trials` | Rejection cost and lack of an accepted actor update |
| `actor_step_checked` | Distinguishes checked updates from the original sum mode |

Metrics average across updates in one `train()` call; default is one update.
`physics_actor_updates` counts update attempts, including skipped actor steps.
Gradient ratios/cosines are recorded as zero for zero denominators; inspect
the corresponding gradient norms before interpreting them. Ordinary DDPG rows
leave unavailable gradient-check diagnostics empty. Tests include aligned and
opposed gradients, a positive-alignment overshoot, an Adam preconditioner that
reverses descent, momentum rejection, exact state rollback, HER and checkpoint
restoration. Passing them verifies implementation, not sample efficiency.

## Reading the uploaded 30,000-transition run

The supplied run at commit `5e8db9a` used generalized holding, a constant 0.1
Jacobian weight and the old unprotected sum. Its 500 completed exploration
episodes contain 124 first hits (24.8%), one final-step hit (0.2%), and zero
final-window holding successes. First-hit frequency rose from 13/100 in the
first 100 episodes to 47/100 in the final 100. Mean final error fell from
20.397 mm to 12.333 mm over those respective windows. This is progress in noisy
collection, with poor precision; it does not establish deterministic performance.

There were 8,668 uniform-mixture proposals after warmup, in addition to the 200
warmup actions, and Gaussian noise on the other proposals. Holding measurements
from exploration therefore do not characterize the frozen actor. The 1,455
unavailable sensitivities (4.85% of 30,000 calls) were masked and did not discard
the corresponding RL transitions. The run finished with finite parameters and
no fatal solver failure. Existing scalar losses cannot identify gradient
conflict retrospectively. Critic/Jacobian losses rising near the end are a
reason to investigate, not proof that either loss caused the problem.

## Reaching first; holding as a separate claim

`generalized_reach` retains exactly the generalized holding distribution of
initial configurations and reachable goals, but ends at the first hit. Its HER
targets recompute goal-dependent termination, including successes at a timeout.
`generalized_hold` remains available and remains the default task for backward
CLI continuity; select the reaching profile explicitly.

Holding is not required for a first-hit reaching/sample-efficiency claim.
Sustained-precision claims require continued evaluation after first hit; physical
stability needs stronger evidence still. Do not change to the restrictive legacy
goal distribution, raise the tolerance, or count holding failures as successes
without explicitly changing the reported endpoint. Early-terminated reaching
evaluations do not establish holding performance.

First evaluate the EXISTING 30k checkpoint without retraining. This single
continuing evaluation reports both first-hit and final-window success:

```powershell
python evaluate_physics_ddpg_her.py runs/reach_hold_guided7101_retry/final_model.zip --episodes 100 --max-steps 60 --seed 810000 --record-trajectories --output-dir runs/reach_hold_guided7101_retry/deterministic_100
```

If narrowing the experiment to reaching, start new run directories with matched
settings. These are development pilots, not final paper results:

```powershell
python train_physics_ddpg_her.py --task-profile generalized_reach --total-timesteps 10000 --physics-weight 0 --seed 7101 --checkpoint-freq 1000 --output-dir runs/reach_ddpg7101
python train_physics_ddpg_her.py --task-profile generalized_reach --total-timesteps 10000 --physics-weight 0.1 --physics-integration sum --seed 7101 --checkpoint-freq 1000 --output-dir runs/reach_sum7101
python train_physics_ddpg_her.py --task-profile generalized_reach --total-timesteps 10000 --physics-weight 0.1 --physics-integration rl_priority --seed 7101 --checkpoint-freq 1000 --output-dir runs/reach_priority7101
```

Evaluate each final checkpoint using identical held-out seeds and tolerance.
For example:

```powershell
python evaluate_physics_ddpg_her.py runs/reach_priority7101/final_model.zip --episodes 100 --max-steps 60 --seed 810000 --record-trajectories --output-dir runs/reach_priority7101/evaluation_100
```

Compare first-hit success, tip errors, action differences, interaction counts,
equilibrium/sensitivity work, and elapsed time jointly. The raw-sum arm is needed
to attribute differences to conflict handling. Hold hyperparameters fixed before
multi-seed comparison and the reserved 1,000-episode final evaluation. If the
checked update mostly skips or eliminates the auxiliary component, or if ordinary
DDPG performs equally well, report that; the method is not validated by merely
achieving a nonnegative gradient dot product. Fewer environment steps alone do
not demonstrate lower total computational cost.
