# DDPG+HER and mechanical task-space guidance

## Two training entry points

`train_ddpg_her.py` trains ordinary DDPG+HER. Its default `--profile paper`
preserves the existing documented 2024 reproduction, including its original
quasi-static Model/Segment/Tube plant, 3x256 late-action critic, 3M budget,
20-to-1-mm tolerance curriculum, and saved exploration settings. See
[the provenance and remaining reproduction differences](paper_reproduction.md).
It is a Gymnasium/SB3 port, not a bitwise recreation of the author's software.

`train_ddpg_her.py --profile mechanics` is the ordinary RL control arm for the
hybrid experiment. `train_jacobian_ddpg_her.py` is its guided counterpart. These
share the SAME unloaded equilibrium plant, joint projection, goal distribution,
observations, reward, HER semantics, exploration, network and optimizer cadence.
Comparing the paper plant against the guided equilibrium plant would confound
the effect of the actor loss. Never pool their evaluation results.

| Setting | Preserved paper profile | Matched mechanics profiles |
|---|---|---|
| Plant | Original corrected quasi-static forward model | Unloaded torsional equilibrium BVP |
| Algorithm | DDPG + future HER, 4 sampled goals | Same |
| Hidden layers | 3x256 ReLU; critic action enters after first layer | Same |
| Actor/critic LR; gamma; tau | 0.0005; 0.95; 0.001 | Same |
| Batch; real-transition replay capacity | 256; 500,000 | Same |
| Rollout/update cadence | 100 transitions, then 50 updates | Same |
| Learning starts | Policy/noise/random mixture from step 1; wait for a complete-episode batch | Same |
| Exploration | Normalized Gaussian sigma [0.0018 x3, 0.025 x3]; uniform-action probability 0.294 | Same |
| Policy features | Egocentric joints, goal error, tolerance | Egocentric joints and error scaled by longest tube length |
| Goal sampling | Independent reachable joint configurations | Signed 2-to-8-command endpoint witnesses; randomized common rotation and independent offsets |
| Joint actions | 1 mm / 5 degrees, repeated 10 substeps | 1 mm / 0.05 rad per command; feasible joint projection |
| Task | Tolerance curriculum, 200-step cap | Fixed 1-mm first-hit reaching, 200-step cap |
| Default budget | 3,000,000 | 10,000 development transitions, explicitly not the paper training budget |

The last four differences are intentional properties of the existing mechanics
experiment, not consequences required by Gymnasium. Holding is optional legacy
functionality; it is not required or silently included in the primary endpoint.
The mechanics profiles use the paper's algorithm settings on a different plant
and task. Their 100/50 update cadence also differs from the old 1/1 pilot runs;
do not combine their learning curves. Tube parameters and solver tolerances have
not been changed by this cleanup. Numerical recovery remains bounded and logged.

## What the loss means

The image supplied with this request suggests a sum of a DDPG actor surrogate
and a Jacobian tracking term. The actor is written as mu_theta(o), not pi(s,a):
it produces the action; it does not take that action as a second policy input.

For a replay observation o at joint configuration q, let

$$a_\theta=\mu_\theta(o)\in[-1,1]^6,\qquad
 D=\operatorname{diag}(0.001,0.001,0.001,0.05,0.05,0.05).$$

The first three entries convert normalized commands into metres and the last
three into radians. With the plant's feasible-joint projection P_Q, define

$$v_\theta=P_Q(q+Da_\theta)-q,\qquad
 \Delta q_\theta=\frac{v_\theta}{\max(1,\|D^{-1}v_\theta\|_\infty)}.$$

This is exactly the joint update implemented by the mechanical environment;
it includes projection and the common step cap. The reward is unchanged:

$$r_t=-\mathbf{1}\{\|p_{t+1}-g\|>\epsilon_t\}.$$

HER recomputes reward and goal-dependent termination using each transition's
recorded tolerance. The critic uses only these transition rewards:

$$y=r+\gamma(1-d)Q_{\bar\phi}(o',\mu_{\bar\theta}(o')),
 \qquad L_C=\mathbb E[(Q_\phi(o,a)-y)^2].$$

Here d denotes task termination, including termination recomputed for a HER
goal. Time-limit truncation bootstraps. This is the usual DDPG target on the
modern replay convention, not a physics-shaped critic target. The actor terms are

$$L_R=-\mathbb E[Q_\phi(o,\mu_\theta(o))],$$
$$d^*(o)=\operatorname{clipnorm}(\alpha(g-p),d_{\max}),\qquad
 L_J=\mathbb E\left[\left\|\frac{J_p(q)\Delta q_\theta-d^*(o)}{c}\right\|_2^2\right].$$

The nominal weighted objective is L_R + lambda L_J. The image is recovered in
unconstrained physical-action coordinates with no target saturation, up to the
explicit normalization. In the implemented loss the action is a displacement,
not a velocity: no physical timestep, acceleration or kinetic energy is assumed.

Defaults are alpha=0.5 (`--physics-gain`), d_max=0.002 m (`--max-tip-step-m`),
and c=0.002 m (`--cartesian-scale-m`). These are declared design hyperparameters,
not numbers derived from tube mechanics or attributed to the paper. Alpha asks
for half the remaining error per local command before saturation. The cap
limits the requested local displacement. The scale makes the penalty
dimensionless; changing c with fixed cap rescales the loss by 1/c^2 without
changing the desired displacement. Tune these using development seeds and
Jacobian linearization-error checks, then freeze them before final testing.
Joint limits can make d* unattainable; the residual is not a feasibility proof.

## Analytical derivative and actor gradient

Production derivatives come from `mechanics/sensitivity.py`. The unloaded model
and frame correspondence are documented in
[paper_jacobian_and_exploration.md](paper_jacobian_and_exploration.md), including
the relationship to Burgner et al. (2014), Eqs. (6)-(7). Here is the mathematical
chain, with our tip-position notation rather than new paper equation numbers.

Let eta be the unknown base torsional strains, y'=f(y,q,eta) the piecewise IVP,
and b(q,eta)=0 the distal torsion residual. Differentiate the IVP with respect to
x=(q,eta). On each moving segment, mapped to t in [0,1] with length h(q),

$$\frac{dS_j}{dt}=h f_y S_j+h f_{x_j}+h_{x_j}f,\qquad S_j=\frac{\partial y}{\partial x_j}.$$

The actual implementation includes the initial-condition derivatives and
moving material boundaries. Implicit differentiation of the distal residual gives

$$\eta_q=-B_u^{-1}B_q,\qquad
 J_p=P_q-P_u B_u^{-1}B_q,$$

where B_q=b_q, B_u=b_eta, P_q=p_q at fixed eta, and P_u=p_eta at fixed q.
The code solves linear systems, not explicit matrix inverses. It also exposes
the equivalent spatial-pose blocks from the paper and converts the spatial
Jacobian to tip-position velocity. Finite differences are used only as a test
oracle, not to generate production training labels.

J_p is computed at each visited SOURCE state when guidance is active, then
stored alongside that transition. During a minibatch update q, J_p, D and the
sampled goal are fixed data. The current actor is evaluated again. Within one
active projection region the chain rule is

$$\nabla_\theta L_J=\frac{2}{c^2}\mathbb E\left[
 (D_\theta\mu_\theta)^T(D_a\Delta q)^T J_p^T
 (J_p\Delta q_\theta-d^*)\right].$$

PyTorch differentiates the actor, selected projection face, step cap and matrix
product. It does not differentiate the previously visited state through earlier
policy actions. That is an off-policy sampled-state actor update, not a
differentiable trajectory rollout. J_p changes between collected states, while
remaining constant with respect to this update's current actor parameters.
At active-set ties the selected branch derivative is not a unique classical
derivative. Missing/singular sensitivities mask only L_J for that sample; its
transition still trains the critic. HER substitutes a goal and recomputes d*,
while keeping the source q and its J_p paired with the original transition.

## Keeping the RL objective primary

The default `rl_priority` integration retains the previously tested rule. Write
g_R=grad_theta L_R and g_J=grad_theta L_J. Remove only the conflicting component:

$$\tilde g_J=g_J-\frac{\min(0,g_R^Tg_J)}{\|g_R\|^2}g_R.$$

At g_R=0 the auxiliary update is disabled. Limit the weighted auxiliary norm
to `--physics-max-aux-ratio` times ||g_R||, then form g_R+lambda tilde_g_J.
This is an asymmetric adaptation of
[PCGrad's conflict projection](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html),
not a new optimality theorem. Gradient alignment is measured in actor parameter
space. It does not imply improvement in actual return.

Because Adam changes the direction and updates have finite size, candidate
parameter steps are checked against the SAME minibatch and fixed critic.
Acceptance requires g_R^T delta_theta < 0 and a finite nonincreasing L_R.
Otherwise the step is shortened, an RL-only step is tried, or the update is
skipped with optimizer state restored. This is a parameter-update check, not a
robot action filter. It certifies neither physical stability nor generalization.
The projected update is NOT the exact gradient of the scalar sum in the image;
`--physics-integration sum` is the explicitly unchecked weighted-sum ablation.

In new guided runs lambda decreases linearly from 0.1 to zero over the first
half of the requested interaction budget by default. The remaining training
uses ordinary DDPG updates and no new sensitivity calls. All schedules are saved;
fixed-weight experiments require an explicit `--physics-final-weight` override.
Annealing removes the permanent auxiliary objective, but cannot undo all biases
in the learned representation or replay distribution. No return guarantee is
claimed. `updates.csv` records alignment, norm ratio, rejected/fallback/skipped
updates and surrogate changes; values aggregate each optimizer block.

## Does this just learn a Jacobian controller?

The question cannot be answered by the form of the loss or a high success rate.
Minimizing L_J alone tries to satisfy J_p Delta_q = d*. Without limits or damping,
its solutions contain J_p^+ d* plus null-space motion. Thus it is locally related
to classical differential inverse kinematics even though it has no teacher
action label or inverse-Jacobian operation in training. That relationship must
be acknowledged. A position Jacobian is also blind to null-space oscillations:
this loss alone does not ensure smooth joint actions.

Use four matched training arms and an evaluation-only classical controller:

| Arm | What determines actor updates | What it tests |
|---|---|---|
| ddpg | Ordinary DDPG+HER | Interaction-learning baseline |
| jacobian | RL-priority mechanical guidance, then RL-only tail | Proposed hybrid |
| mechanics_only | L_J only; no Q gradient, Q filtering or RL acceptance | Whether the physics objective alone explains performance |
| rl_only_checked | L_R only with the same finite Adam-step checks | Whether the optimizer checks explain the difference |
| Classical controller | Online constrained damped Jacobian QP; no learned actor | Whether a network/RL is useful for this task at all |

The mechanics-only arm still fits its critic to rewards for matched diagnostics,
but that critic never influences actor updates. Both learned arms use real
simulated transitions, the same HER sampler, and the same exploration mechanism.
The classical controller solves

$$\min_a \|(J_pDa-d^*)/c\|^2+\rho^2\|a\|^2,
 \quad -1\le a\le1,\quad q+Da\in\mathcal Q.$$

It is a convex constrained damped least-squares comparator, implemented only
in `mechanics/classical.py` and the evaluator. Its damping is saved explicitly.
It never supplies replay actions or training labels. Failure to compute its
Jacobian or solve its QP is reported as failure, not a successful held state.

## Reproducible commands

Use fresh output directories. The two commands below are the primary matched
comparison, NOT the separate historical-paper experiment:

```powershell
python train_ddpg_her.py --profile mechanics --total-timesteps 10000 --seed 7101 --output-dir runs/clean_ddpg7101
python train_jacobian_ddpg_her.py --total-timesteps 10000 --seed 7101 --output-dir runs/clean_guided7101
```

For the control ablations:

```powershell
python train_jacobian_ddpg_her.py --ablation mechanics-only --total-timesteps 10000 --seed 7101 --output-dir runs/clean_mechanics_only7101
python train_jacobian_ddpg_her.py --ablation rl-only-checked --total-timesteps 10000 --seed 7101 --output-dir runs/clean_checked_rl7101
```

For ordinary frozen-actor evaluation (no sensitivity calls), apply this to each
final checkpoint with the SAME task seeds, horizon and tolerance:

```powershell
python evaluate_physics_ddpg_her.py runs/clean_guided7101/final_model.zip --episodes 1000 --seed 910000 --output-dir runs/clean_guided7101/final_1000
```

The evaluation horizon defaults to the checkpoint's horizon. To evaluate the
classical comparator on the same tasks, use `--modes jacobian` in a separate
output directory. To measure differences between actor actions and classical
actions at the actor's own states, use `--modes actor --controller-audit` on a
small development set. This flag adds diagnostic sensitivity/QP calls without
changing actor actions; do not quote its runtime as frozen-policy inference time.
Missing diagnostic derivatives are counted and are not treated as agreement.

A frozen study plan supports matched seeds, all four training arms, checkpoint
curves, and separate final evaluation. Change its settings only in a new folder:

```powershell
python run_physics_study.py --stage plan --arms ddpg jacobian mechanics_only rl_only_checked --seeds 7100 7101 7102 --total-timesteps 10000 --physics-anneal-steps 5000 --output-dir runs/clean_study
python run_physics_study.py --stage train --output-dir runs/clean_study
python run_physics_study.py --stage evaluate --output-dir runs/clean_study
python run_physics_study.py --stage report --output-dir runs/clean_study
```

After development choices are frozen, `--stage final-evaluate` evaluates only
final checkpoints on the separate 1000-task seed block and writes the report.
The classical comparator is evaluated once per task set/configuration, not once
per training seed. Old frozen study plans remain readable for reports but cannot
silently continue training against changed source code.

Report success and learning-curve area versus environment interactions AND total
equilibrium/sensitivity work, wall time and update count. Compare action
differences only alongside task success, including paired successful tasks;
a stationary failing policy is not a smoothness improvement. Without physical
dt, these are command differences, not physical accelerations or jerk.

At least multiple independent training seeds are needed for a performance claim;
1000 task episodes for one policy do not measure training-seed variability.
Actor/controller agreement is descriptive. If the hybrid is indistinguishable
from the mechanics-only arm, or offers no useful accuracy/robustness/runtime
tradeoff against the classical controller, this task does not establish the
claimed benefit of RL. Reaching alone is a weak test of long-horizon advantages;
any later path, obstacle or load experiment needs its own validated task/model.
Do not add those tasks silently to this comparison.

Finally, the BVP can admit multiple equilibria. The analytical derivative is
local to the selected root, and deterministic restarts are not a physical
stability certificate. Solver failures, derivative masks and root recovery counts
remain visible. A passing smoke test establishes implementation integrity, not
sample efficiency, global convergence, publication novelty or physical validity.
