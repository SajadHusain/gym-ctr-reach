# Analytical CTR Jacobian and matched exploration

Current entry points and defaults are in [the experiment guide](clean_experiment.md).
The equations below document the unchanged analytical mechanics. Historical
reaching-and-holding commands are not the default experiment.

This implementation uses the analytical sensitivity method of Burgner et al.,
*A Telerobotic System for Transnasal Surgery*, IEEE/ASME Transactions on
Mechatronics, 19(3), 996–1006 (2014),
[doi:10.1109/TMECH.2013.2265804](https://doi.org/10.1109/TMECH.2013.2265804).
Equation numbers below refer to that paper; other formulas are explicitly
derived here. The old positional Jacobian already used variational ODEs and
implicit shooting. This revision exposes the paper's spatial formulation and
uses its conversion to a point Jacobian for the actor loss. It is not a change
from finite differences to analytical differentiation.

## Model scope and frame correspondence

The repository models unloaded, inextensible, unshearable concentric tubes with
isotropic bending stiffness, prescribed intrinsic curvature and zero intrinsic
torsion. All four configured robots have intrinsic curvature
$u_i^*=[k_i,0,0]^T$. The general parameter class still accepts a second component;
the configured experiments do not use it.

The paper's Eqs. (1)–(4) also accommodate a distal force and moment. Here these
loads are zero. The resultant moment is zero, allowing the bending moment
unknowns to be eliminated. The shooting unknown is consequently only
$\eta=[u_{1z}(0),\ldots,u_{nz}(0)]^T$, rather than the paper's augmented vector
including two base bending moments. Wrench sensitivities and Eq. (8)'s compliance
matrix are not implemented.

The paper stores tube 1's material frame $R_1$. Our solver stores a twist-free
Bishop frame $R_B$ and absolute tube angles $\phi_i$ relative to it. Their exact
coordinate relationship is

$$R_i=R_B R_z(\phi_i),\qquad \theta_i^{\rm paper}=\phi_i-\phi_1.$$

Writing $K_i=E_iI_i$ and $C_i=G_iJ_i$ (here $J_i$ is the polar second moment of
area, not the robot Jacobian), zero resultant bending moment gives

$$\kappa_B=\frac{\sum_{i\in A(s)}K_i R_z(\phi_i)[k_i,0,0]^T}
                       {\sum_{i\in A(s)}K_i},\qquad (\kappa_B)_z=0.$$

This is the unloaded moment balance in the paper's Eq. (2), expressed in $R_B$.
Rotating into tube $i$ gives
$u_{iy}=-\sin(\phi_i)\kappa_{Bx}+\cos(\phi_i)\kappa_{By}$.
Together with Eq. (1), the corresponding ODEs are

$$p'=R_Be_3,\quad R_B'=R_B\widehat{\kappa_B},\quad
  \phi_i'=u_{iz},\quad u_{iz}'=-\frac{K_i}{C_i}k_i u_{iy}.$$

These equations apply while the relevant tube is present. The implementation
freezes each tube's torsion state after its own distal end. Eq. (3) becomes
$\phi_i(0)=\alpha_i-\beta_i\eta_i$, $p(0)=0$, $R_B(0)=I$; Eq. (4) becomes the
distal torque residual $b_i=C_i u_{iz}(\ell_i)=0$.

There is a notation trap in the supplied PDF: the last line of Eq. (2) prints
$[\sin\theta_i\ \cos\theta_i]$ for the rotation into tube $i$. With the paper's
$R_i=R_1R_z(\theta_i)$ convention, that row is instead
$[-\sin\theta_i\ \cos\theta_i]$, obtained directly from $R_z(-\theta_i)$.
The code uses this consistent rotation; the printed plus sign is not copied.

The current integrator is adaptive DOP853, not the paper's fixed-step RK4.
Agreement with these equations is not a reproduction of its runtime, loaded
robot validation or hardware results.

## From ODE sensitivities to the paper's Jacobian

Let $x=[q,\eta]$ contain independent joint and shooting variables before imposing
the distal boundary conditions. The repository orders joints as
$q=[\beta_1,\ldots,\beta_n,\alpha_1,\ldots,\alpha_n]$ in metres and radians.
The paper orders rotations before translations. The returned
`spatial_jacobian_alpha_beta` property supplies that column permutation.

For a segment ODE $y'=f(y,x)$, differentiating by $x_j$ gives the variational
equation $V_j'=f_y V_j+f_{x_j}$. Moving tube tips and curvature boundaries also
depend on translations. We map each segment to $t\in[0,1]$ and differentiate
$y_t=h(q)f(y)$, obtaining

$$\frac{dV_j}{dt}=h f_y V_j+f\,\frac{\partial h}{\partial x_j}.$$

Within a fixed ordering of material boundaries this includes the changes in
segment lengths. Initial sensitivities include derivatives of
$\alpha-\beta\eta$. The integration internally scales coordinates for numerical
accuracy, then converts all returned matrices back to physical units.

For $g=[R_1,p;0,1]$, the paper defines in Eq. (6)

$$E_j=(g_{,x_j}g^{-1})^\vee,\qquad
  V=\frac{\partial y}{\partial x},\qquad B=\frac{\partial b}{\partial x}.$$

`Sensitivity.paper` exposes `E_q`, `E_u`, `V_q`, `V_u`, `B_q`, `B_u` for the
unloaded reduced state. The `u` suffix denotes $\eta$ here. Along an equilibrium,
differentiating $b(q,\eta(q))=0$ yields

$$B_q+B_u\eta_q=0,\qquad \eta_q=-B_u^{-1}B_q.$$

Substitution into the pose derivative gives the paper's Eq. (7):

$$\boxed{J^s=E_q-E_u B_u^{-1}B_q.}$$

The implementation solves linear systems; it does not explicitly invert $B_u$.
It obtains $R_1$ and its sensitivity from $R_B R_z(\phi_1)$, so tube twist is
included in the orientation Jacobian.

Spatial twist is ordered $[v_s,\omega_s]$. Since
$dg\,g^{-1}$ has translational component $dp-dR_1R_1^Tp$, the tip velocity is
$\dot p=v_s+\omega_s\times p$. Therefore the point Jacobian used for reaching is

$$\boxed{J_p=J^s_{1:3,:}-\widehat p\,J^s_{4:6,:}.}$$

Taking just the first three rows of $J^s$ would be wrong. The body Jacobian is
also exposed, with blocks $[R_1^T J_p;R_1^T J^s_{4:6,:}]$.

No finite-difference equilibrium solves occur in this calculation. Central
differences appear only in tests, comparing both position and material-frame
rotation derivatives on all four robot parameter sets. Tests also cover
single-tube closed forms, input units, column order and the implicit boundary
identity.

An invertible boundary derivative is a mathematical requirement for this local
Jacobian, not an elastic-stability test. A singular derivative or coincident
moving interfaces can make it unavailable. Such a sample has zero auxiliary
weight while its transition remains available for RL. There is no branch
tracker, elastic-state filter, Jacobian controller or action backtracking in
this trainer. A failed equilibrium solve still stops the run rather than
inventing a successful transition.

## How the Jacobian guides the actor

For the current actor output $a_\theta=\mu_\theta(o,g)$, let $D$ contain the
physical joint-increment scales. The environment projects extensions to the
feasible set and enforces each increment cap. If $P$ denotes that joint
projection, the differentiable loss applies exactly the same map:

$$d=P(q+Da_\theta)-q,\qquad
  \Delta q_\theta=\frac{d}{\max(1,\|D^{-1}d\|_\infty)}.$$

Set $e=g-p$, $c=0.002$ m and $k=0.5$. Define
$d_{\rm des}=ke/\max(1,\|ke\|/c)$. The implemented objective is

$$L_{\rm actor}=-\mathbb E[Q(o,g,\mu_\theta(o,g))]
 +\lambda_J\mathbb E_{\rm valid}\left[
 \left\|\frac{J_p(q)\Delta q_\theta-d_{\rm des}}{c}\right\|^2\right].$$

The second term is our RL regularizer, inspired by the Cartesian tracking term
in the paper's Eq. (9). The paper does not train this DDPG objective. Division
by $c$ makes the term dimensionless and its weight independent of whether
displacements are expressed in metres or millimetres, provided all dimensional
quantities are converted together.

For a valid sample, with $r=J_p\Delta q_\theta-d_{\rm des}$, the chain rule gives

$$\nabla_\theta L_J=\frac{2}{c^2}
 (\partial_\theta\mu_\theta)^T
 (\partial_a\Delta q)^T J_p^T r.$$

PyTorch differentiates the actor, the selected joint-projection face, increment
scaling and matrix product. The source-state Jacobian, joints and goal target
are detached replay data. No gradient through the shooting solver is needed.
At changes of projection face this is a piecewise derivative, not a globally
smooth map.

HER changes the goal, reward and goal-dependent termination consistently. It
does not change the source configuration or its Jacobian. The guidance target
is recomputed from each relabelled goal. The loss is evaluated on the **current
actor output**, not the stored random action. The critic learns values of raw
proposals sent to the joint-constrained plant; replay preserves those proposal
labels even when projection changes execution.

This is a goal-directed kinematic penalty. It is not a temporal action
smoothness penalty or a Lyapunov constraint, and provides neither smoothness
nor convergence guarantees.

## Exploration and evaluation

Both arms now use the same sampling class. The default `--exploration-profile
paper` takes values from the repository's earlier `paper_config.py`:

| Quantity | Value |
|---|---|
| Warmup proposals | Uniform in all six action coordinates, for `--learning-starts` transitions (default 200) |
| Uniform probability after warmup | 0.294 per action vector |
| Otherwise | Actor output plus Gaussian noise, clipped to [-1, 1] |
| Normalized Gaussian standard deviations | [0.0018, 0.0018, 0.0018, 0.025, 0.025, 0.025] |
| Current physical increment caps | 0.001 m per translation; 0.05 rad per rotation |

These reproduce the earlier exploration **mixture and noise values**. They do
not reproduce every setting of the older training script, including its warmup,
plant action substeps or goal distribution. The previous simple-run sampler is
available as `--exploration-profile gaussian` (sigma 0.05, no uniform mixture
after warmup). `--noise-std` and `--random-exploration` explicitly override the
selected profile. Both study arms receive the same resolved settings.

Configuration and summary files record these values; summaries and checkpoint
budgets count warmup-uniform, mixture-uniform and noisy-policy proposals.
Uniform exploration gives every open region of the proposal box positive
probability. It does not guarantee finite-run coverage of reachable states:
joint projection, starts, goals and episode lengths still determine visitation.
Guidance never replaces these proposals with a controller action.

Evaluation uses the frozen actor with deterministic prediction and
`compute_jacobian=False`. There is no exploration noise, HER sampling, physics
loss, optimizer update or Jacobian computation during evaluation. Forward
equilibrium solves still supply the simulated tip position after each action.
Joint projection remains part of the plant.

The existing aligned starts and four-command local goal generator remain a
limited benchmark. A fixed command already solved many of those goals in the
previous audit. Higher success on this distribution cannot by itself establish
general inverse-kinematics learning, full-workspace accuracy or novelty.

## Why retain DDPG+HER instead of switching to PPO now?

[HER](https://arxiv.org/abs/1707.01495) was designed to reuse failed-goal
transitions with off-policy algorithms, which is useful when binary reaching
rewards are sparse. DDPG+HER is also the existing controlled baseline.

[PPO](https://arxiv.org/abs/1707.06347) is a valid comparison. Standard PPO
collects fresh rollouts and uses them for several optimization epochs; it does
not use ordinary HER replay. Goal relabelling changes the policy conditioning,
advantages and likelihood ratios, so an ordinary HER buffer cannot simply be
attached to PPO. PPO without HER avoids that bookkeeping, but may need more
new simulator transitions to learn from sparse rewards. That is a tradeoff to
measure, not a universal guarantee about either algorithm.

The same detached $J_p$ and tracking term could guide PPO's current policy mean,
added to its clipped-surrogate actor loss. Applying it only to stored rollout
actions would supply no actor gradient. PPO's stochastic policy/entropy governs
its exploration; this differs from the explicit DDPG action mixture above.
Neither PPO clipping nor a Jacobian tracking penalty proves closed-loop
stability. Switching algorithms also does not fix the local-goal shortcut.

The present change therefore keeps DDPG+HER for the matched comparison. PPO
would be a separate algorithm ablation, with an explicitly matched task and
interaction budget, not an unverified replacement advertised as a HER fix.

## Commands (PowerShell, from the repository root)

Local verification for this revision: all 34 focused tests passed. Two
128-transition training smoke runs each completed 64 optimizer updates with
finite parameters and zero replay-label error. Both sampled 64 warmup-uniform,
15 mixture-uniform and 49 noisy-policy actions. The guided arm performed 64
physics updates; the ordinary arm performed zero. Both saved actors completed
evaluation with zero sensitivity and stability calls. These are implementation
checks, not evidence of a learning-performance improvement. Tests ran on
Python 3.12, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.7.1 CPU, Gymnasium 1.3.0 and
Stable-Baselines3 2.9.0; the commands below check your Windows environment.

```powershell
git switch simple-jacobian-rl
git pull --ff-only
python -m pytest tests/test_paper_jacobian.py tests/test_physics_exploration.py tests/test_simple_jacobian_rl.py tests/test_simple_reaching_audit.py -q
```

Use new output directories to preserve earlier experiments. These two commands
match exploration, seed, architecture, task and transition budget:

```powershell
python train_physics_ddpg_her.py --total-timesteps 10000 --physics-weight 0 --exploration-profile paper --seed 7101 --checkpoint-freq 1000 --output-dir runs/paper_jacobian_ddpg7101
python train_physics_ddpg_her.py --total-timesteps 10000 --physics-weight 0.1 --exploration-profile paper --seed 7101 --checkpoint-freq 1000 --output-dir runs/paper_jacobian_guided7101
```

For a study, create a new plan with `run_physics_study.py --stage plan
--exploration-profile paper --output-dir runs/paper_jacobian_study`; then use
its `train`, `evaluate`, and `report` stages. Existing study manifests remain
frozen and are not silently migrated to the new sampler.
