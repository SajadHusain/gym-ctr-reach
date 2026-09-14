# Matched DDPG/HER using the original forward model

This experiment removes nonlinear equilibrium shooting from both training
arms. It uses `envs/model.py`, the original joint sampler, egocentric features,
action update, sparse reward, and goal-tolerance schedule. The guided arm adds
an analytical source-state sensitivity and an actor update objective. No
Jacobian controller selects collection actions or evaluation actions.

This is a change of **plant**, not another increase of the shooting budget.
Do not continue an equilibrium-model checkpoint in this environment. Keep
previous results and start new directories. A short successful software test
does not establish convergence at 1 mm.

## Which forward model?

The original solver sets all three initial torsional strains to zero, then
integrates an IVP. It does not solve for the unknown initial strains needed to
satisfy distal zero-torque boundary conditions. Consequently this experiment
must not be described as the fully solved free-tip equilibrium BVP, or as an
exact implementation of an externally loaded equilibrium Jacobian paper.

Two numerical variants are explicit:

| `--segment-mode` | Segment lengths | Forward integration | Purpose |
|---|---|---|---|
| `legacy` (default) | Original floor to 10 micrometres | Original RK45 defaults | Preserve the existing paper-profile transition |
| `continuous` | Unrounded lengths | DOP853, rtol 1e-8, atol 1e-10 | A meaningful local extension derivative for the guided comparison |

Both retain the original 1-micrometre subtraction from each segment integration
end. Removing that offset would be another model change. Continuous mode is
an explicit numerical variant of the original IVP, not bitwise reproduction.
Both arms must select the same variant. The existing `--profile paper` command
retains its legacy defaults.

Within a legacy rounding cell, individual segment lengths are constant even
when tube extensions change. Differentiating an unrounded geometry and calling
it the derivative of that rounded forward map would be incorrect. Legacy mode
therefore uses the literal cell derivative; at rounding boundaries it marks
the auxiliary derivative unavailable. Continuous mode is recommended for the
new guided comparison, with an ordinary baseline on that same model.

## Analytical IVP sensitivities

These equations are derived by differentiating the repository's `Model._ode_eq`;
they are not asserted to be a verbatim equation from the equilibrium papers.
The state used by the source code is

\[
y=(u_{1z},u_{2z},u_{3z},\alpha_1,\alpha_2,\alpha_3,p^T,\operatorname{vec}_r(R)^T)^T.
\]

Here `vec_r` means the row-major array ordering used in NumPy, and there are 18
scalars. In a constant-property segment, let
\(w_j=E_jI_j/\sum_k E_kI_k\), \(d_{ij}=\alpha_i-\alpha_j\).
The code calculates

\[
u_{ix}=\sum_jw_j(\bar u_{jx}\cos d_{ij}+\bar u_{jy}\sin d_{ij}),\quad
u_{iy}=\sum_jw_j(-\bar u_{jx}\sin d_{ij}+\bar u_{jy}\cos d_{ij}).
\]

The repository robot configurations have intrinsic curvature along x; their
\(\bar u_{jy}\) values are zero. Keeping the general formula in the derivative
implementation does not add y-precurvature to those configurations.

For active tubes, the implemented equations are

\[
u'_{iz}=\frac{E_iI_i}{G_iJ_i}(u_{ix}\bar u_{iy}-u_{iy}\bar u_{ix}),\quad
\alpha_i'=u_{iz},\quad p'=Re_z,\quad R'=R\widehat u_1.
\]

Inactive tube strain/angle derivatives are zero, exactly as in the source. The
initial state has \(u_z(0)=0\), \(p(0)=0\), \(R(0)=R_z(q_4)\), and
\(\alpha(0)=q_{4:6}\). These initial conditions are differentiated as well.

Write this segment ODE as \(y'=F_i(y)\). For an interval of length
\(h_i(q)=S_{i+1}(q)-S_i(q)-10^{-6}\), use a fixed coordinate
\(\tau\in[0,1]\). With \(Z=\partial y/\partial q\), the chain rule gives

\[
\frac{dy}{d\tau}=h_iF_i(y),\qquad
\frac{dZ}{d\tau}=h_iF_{i,y}Z+F_i(y)\frac{\partial h_i}{\partial q}.
\]

The last outer-product term accounts for moving tube ends and curvature
boundaries. Omitting it would give incorrect extension columns. `Segment`
provides the endpoint derivatives within a fixed ordering; the augmented IVP
integrates the state and all six sensitivity columns together. At the last
endpoint, \(J_p=Z_{p,:}\). No finite differences or equilibrium root searches
are used to generate training Jacobians. Tests use independently perturbed tip
solutions to check this analytical implementation.

An event-order collision, a rounding switch, or a material discrepancy between
the variational solution and the forward tip makes that sample's Jacobian
invalid. It contributes no auxiliary gradient, but its original RL transition
is retained. This is a derivative-domain check, not an elastic-stability test,
branch-selection algorithm, or action safeguard. Invalid counts and reasons
are reported.

## Actions, loss, and gradients

The policy outputs a six-dimensional normalized proposal \(a\in[-1,1]^6\).
For each of the saved `n_substeps`, the environment adds

\[
D a,\qquad D=\operatorname{diag}(d_\beta,d_\beta,d_\beta,
 d_\alpha,d_\alpha,d_\alpha),
\]

then applies the original box limits and ordered tube-extension min/max
operations. \(d_\beta\) is in metres and \(d_\alpha\) in radians; these are
displacements, not velocities. Define the complete repeated operation as
\(q^+=T(q,a)\). The loss uses \(\Delta q_\theta=T(q,\mu_\theta(o))-q\), not
an unconstrained `n_substeps * D * action`. PyTorch differentiates this same
piecewise action map. At a min/max tie it uses the operator's subgradient.

At each collected source state, the guided environment computes \(J_p(q_t)\)
before applying the proposal and stores it alongside that transition. It is
not one fixed Jacobian per episode. During replay optimization the sampled
\(q_t,J_p(q_t)\) are constants; gradients flow through the current actor and
\(T(q_t,a)\). We do not differentiate backward through the trajectory that
originally produced \(q_t\). This is the ordinary off-policy state convention,
and is the exact derivative of the stated *local surrogate*, not a claim to
differentiate the nonlinear future simulator trajectory.

Let \(e=g-p\), and define a capped local target

\[
d_{\rm des}=k e\min\left(1,\frac{d_{\max}}{\|ke\|}\right),\qquad
L_J=\mathbb E\left[\left\|\frac{J_p(q)\Delta q_\theta-d_{\rm des}}{c}\right\|^2\right].
\]

The zero-error target is zero. Defaults are \(k=0.5\), \(d_{\max}=2\) mm,
and \(c=2\) mm. These are explicit experimental hyperparameters, not constants
derived from rod mechanics. `c` makes the residual dimensionless, and changing
it changes the effective weighting unless lambda is adjusted. The cap controls
the desired local displacement, not the actual joint command. Report and ablate
these choices; the scalar gain and cap are not stability guarantees.

The nominal actor objective is

\[
L_R+\lambda L_J=-\mathbb E[Q_\phi(o,\mu_\theta(o))]+\lambda L_J.
\]

Reward and critic targets are unchanged: sparse 0/-1 reward, ordinary discounted
HER critic updates. HER changes the desired goal and recomputes historical
reward and termination; it retains the source joint configuration and Jacobian.
The target \(d_{\rm des}\) is recomputed from the relabelled goal at update time.
Replay stores the normalized proposal, including proposals altered by joint
constraints. The plant's projection is part of the transition. A float32
round-trip difference below 1e-6 from SB3 action unscaling is allowed and logged.

`--physics-integration sum` differentiates the nominal sum directly. The default
`rl_priority` instead uses the existing projected-gradient integration: remove
the component of the auxiliary gradient opposing the RL gradient, cap its
norm relative to the RL gradient, and check the finite Adam step against that
minibatch's fixed-critic RL surrogate. Backtracking/RL-only fallback/skip counts
are logged. This is an update rule, not exactly gradient descent on the nominal
weighted sum. It ensures neither actual-return improvement nor physical
closed-loop stability. See `rl_priority_jacobian.md` for its precise checks.

New guided commands default to `--physics-max-aux-ratio 0.1`, bounding the
weighted auxiliary parameter gradient at 10% of the RL-gradient norm before
Adam. The omitted final weight is min(initial weight, 0.01), so default guidance
remains nonzero after the curriculum. Zero-weight baselines and explicit
overrides are preserved, as are settings loaded from model checkpoints.
These are experimental defaults, not evidence of improved performance.
The cap is not a bound on Adam's parameter displacement, and while it is
saturated, annealing the raw weight may not reduce the effective contribution.
The update metrics `auxiliary_norm_clipped`, `auxiliary_clip_scale` and
`auxiliary_norm_cap` expose this behavior alongside the existing before/after
norm ratios. They describe gradient combination, not robot-motion guarantees.

The loss does not use inverse-Jacobian action labels, but it remains closely
related to local differential kinematics. To show that RL contributes, compare
ordinary DDPG, guided DDPG, `--actor-objective mechanics_only`, and
`--actor-objective rl_only_checked`. A classical-controller comparison must use
this same selected plant and action limits; old equilibrium-controller scores
are not comparable. Better RL performance, generalization, and smoothness remain
experimental questions. The loss contains no explicit temporal smoothness term.

## Configuration, curriculum, and commands

Both public scripts accept `--profile original`. Without overrides the resolved
settings come from `paper_configuration()`: three hidden layers of 256, late
critic action injection, 0.0005 learning rates, gamma 0.95, tau 0.001, 100
collection steps per 50 gradient updates, batch 256, four future HER goals,
the saved Gaussian noise and 0.294 uniform-action mixture, 200-step episodes,
and 10 action substeps. There is no silent change to small local witness goals.
Starts and reachable goals are sampled independently using the original sampler.

The default exponential curriculum is 20 mm to 1 mm over 1.5M interactions,
with a 3M training budget. `--total-timesteps` alone does **not** compress this
schedule. Both can be explicitly overridden. The schedule is

\[
\epsilon(t)=\epsilon_0(\epsilon_f/\epsilon_0)^{\min(t/N,1)}.
\]

For example, the following **new matched experiment**, not an assertion about
the paper's exact training schedule, uses 600k interactions and reaches the
final tolerance at 200k. All settings and full tube parameters are saved in
`config.json` and inside each checkpoint. `--config` can import a complete
paper-profile `run_config.json` or this profile's saved config; command-line
overrides are then applied. It does not resume model weights/replay.

```powershell
$ctrIvpArgs = @(
  '--profile', 'original',
  '--segment-mode', 'continuous',
  '--total-timesteps', '600000',
  '--tolerance-curriculum', 'exponential',
  '--initial-tolerance-m', '0.02',
  '--tolerance-m', '0.001',
  '--curriculum-steps', '200000',
  '--checkpoint-freq', '10000',
  '--progress-every', '1000',
  '--seed', '7101'
)
python train_ddpg_her.py @ctrIvpArgs --output-dir runs/ivp_ddpg7101_v1
python train_jacobian_ddpg_her.py @ctrIvpArgs --physics-integration rl_priority --physics-max-aux-ratio 0.1 --physics-weight 0.1 --physics-final-weight 0.01 --physics-anneal-steps 200000 --output-dir runs/ivp_guided7101_v2
```

The raw weight decays linearly to 0.01 by 200k and remains there during the
1-mm stage. Sensitivity collection continues, increasing simulation cost
relative to switching guidance off. Use a fresh output directory: these commands
train new models, not resume old checkpoints. Compare caps 0.05 and 0.1 with
ordinary DDPG across at least three matched training seeds (for example 7100,
7101, 7102); keep plant, budgets and evaluation targets identical. Include the
checked-RL-only arm to separate Jacobian effects from the update safeguard.

The previous switch-off experiment remains reproducible with
`--physics-max-aux-ratio 1 --physics-final-weight 0 --physics-anneal-steps 200000`.
With final weight zero, ordinary RL updates and derivative shutdown at 200k
are intentionally preserved. The local-Jacobian approximation and unavailable
derivatives have not been changed by the new guidance defaults.

First compare checkpoints with 100 development episodes at seed 810000. Once
the protocol/checkpoint selection is fixed, use 1000 held-out episodes at seed
910000. The commands below use the final checkpoint and 1-mm evaluation:

```powershell
python evaluate_original_ddpg_her.py runs/ivp_ddpg7101_v1/final_model.zip --episodes 1000 --seed 910000 --tolerance-m 0.001 --progress-every 50 --output-dir runs/ivp_ddpg7101_v1/evaluation_1000
python evaluate_original_ddpg_her.py runs/ivp_guided7101_v2/final_model.zip --episodes 1000 --seed 910000 --tolerance-m 0.001 --progress-every 50 --output-dir runs/ivp_guided7101_v2/evaluation_1000
```

Evaluation restores the exact saved plant, fixed final tolerance, and saved
episode horizon. The policy is frozen and deterministic; sensitivity calls
must be zero. `--record-trajectories` saves the joint proposals, applied
increments, tips, and goals. Every episode has a task fingerprint to verify
paired start/goal configurations. Failed episodes remain in the success-rate
denominator; initially satisfied goals are reported. Error means exclude
numerically failed episodes and explicitly state that convention.

Motion metrics report normalized action-change RMS and separate physical
translation/rotation increment changes. They are per-decision statistics,
not velocity, acceleration, or jerk without a physical timestep. Compare them
alongside success and episode length; a motionless failed policy can look smooth.

Use fresh output directories. Training aborts and saves diagnostics if the
forward IVP fails; it does not invent a successful transition. No numerical
solver is guaranteed never to fail. Failed sensitivity calculations only mask
the auxiliary term. Compare transitions, forward calls, sensitivity cost,
wall time, invalid derivative rate, and linearization error. A Jacobian solve
is extra computation even though it is not an environment interaction.

The included checks cover legacy transition parity, the analytical ODE
derivative, all four robots' tip Jacobians, action constraints/substeps, gradient
checks, HER relabelling, the reported shooting-failure configuration, Gymnasium
compliance, and exact zero-weight update parity. Run:

```powershell
python -m pytest tests/test_original_ivp.py tests/test_env.py tests/test_actor_gradients.py -q
```

## Recorded implementation verification

The accompanying `docs/validation/original_ivp_integration.json` records two
4,000-transition runs, 1,850 optimizer updates per run, finite parameters, and
zero forward-solver failures. Ten evaluation tasks were identical across arms,
and both evaluators used zero sensitivity calls. Neither short-run policy
reached 1 mm on those ten tasks. These are software checks, not evidence of
learning superiority. The guided run had unavailable derivatives on about 53%
of collected states, and its mean finite-action linearization error was 8.56 mm.
Large exploratory commands and active constraints therefore remain material
limitations of local guidance, even with an accurate analytical derivative.

`docs/validation/original_ivp_numerics.json` compares 64 seeded configurations
across all four robots: the largest legacy/continuous tip difference was
0.0588 mm, and the largest continuous forward/variational tip discrepancy was
3.88e-10 m. This is a finite numerical check, not a universal error bound.
