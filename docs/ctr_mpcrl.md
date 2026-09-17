# Learning the CTR MPC policy with RL

This experiment uses **MPC itself as the policy and value-function approximation**.
There is no demonstration dataset, imitation objective, or separate neural actor.
RL changes the MPC cost parameters from observed transitions and rewards. MPC
remains in the execution loop after training.

Sources:

- Gros and Zanon, [Data-driven Economic NMPC using Reinforcement Learning,
  arXiv:1904.04152v1](https://arxiv.org/abs/1904.04152v1), especially equations
  (21)-(24), (29), and the Q-learning update in Section IV.
- Gros and Zanon, [Towards Safe Reinforcement Learning Using NMPC and Policy
  Gradients: Part II - Deterministic Case,
  arXiv:1906.04034v1](https://arxiv.org/abs/1906.04034v1). This paper covers
  optimization-based deterministic policies, constrained exploration, and the
  additional requirements for robust safety and policy-gradient estimation.
- The [mpcrl library](https://mpc-reinforcement-learning.readthedocs.io/en/latest/),
  used as a dependency at release **1.4.1**. Its `LstdQLearningAgent`, NLP
  Lagrangian sensitivity and first-order `GradientDescent` optimizer support this
  implementation. `csnlp` and CasADi/IPOPT construct and solve the nonlinear MPC.

The implemented algorithm is the **first-order Q-learning route**, with a nominal
CTR model and fixed constraints. It does not implement the second paper's robust
DPG algorithm, its exploration-bias corrections, or its safety guarantees.

## What RL learns

Let `s=(q,g)`, where q is the six-joint configuration and g is the Cartesian goal.
The model integrates the existing continuous original-IVP mechanics to obtain
`p(q)`. The state uses normalized q and goal coordinates for numerical scaling.
The goal is held constant along each prediction horizon.

The internal objective is

\[
F_\theta = b_\theta + \sum_{k=0}^{H-1}\gamma^k
\left[\sum_{j=1}^{3} w_j e_{k,j}^2
+ r_\beta\|a_{k,1:3}\|^2+r_\alpha\|a_{k,4:6}\|^2\right]
+\gamma^H\sum_{j=1}^{3}p_j e_{H,j}^2,
\qquad e_k=\frac{p(q_k)-g}{0.01\,\mathrm{m}}.
\]

There are nine learnable parameters:

| Parameters | Initial values | Bounds | Effect |
|---|---|---|---|
| `stage` (3) | 1 each | [0.001, 20] | XYZ tracking costs |
| `move` (2) | 0.001 each | [0.00001, 2] | Translation/rotation effort costs |
| `terminal` (3) | 1 each | [0.001, 40] | Terminal cost approximation |
| `offset` (1) | 0 | [-100, 100] | Value-level correction; does not directly change the policy |

Tube geometry, elastic parameters, forward mechanics, joint bounds, and joint
increment caps are fixed. Positive cost weights alone do not establish stability.
The nine-parameter quadratic approximation is deliberately limited; the existence
results for a sufficiently rich MPC parameterization do not guarantee that this
particular parameterization attains the globally optimal policy.

Define

\[
V_\theta(s)=\min F_\theta,\quad
Q_\theta(s,a)=\min_{a_0=a}F_\theta,\quad
\pi_\theta(s)=a_0^\star.
\]

Both minimizations enforce the physical joint constraints and the full nonlinear
CTR prediction at each stage. This is a local nonlinear optimization, not a
guarantee of the global minimum. In particular, a poorly chosen horizon or a local
minimum can prevent reaching even if learning is implemented correctly.

The actual learning cost is `c=-reward`: 1 outside tolerance and 0 inside, exactly
from the original environment. The internal smooth MPC cost is the learnable
value-function approximation; it does **not** replace the environment reward.

\[
\delta_t=c_t+\gamma(1-d_t)V_\theta(s_{t+1})-Q_\theta(s_t,a_t),
\qquad
\theta\leftarrow\operatorname{boundedUpdate}
\left(\theta+\alpha\delta_t\nabla_\theta Q_\theta(s_t,a_t)\right).
\]

Here `d_t` means successful termination. A time-limit truncation retains its
bootstrap. The library's bundled continuing-task loop always bootstraps; this
repository therefore uses a custom episodic loop around its Q-learning components.
Parameters are held fixed across the Q and next-V solves used for one update.

`mpcrl` computes the parameter derivative from the optimized NLP Lagrangian.
Since the learned parameters occur only in the objective, this does not require
second derivatives of the rod ODE. IPOPT uses a limited-memory Hessian; the CTR
callback provides central differences (second-order one-sided differences at
box bounds) for first derivatives with respect to joints. The exact affine joint
integration is condensed, leaving only the action sequence as decision variables.
These derivatives are recomputed at candidate horizon states. This is not the old
actor Jacobian loss, nor propagation of one frozen current-state Jacobian.

## Exploration and execution

Training perturbs the objective by `noise.T @ a0`, then resolves the constrained
MPC. It does not add unconstrained action noise after optimization. Q and the
next-state V used in TD targets are always computed with zero perturbation.
This is exploratory behavior for off-policy Q-learning; it is not claimed to
satisfy the second paper's corrected deterministic-policy-gradient estimator.

The planner uses normalized actions in [-1,1]. Joint increments match the saved
original environment's action scales times `n_substeps`. Constraints keep the
nominal joint trajectory feasible, so its straight substep path is feasible. The
first planned command is additionally checked against the exact original repeated
projection before execution. Joint and dynamics residuals are checked independently
of the optimizer's success flag. Training stops and saves its current parameters
on a solver or update failure rather than learning from an unvalidated solution.
Evaluation retains numerical failures in the success denominator.

This version uses the **original zero-initial-torsion IVP plant** for direct RL
comparisons. The previously implemented `paper_bvp` free-tip formulation is a
different plant and is not silently substituted here. Prediction uses a private
forward model; planning does not mutate the actual environment.

## Install and run a diagnostic training pilot

Use your activated virtual environment (Python 3.11 is tested):

```powershell
python -m pip install -e ".[mpc-rl,test]"

python run_ctr_mpcrl.py train `
  --config runs/ivp_short_h2_seed10_v1/config.json `
  --steps 20 --seed 10 `
  --tolerance-m 0.0015 --max-steps 200 `
  --horizon 2 --learning-rate 0.001 `
  --exploration-strength 0.01 `
  --output-dir runs/ctr_mpcrl_h2_pilot
```

The saved source configuration must use continuous original-IVP mechanics, one
system and `physics_observation.mode=none`. This pilot uses a fixed tolerance,
not a hidden tolerance curriculum. It starts from the initial MPC weights; it does
not load or convert DDPG neural weights. `--freeze` performs the same exploratory
control run without updating parameters.

Each learning transition can need three nonlinear optimizations: exploratory
policy, fixed-action Q, and unperturbed next-V. Each optimizer calls the CTR model
many times. `--max-model-evaluations` bounds the number of uncached FK evaluations
per solve; `--max-iterations` bounds IPOPT iterations. Start with the pilot rather
than reusing a 600,000-step DDPG command. A completed 20-step run is an integration
diagnostic, not evidence of a learning advantage. Inspect solve time and failures
before increasing to 200 or more interactions in a new output directory.

## Evaluate initial and learned policies on identical tasks

Both snapshots are saved automatically. Use separate output directories:

```powershell
python run_ctr_mpcrl.py evaluate `
  --checkpoint runs/ctr_mpcrl_h2_pilot/checkpoint_initial.json `
  --episodes 5 --seed 920000 `
  --output-dir runs/ctr_mpcrl_h2_pilot/eval_initial

python run_ctr_mpcrl.py evaluate `
  --checkpoint runs/ctr_mpcrl_h2_pilot/checkpoint_final.json `
  --episodes 5 --seed 920000 `
  --output-dir runs/ctr_mpcrl_h2_pilot/eval_learned
```

Five tasks are an operational check only; use a larger matched task set and
multiple training seeds to estimate performance. Evaluation freezes parameters
and disables exploration. It preserves the source
environment fingerprint and records each task's joint/goal hash. Keep training
seeds separate from evaluation seeds. Compare initial versus learned MPC first:
comparing only with the old DDPG mixes the effect of planning with that of RL.
Then compare with DDPG at a matched interaction budget and report computation
separately. Longer experiments should use multiple training seeds.

`checkpoint_*.json` stores parameters, source configuration, formulation and
settings, plus an integrity fingerprint. These are inference checkpoints; they
do not contain replay, optimizer or RNG state for exact training resumption.

## Outputs and interpretation

- `updates.csv`: environment reward/error, terminal flags, Q and next-V, TD error,
  parameter-gradient norm, actual parameter change, optimizer status and compute.
- `episodes.csv`: seeds, task fingerprints, success, errors, failures and motion.
- `summary.json`: completion/abort reason, policy parameters, performance,
  optimizer solves, prediction-model calls and wall time.
- `checkpoint_initial.json`, periodic checkpoints and `checkpoint_final.json`.

Check all three levels: (1) solver convergence and action/plant consistency,
(2) finite TD gradients and actual bounded parameter changes, (3) improvement of
the frozen learned policy on held-out tasks. Lower TD error or changing parameters
alone does not establish a better controller. Partial training episodes are marked
and should not be compared with full frozen-policy evaluation episodes.

If fixed MPC already fails most tasks, first investigate nonlinear optimization,
prediction accuracy, horizon length and the need for a feasible reference path.
Learning a small set of cost weights may not fix a global planning failure. If
fixed MPC works but learned MPC deteriorates, investigate the value parameterization,
TD targets, learning rate and exploration before spending more interactions.

The nonlinear CTR experiment has no robust uncertainty set, recursive-feasibility
proof, terminal invariant set, or elastic-stability certificate. Consequently the
safety guarantees in arXiv:1906.04034v1 are not established by this implementation.
