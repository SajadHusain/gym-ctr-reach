> Historical implementation notes. For current training commands and defaults, use [clean_experiment.md](clean_experiment.md). The old root training entry points have been consolidated.

# Joint-constrained DDPG + HER with a Jacobian actor loss

The training and evaluation entry points now use `JointConstrainedReachEnv`.
The environment applies the full tube insertion constraints and fixed joint
increment caps, then solves the unloaded geometrically exact equilibrium.
The actor optionally receives the existing analytical-Jacobian tracking loss.

There is no branch tracker, elastic eigenvalue test, segment-order restriction,
midpoint/reverse check, action backtracking, goal-progress safeguard, or
Jacobian controller in this training/evaluation path. No gradient balancing,
Lyapunov penalty, or temporal smoothness loss was added.

## Install and run on Windows

Use your existing Python 3.11 virtual environment from the repository root:

```powershell
python -m pip install -e ".[physics-train,test]"
python -m pytest tests/test_simple_jacobian_rl.py tests/test_mechanics_jacobian.py tests/test_mechanics_study.py -q
```

Train a fresh guided actor for 10,000 interactions:

```powershell
python train_physics_ddpg_her.py --total-timesteps 10000 --learning-starts 200 --episode-steps 60 --physics-weight 0.1 --physics-final-weight 0.1 --seed 7101 --checkpoint-freq 1000 --output-dir runs/simple_jacobian7101
```

Evaluate the actor on 20 development tasks:

```powershell
python evaluate_physics_ddpg_her.py runs/simple_jacobian7101/final_model.zip --episodes 20 --max-steps 60 --seed 810000 --record-trajectories --output-dir runs/simple_jacobian7101/evaluation
```

The ordinary DDPG + HER control uses exactly the same plant:

```powershell
python train_physics_ddpg_her.py --total-timesteps 10000 --learning-starts 200 --episode-steps 60 --physics-weight 0 --physics-final-weight 0 --seed 7101 --checkpoint-freq 1000 --output-dir runs/simple_ddpg7101
python evaluate_physics_ddpg_her.py runs/simple_ddpg7101/final_model.zip --episodes 20 --max-steps 60 --seed 810000 --record-trajectories --output-dir runs/simple_ddpg7101/evaluation
```

Each run requires a new empty output directory. A 1,000-episode evaluation is
available by changing `--episodes`; it is not performed automatically. The
study runner also uses only the `ddpg` and `jacobian` arms and actor-only
evaluation. Create a new study directory; previous plans use a different plant.

## Exact action and loss definitions

The normalized command is `a` in `[-1, 1]^6`, with joint order
`[beta_1, beta_2, beta_3, alpha_1, alpha_2, alpha_3]`. Beta uses metres and alpha
uses radians. The fixed scales are

```
S = diag(0.001, 0.001, 0.001, 0.05, 0.05, 0.05).
```

First compute `dq = project(q + S*a) - q`, then divide `dq` by
`max(1, max(abs(dq / action_scales)))`. This second operation preserves the
per-joint caps when coupled projection moves more than one insertion joint.
Both the plant and differentiable actor-loss projection use this definition.
The new tip is the equilibrium solution at `q + dq`.

The solver starts from its deterministic default guess at each configuration;
previous torsion is not supplied. Consequently, no branch-selection history is
part of the policy state. This deliberately removes tracked-root continuity
and elastic-stability screening. The solver still checks convergence, finite
values and its boundary residual. Its numerical curvature continuation, if
needed, solves the same final equilibrium equations.

The actor objective is

\[
L_{actor}=-\mathbb{E}[Q(o,\mu(o))]+\lambda_J L_J,
\qquad
L_J=\operatorname{mean}_{valid}
\left\|\frac{J(q)\Delta q_\theta-d_{goal}}{0.002}\right\|^2,
\]

where `d_goal` is `0.5 * (goal - tip)` capped to a Euclidean norm of 0.002 m.
The default coefficient is constant at 0.1; supplying zero selects ordinary
DDPG + HER. The optional final-weight argument enables an explicit schedule.

`J(q)` is computed by the existing variational ODE and implicit shooting
derivatives at the current solved equilibrium. It is not a finite-difference
Jacobian. The solver-derived quantities are detached replay data. Gradients
pass through the actor, joint projection and matrix products. Every HER sample
recomputes the desired displacement using its relabelled goal.

At coincident segment events or a numerically singular shooting derivative,
the Jacobian can be unavailable. Such a source sample records
`jacobian_valid=false`; it still generates the normal joint-constrained
transition and remains in the real/HER critic update. It contributes zero to
the auxiliary loss. A fully masked batch has zero auxiliary gradient. No
finite-difference fallback or rejection of the robot command is introduced.
`updates.csv` records the valid fraction in each physics-update batch.

## Observations, replay and numerical failures

The three-tube observation contains nine normalized egocentric joint features
and the tolerance. The feature extractor appends the current goal error,
reconstructing it after HER relabelling. There are no basal-torsion features.
Old checkpoints have a different observation shape and must not be resumed in
this environment. They remain usable with the previous implementation.

Replay stores the submitted command, matching the raw-action actor and critic
queries. The executed joint increment is logged separately. Goal relabelling
recomputes reward and success termination. Success terminates; the episode
command limit truncates.

A failed equilibrium solve raises an error before recording a transition.
Training saves an interrupted checkpoint and a failure summary. Evaluation
records the failed episode. A failed solve is never silently turned into a
successful motion or a repeated held transition. An unavailable Jacobian is
handled separately and does not stop an otherwise valid equilibrium solve.

Random resets use feasible aligned joint configurations. Goals are generated
by four joint-constrained witness commands, with an equilibrium solve at their
endpoint. The baseline and guided arms share that goal distribution. It differs
from the previous filtered-witness distribution, so comparisons with the old
branch-tracked experiments are developmental rather than matched ablations.

Both arms retain the 3-by-256 actor/critic architecture, learning rate 0.0005,
discount 0.95, target update rate 0.001, batch size 128, Gaussian exploration
standard deviation 0.05, and four future HER goals. The baseline skips Jacobian
computation. Actor evaluation for both arms also skips Jacobian computation.
Solver costs, command variation, error and success remain recorded.

These are per-command quasi-static motions, without a physical timestep.
The Jacobian is a local approximation, and its tracking penalty does not
penalize temporal joint-action differences or establish a stability guarantee.
