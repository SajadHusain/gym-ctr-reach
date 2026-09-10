# Investigating high success on the local reaching task

The current success rate measures entering a 1 mm goal ball before the episode
limit. It does not measure how accurately the robot holds that point after the
episode ends, or performance across arbitrary starts and goals.

The reset distribution is deliberately local. All starting tube rotations are
zero. Each goal is produced by four commands with insertion increments of
-0.5 mm per tube and rotation increments near +0.04 rad per tube, subject to
joint constraints and action caps. The variation in those rotations is small.
Consequently, an approximately correct direction and travel distance can be
predicted from the task distribution before observing the individual goal.

This is a limitation of the benchmark introduced for the pilot, not evidence
that a high reported rate is fabricated. Keep the trained models and test for
shortcuts before expanding or retraining the environment.

## What the implemented Jacobian loss does

For source configuration q, let p=f(q) be the tip from the equilibrium solver
and J(q)=partial f/partial q the 3-by-6 positional Jacobian. The columns follow
the three insertion coordinates in metres and three rotation coordinates in
radians. The sensitivity module integrates the variational ODE and accounts
for the implicit shooting boundary conditions.

The actor produces a normalized command a=mu_theta(o). Define C(q,a) as the
actual constrained joint increment: project q+S*a into the joint set, subtract
q, and apply the uniform per-joint increment cap. The diagonal entries of S
are 0.001 m for insertion and 0.05 rad for rotation.

First-order expansion of the same forward model gives

\[
f(q+C(q,a))-f(q)\simeq J(q)C(q,a).
\]

The implemented target is

\[
d_{des}=\operatorname{cap}_{\|\cdot\|\le d_s}
          \bigl(0.5(g-p)\bigr),\qquad d_s=0.002\;\mathrm{m}.
\]

The actor objective, averaged over the replay minibatch, is

\[
L_{actor}=-\mathbb{E}[Q(o,\mu_\theta(o))]+\lambda_J L_J,
\qquad
L_J=\frac{1}{N_v}\sum_{i\in valid}
\left\|\frac{J_i C(q_i,\mu_\theta(o_i))-d_{des,i}}{d_s}\right\|^2.
\]

Here lambda_J=0.1 in the guided run. N_v counts valid source Jacobians. If it
is zero, the auxiliary term and its gradient are zero, while the RL update
still proceeds. HER relabels the target g; d_des is recomputed for that goal,
while q and J still refer to the same physical source transition.

The scalar division by d_s makes the residual dimensionless. It does not
change metres into radians: the joint scales and the corresponding Jacobian
columns already account for those different coordinates. It sets the size
of the regularizer relative to Q; lambda_J must be interpreted with this
normalization.

For residual r_i=J_i*C(q_i,mu_theta(o_i))-d_des,i, the chain rule gives

\[
\nabla_\theta L_J = \frac{2}{N_v d_s^2}\sum_{i\in valid}
\left(\frac{\partial \mu_\theta(o_i)}{\partial\theta}\right)^T
\left(\frac{\partial C(q_i,a)}{\partial a}\right)^T
J_i^T r_i.
\]

This expression applies within a selected projection region. The joint
projection is piecewise differentiable; the implementation differentiates
through its selected face and step cap. Source q, J and d_des are detached
constants. PyTorch differentiates the actor, constrained increment and matrix
product. It does not backpropagate through the equilibrium ODE solver.

For example, if d_des is 2 mm along +x, a command predicted to move 1 mm along
-x has normalized squared loss 2.25. A command predicted to move 1.8 mm along
+x has loss 0.01. Their policy gradients encourage different future outputs
at similar observations. The RL term continues to value actual simulated
outcomes and future returns. This is a direct actor tracking regularizer,
not a learned auxiliary prediction head or a temporal action-smoothness loss.

The sparse reward provides little distinction among unsuccessful moves. The
Jacobian term supplies a local, directional training signal. This explains
why it can improve learning in this task. It does not make the linearization
exact for every finite action or provide a convergence guarantee.

Code references:

- [Loss and gradient computation](../ctr_reach_envs/mechanics/rl_jacobian.py): `ProjectedJacobianLoss.forward` and `JacobianDDPG.train`.
- [Source Jacobian collection and physical transition](../ctr_reach_envs/mechanics/simple_rl_env.py): `_physics_context` and `step`.
- [Goal error presented to the actor](../ctr_reach_envs/mechanics/rl_policy.py): `EquilibriumStateExtractor.forward`. This input uses tip minus goal; the loss target uses goal minus tip.
- [Ordinary evaluation](../evaluate_physics_ddpg_her.py): `compute_jacobian=False` and `model.predict(..., deterministic=True)`.

## Evaluation uses fixed policy weights

Training stores source Jacobians in replay and uses them during actor updates.
The Jacobian is not part of the policy observation. During evaluation, the
saved actor maps observations directly to commands without optimizer updates,
Jacobian calculations, inverse-Jacobian control or controller safeguards.
The equilibrium solver still computes the actual next tip from the command.
Thus the learned weights retain the effect of training guidance.

The audit additionally hashes the policy tensors before and after evaluation,
checks that the optimizer update count did not change, and reports sensitivity
and stability call counts. It does not resume training.

## Run the audit

From the same branch and activated virtual environment:

```powershell
git pull --ff-only
python -m pytest tests/test_simple_reaching_audit.py -q
python audit_simple_reaching.py runs/simple_jacobian7101/final_model.zip --episodes 20 --max-steps 8 --seed 920000 --output-dir runs/simple_jacobian7101/audit_20
```

This is an intentionally short diagnostic, with an eight-command limit and
five continuation commands after a normal actor first succeeds. It is not
the standard 60-command benchmark. Set `--max-steps 60` explicitly if needed
for a separate comparison. Choose a fresh output directory on each run.

| Mode | Intervention | Interpretation |
|---|---|---|
| `actor` | Correct goal, normal deterministic actor | Reference for this diagnostic budget |
| `fixed` | Repeat [-0.5,-0.5,-0.5,0.8,0.8,0.8] four times in normalized coordinates | Measures success achievable without learning or goal feedback |
| `goal_zero` | Show the actor its current tip as the goal; retain the real goal for scoring | Checks whether goal information is necessary for observed reaching |
| `goal_shuffled` | Cyclically swap goals between tasks in policy input only | Tests sensitivity to deliberately incorrect target information |
| `reverse_actor` | Reachable nearby goals generated with negative rotation commands | Checks a direction outside the current training distribution |

The first four modes share initial configurations and true targets. Reverse
targets use the same initial configurations and separate labels. Goal
corruption changes only a copied policy observation. The true scoring goal
remains fixed. Shuffled/zero-goal and reversed-target tests deliberately
change the policy's input distribution; failure in those modes is not itself
evidence of an implementation bug.

For normal actor successes, the hold phase resumes the simulator at the
reached q with the same target and asks the policy for five further commands.
It reports the maximum error and whether all those samples remain inside
1 mm. This continuation is explicitly separate from the terminated reaching
episode. It does not establish a dynamics or stability theorem.

Every step recomputes Euclidean tip-to-target distance independently of the
environment's success Boolean and checks that the goal did not change. This
validates accounting; it is not independent physical-model validation.

Outputs are `summary.json`, `episodes.csv`, `trajectories.csv` and `tasks.json`.
Failures remain visible and produce an unsuccessful audit exit status. A
failure before a verified hit is not counted as a successful reach. A failure
during holding preserves the recorded reach outcome and is reported separately.
No branch checks or training losses are added.

The fixed control can run without a checkpoint or PyTorch:

```powershell
python audit_simple_reaching.py --modes fixed --episodes 100 --seed 910000 --output-dir runs/fixed_control_100
```

The first development check on seeds 810000 through 810019 reached 12/20
goals using only the four fixed commands, with mean final error 1.046 mm.
The previously uploaded guided evaluation reached 20/20 on those tasks.
This supports examining task simplicity while recognizing the actor's
additional improvement. The fixed command was set to the generator's nominal
command, not tuned separately to each target.

A second run on seeds 910000 through 910099 reached 64/100 goals, with mean
final error 0.891 mm and no initially satisfied goals. It used the same fixed
command and no learned model or Jacobian. Runtime versions were Python 3.12,
NumPy 2.3.5 and SciPy 1.17.0; mechanics source was commit e92b4b0. The guided
actor's reported 99% needs its own audit on the user's checkpoint; the uploaded
guided archive contains logs but no model weights.

High actor success accompanied by low goal-corruption success supports goal
dependence. High success even with a corrupted goal warrants broader task
design and comparison with the fixed control. Failure to hold the goal limits
the claim to first entry into the tolerance region. Use these results before
changing the loss or beginning a new training run.
