# Diagnose Jacobian guidance and compare progress loss

Original-IVP training now accepts `--physics-loss tracking` (the compatible
default) or `--physics-loss progress`. Both use the same constrained action map,
source-state Jacobian, HER goal reconstruction and actor update rule. The two
diagnostics below can be used with either objective.

## 1. Observe accepted training updates

Add `--physics-diagnostics` to **the original-IVP training command**, with
`--profile original` explicitly present. This flag is opt-in and currently exposed
by the original profile only. No change to weights, cap, reward, curriculum or
training budget is needed. The existing `updates.csv` JSON field gains:

| Metric | Meaning |
| --- | --- |
| `jacobian_loss_before` | Existing loss on the sampled real/HER batch before the actor step |
| `jacobian_loss_after` | Same loss, goals and sensitivities after the accepted step, including restoration if skipped |
| `jacobian_loss_change` | After minus before; negative is an auxiliary improvement |
| `jacobian_loss_decreased` | Fraction of actor updates with strictly lower auxiliary loss |
| `nominal_weighted_loss` | Clearer alias of `total_actor_loss`; not the exact objective of the projected optimizer |
| `physics_loss_kind` | `tracking` or `progress`; identifies what `jacobian_loss` and its before/after values mean |

Each CSV record averages numeric metrics over the gradient updates in that training block. Retain
`jacobian_valid_fraction`, `actor_step_skipped`, and the before/after auxiliary
gradient ratios when interpreting these numbers. An all-invalid batch has zero
auxiliary loss and provides no evidence of good physics. The extra evaluation
uses no replay sampling, backward pass, simulator call, or random draw. These
checks require deterministic networks without batch normalization or dropout.
The no-diagnostic path retains its previous behavior.

## 2. Diagnose an existing original-IVP checkpoint offline

This works on an existing checkpoint without retraining or an old replay buffer.
Keep its original `config.json` next to the ZIP. Use a **new output directory**:

```powershell
python diagnose_original_jacobian.py `
  runs/ivp_guided10_v4_original/final_model.zip `
  --states 16 `
  --seed 910000 `
  --rollout-steps 8 `
  --action-fractions 0.1 0.5 1.0 `
  --relative-step 0.0001 `
  --output-dir runs/ivp_guided10_v4_diagnostics
```

Replace the model path with the actual checkpoint path. The command validates
the saved original-IVP profile, the environment fingerprint and the checkpoint's
embedded configuration. A mechanics/equilibrium checkpoint is rejected rather
than evaluated with the wrong plant. It runs on CPU.

**Prediction accuracy:** A fixed actor supplies actions at independently seeded
source states along its rollouts. For each held-out state, the diagnostic tests
the actor proposal and a fixed random proposal at 10%, 50% and 100% amplitude,
plus a zero-action check. Each trial restores exactly the same source state and
uses the real environment's constrained transition. It compares observed tip
motion against `J @ projected_delta(actor_action)`, and independently compares
the differentiable joint projection with the actual executed joint change.

**Auxiliary usefulness:** Separate fit states select one normalized SGD step
that decreases the existing auxiliary loss on a **copy** of the actor. The step
norm starts at `relative_step * max(parameter_norm, 1)` and is backtracked at most
eight times using only the fit loss. Held-out states do not choose the step.
The command measures held-out auxiliary loss and actual one-step goal progress
before and after that change. The critic is not involved in this probe; it is
not an alternative training optimizer. A failed fit descent or zero valid
coverage is reported explicitly. An accepted fit step does not establish
generalization.

The checkpoint, live actor, target networks, optimizer and replay are never
updated. All diagnostic simulator calls are separate from training and counted
in the summary. This is not a full policy evaluation or proof of improved return.
Initially successful source states are counted and excluded from decision probes.
Terminal rollout states are replaced by their preceding decision state.

## Outputs and interpretation

| File | What to inspect |
| --- | --- |
| `prediction.csv` | Absolute/relative prediction error, motion cosine, predicted/actual goal progress, constraint activation, projection discrepancy and failures for each action amplitude |
| `auxiliary_probe.csv` | Paired held-out progress; positive `actual_progress_change_m` means the copied actor improved one-step progress |
| `states.json` | Exact joints, goals, observations, sensitivities and disjoint fit/probe task seeds |
| `summary.json` | Grouped errors, sample counts, failure counts, valid coverage, fit/held-out losses, solver costs and checkpoint identity |

Distances are metres. `projection_error_inf` is only a numerical consistency
check over joint coordinates (metres/radians), not a Cartesian distance.
Relative prediction error uses a 1-micrometre denominator floor. Motion cosine
and progress-sign agreement are unavailable for numerically negligible motion.
Invalid sensitivities have null predicted quantities but retain actual motion.
Solver failures are counted, never averaged as zero error. Partial/interrupted
runs write `complete: false`; failed numerical trials also make completion false.
Inspect `auxiliary_probe_informative` in addition to `complete`.

Interpretation should be conditional:

- Good small-action prediction but poor full-action prediction indicates a local
  approximation problem; a larger auxiliary gradient may amplify it.
- Good prediction but increasing same-batch auxiliary loss indicates that the
  combined update does not reliably optimize the mechanical objective.
- Lower fit/held-out auxiliary loss without better actual progress suggests the
  displacement-matching objective may be inappropriate.
- Better local progress without better evaluation returns suggests a gap between
  local reaching and the long-horizon task. Compare independent evaluation runs
  with matching goals, tolerance, budgets and multiple training seeds.

For a first pilot, 16 states per split bounds cost; it is not a statistical
performance claim. Repeat with larger independent banks before redesigning the
loss. Do not tune the probe step on held-out outcomes and then call the same
outcomes an independent test.

## Minimum-progress actor loss

Let `e = desired_goal - achieved_goal` for the sampled real/HER transition and
`dx = J(q) @ projected_delta(actor(observation))`, in metres. The optional loss is

```text
r = ||e||
d = min(gain * r, r, max_tip_step_m)
L_progress = mean_valid(relu((||e - dx|| - r + d) / scale_m)^2)
```

The defaults are `gain = 0.5`, `max_tip_step_m = 0.002`, and `scale_m = 0.002`.
Thus it requests at least half the remaining distance or 2 mm of predicted
progress, whichever is smaller. The additional `r` bound makes the target
well-defined if a previous tracking configuration used a gain above one.
Here `max_tip_step_m` caps the **required decrease**, not the permitted action
or tip displacement. A predicted 20 mm improvement can have zero loss even
when the required decrease is only 2 mm. Stalling, moving away, sideways motion,
and overshoot that violates the required decrease receive a penalty. Mild
overshoot that still meets the required decrease is allowed.

Physics supplies the detached Jacobian and the differentiable transcription of
the actual constrained joint update, including all substeps. The actor receives
the gradient through its action; no gradient is needed through the ODE solver.
The goal is read afresh from each sampled batch, so HER relabeling changes the
progress direction. Invalid Jacobians contribute zero auxiliary loss and are
excluded from its mean; their RL transitions are retained. Old configurations
and checkpoints without a loss-kind field still use tracking.

`rl_priority` still projects conflicting auxiliary gradients, caps their
parameter-gradient norm relative to the RL gradient, and checks the resulting
step against the same-batch fixed critic. `--physics-max-aux-ratio 0.1` is a
gradient-norm cap, not a bound on the scalar loss ratio. The nominal weighted
loss can therefore exceed 1.1 times the RL loss. Progress is a local model
prediction, not a guarantee of improved physical motion or long-horizon return.
Large-action Jacobian errors observed in diagnostics still matter.

### Probe the new loss on the existing checkpoint

The optional diagnostic `--physics-loss` overrides only the copied-actor probe;
it never modifies the checkpoint or its adjacent config. Without that argument,
the diagnostic selects the saved loss, falling back to tracking for old runs.
`summary.json` records both `saved_physics_loss_kind` and
`probe_physics_loss_kind`.

```powershell
python diagnose_original_jacobian.py `
  runs/ivp_guided10_v4_original/final_model.zip `
  --physics-loss progress `
  --states 16 `
  --seed 910000 `
  --rollout-steps 8 `
  --action-fractions 0.1 0.5 1.0 `
  --relative-step 0.0001 `
  --output-dir runs/ivp_guided10_progress_probe_v1
```

Using the same checkpoint, seeds, state count and rollout settings as the
tracking probe produces the same fit/held-out state banks. Compare paired
`actual_progress_change_m`, valid coverage and failures, not the raw magnitudes
of two differently defined losses. A zero progress loss/gradient can mean all
valid samples already meet the threshold; it does not imply broken backpropagation.
After the pilot, use larger fresh state banks rather than repeatedly tuning on
the same held-out tasks.

### Start a matched training experiment

This starts a **fresh model** and copies the experiment specification and plant
from the previous original-IVP `config.json`. It does not resume that checkpoint.
The total budget, seed and physics settings below are explicit overrides.
The saved specification retains the previous tolerance, episode length,
network, exploration and observation settings. Verify that this is the config
of the run you intend to compare; an equilibrium/mechanics config is rejected.

```powershell
python train_jacobian_ddpg_her.py `
  --profile original `
  --config runs/ivp_guided10_v4_original/config.json `
  --total-timesteps 600000 `
  --seed 10 `
  --physics-loss progress `
  --physics-weight 0.1 `
  --physics-final-weight 0.1 `
  --physics-anneal-steps 200000 `
  --physics-integration rl_priority `
  --physics-max-aux-ratio 0.1 `
  --physics-gain 0.5 `
  --physics-scale-m 0.002 `
  --physics-max-tip-step-m 0.002 `
  --physics-diagnostics `
  --output-dir runs/ivp_guided10_progress_v1
```

With equal initial/final weights there is no annealing. `--config` supplies the
saved experiment specification and environment; physics flags must be stated
explicitly as above. The new config has `physics.loss_kind: "progress"`, and
update records contain `physics_loss_kind: "progress"`. The loss kind is also
stored in the model ZIP and used when reconstructing the training loss.
Use a new output directory for every run. Compare progress against tracking and
ordinary DDPG on matched evaluation tasks; repeat training seeds before drawing
sample-efficiency conclusions.

## Papers related to the progress objective

These papers use Lyapunov decrease conditions or related controller losses.
They are precedents for the principle, not prior implementations of our exact
CTR Jacobian + DDPG/HER hinge penalty.

1. **Chang, Roohi and Gao, Neural Lyapunov Control (NeurIPS 2019).**
   [Paper](https://arxiv.org/abs/2005.00611).
   Penalizes violations of positivity and negative Lie derivative using the
   dynamics, and uses a falsifier for certification. This motivates penalizing
   failure to reduce an error function rather than matching a specific action.
2. **Mukherjee et al., Neural Lyapunov Differentiable Predictive Control
   (2022 preprint).** [Paper](https://arxiv.org/abs/2205.10728).
   Trains a policy and Lyapunov function through a dynamics model with decrease
   and constraint penalties. Our detached analytical Jacobian would supply only
   a local motion derivative, rather than their full predictive computation.
3. **Han et al., Actor-Critic Reinforcement Learning for Control with Stability
   Guarantee (2020).** [Paper](https://arxiv.org/abs/2004.14288).
   Incorporates a Lyapunov decrease condition through a learned critic in
   model-free actor-critic learning. It connects decrease conditions to RL, but
   does not supply our analytical robot Jacobian or retain our exact DDPG update.
4. **Long, Cortes and Atanasov, Certifying Stability of Reinforcement Learning
   Policies using Generalized Lyapunov Functions (2025 preprint).**
   [Paper](https://arxiv.org/abs/2505.10947).
   Uses generalized decrease across multiple steps and a related training loss.
   This is relevant if requiring progress on every individual step proves too
   restrictive for a long-horizon policy.

For CTR tip error, one candidate is `V(e) = 0.5 * ||e||^2`. The local model gives
`e_next = e - J(q) * delta_q`, hence predicted decrease is
`V(e_next) - V(e) = -e.T * J * delta_q + 0.5 * ||J * delta_q||^2`.
Physics enters through `J` and the constrained action-to-joint mapping; the
choice of `V` and a required decrease is a control design choice, not a physical
law. The Euclidean task error is not positive definite in the full redundant
robot configuration. A sampled penalty with a local model does not inherit the
papers' full-state stability guarantees or certify mechanical branch stability.
