# Diagnose Jacobian guidance before changing its loss

This change keeps the existing displacement-matching loss and actor update rule.
It adds two independent measurements. No progress loss is implemented yet.

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

Each CSV record averages the gradient updates in that training block. Retain
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

## Papers related to a future progress objective

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
