# Physics as information in original-IVP DDPG + HER

This experiment gives the actor and critic mechanical information and trains
them with the ordinary DDPG + HER objectives. It adds no Jacobian tracking loss,
teacher action, action correction, reward shaping, or optimizer projection when
run through `train_ddpg_her.py --profile original`.

Use the original-IVP profile for this feature. The paper reproduction and
equilibrium-mechanics profiles are unchanged. This describes implementation,
not evidence of better reaching performance.

## Feature modes

| `--physics-observation` | Extra inputs | Sensitivities at deployment |
|---|---:|---|
| `none` (default) | 0 | No |
| `jacobian` | 18 scaled Jacobian entries + validity flag | Yes |
| `jacobian_limits` | Above + 10 extension-limit margins | Yes |
| `zeros` | 29 zeros; same architecture as `jacobian_limits` | No |

All modes retain the original ten-element `observation` and both Cartesian
goal fields. A separate `physics` vector is appended by the feature extractor
after the original 13 policy inputs (nine joint features, recomputed goal
error, tolerance). Both actor and critic receive it. `jacobian_limits` therefore
has 42 total inputs; `jacobian` has 32. Their hidden widths stay unchanged.

Let J(q) be the 3-by-6 world-frame tip Jacobian in physical joint coordinates,
D the diagonal matrix of per-substep action scales (metres and radians), and
n the saved number of substeps. Define

    B = J(q) (n D) / c
    feature(B_ij) = B_ij / (1 + abs(B_ij))

The matrix is flattened by Cartesian row (x, y, z), with columns ordered as
three extensions followed by three rotations. The scale c defaults to 0.002 m
and is controlled independently of the actor-loss scale by
`--physics-observation-scale-m`. The dimensionless signed compression retains
sign and magnitude information while bounding the entries to [-1, 1]. It is
nonlinear: ratios between compressed entries differ from the original ratios.

`n D` describes NOMINAL unconstrained joint displacement per decision. It is
not the derivative of the repeated joint-limit projection, and these features
are not exact predictions of finite actions near constraints. No physical time
step is implied. This is information the network may learn to use or ignore.

The validity flag is 1 for a usable derivative and 0 otherwise. When a
sensitivity is unavailable or nonfinite, all 18 entries and the flag are zero;
the actual robot observation and RL transition remain. No old Jacobian is
substituted. The feature cache includes invalid results and is cleared on reset.

For `jacobian_limits`, the last ten entries are clipped to [0, 1]:

1. Three distances from the lower extension boxes, divided by their spans.
2. Three distances from the upper extension boxes, divided by their spans.
3. Two gaps beta_(i+1) - beta_i, divided by L_i - L_(i+1).
4. Two complementary gaps 1 minus the preceding normalized gaps.

These are extension margins, including the adjacent-tube ordering constraints.
They do not encode rotation-limit, collision, strain, or stability margins.
The margins remain available when the Jacobian is invalid.

## State alignment, HER, and saved models

Features are computed on reset and for the resulting state after every step.
The next call can reuse that state's cached sensitivity for source-transition
diagnostics and, if explicitly enabled, an auxiliary loss. Only an identical
joint configuration and system within the same reset may reuse the cache.
This avoids a second sensitivity solve at the same state. A failed forward
solve still follows the existing abort behavior of the training command.

The vector contains no desired-goal information. HER changes the desired goal,
reward, and termination while retaining the correct source and next-state
physics vectors. Goal error is recomputed inside the feature extractor on every
forward pass. Terminal observations carry terminal-state physics, including
when the vector environment automatically resets afterwards.

`spec.physics_observation` stores the mode, scale, and feature version in both
`config.json` and the checkpoint. Loading that config for another training run
preserves the settings unless explicitly overridden. Old configs without this
field use `none`. Evaluation reconstructs the saved features automatically and
reports `actor_uses_jacobian`, the mode, and sensitivity costs. Changing modes
requires a new training run; an old 13-input actor cannot consume 42 inputs.

The existing auxiliary-weight shutdown must not disable derivatives needed by
the observation. Information-enabled policies need the mechanical calculation
at deployment too. This adds computation even though the RL update is model-free.

## Matched experiment in PowerShell

Use fresh output directories. Start with the short integration check:

```powershell
python -m pytest tests/test_physics_observations.py tests/test_original_ivp.py -q
python train_ddpg_her.py --profile original --segment-mode continuous --physics-observation jacobian_limits --total-timesteps 256 --episode-steps 16 --batch-size 16 --buffer-size 1024 --hidden-width 32 --train-freq 16 --gradient-steps 2 --checkpoint-freq 0 --progress-every 64 --output-dir runs/ivp_info_smoke
python evaluate_original_ddpg_her.py runs/ivp_info_smoke/final_model.zip --episodes 5 --max-steps 16 --output-dir runs/ivp_info_smoke/evaluation
```

The software check should finish with finite parameters, optimizer updates,
zero physics-loss updates, and nonzero sensitivity calls in both training and
evaluation. It is not a convergence check; 256 transitions are insufficient.

For the main comparison, use the SAME original-IVP model, curriculum, training
budget, and seeds for each arm. Example for one seed:

```powershell
$ctrInfoArgs = @(
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
python train_ddpg_her.py @ctrInfoArgs --physics-observation none --output-dir runs/ivp_info_baseline7101
python train_ddpg_her.py @ctrInfoArgs --physics-observation jacobian_limits --output-dir runs/ivp_info7101
python train_ddpg_her.py @ctrInfoArgs --physics-observation zeros --output-dir runs/ivp_info_zeros7101
```

These use the baseline entry point intentionally: auxiliary weights are zero.
Do not substitute the guided entry point's nonzero defaults for an information-
only experiment. Explicit combinations with the existing auxiliary loss are
supported but should be separate ablations. Use `jacobian` as an optional arm
to test whether the extra extension margins matter.

Repeat with at least three seeds, for example 7100, 7101, 7102, changing output
names accordingly. Identical seeds do not make trajectories identical once
different policies start acting. The zero-feature arm matches the architecture
and initial parameters of the full-information arm; the `none` arm has fewer
input weights. Physics features do not change the saved plant fingerprint.

Evaluate each checkpoint on the same held-out tasks:

```powershell
python evaluate_original_ddpg_her.py runs/ivp_info_baseline7101/final_model.zip --episodes 1000 --seed 910000 --tolerance-m 0.001 --output-dir runs/ivp_info_baseline7101/evaluation
python evaluate_original_ddpg_her.py runs/ivp_info7101/final_model.zip --episodes 1000 --seed 910000 --tolerance-m 0.001 --output-dir runs/ivp_info7101/evaluation
python evaluate_original_ddpg_her.py runs/ivp_info_zeros7101/final_model.zip --episodes 1000 --seed 910000 --tolerance-m 0.001 --output-dir runs/ivp_info_zeros7101/evaluation
```

Compare success at 1 mm, steps to success, errors, numerical failures, training
and deployment wall time, and invalid-sensitivity frequency. Task fingerprints
must match across corresponding evaluation episodes. First-crossing success
does not by itself establish accuracy substantially below the 1-mm tolerance.
