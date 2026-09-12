# Recovering failed equilibrium shooting solves

The seed-7101 ordinary DDPG run stopped at 3,663 completed transitions with
`Continuation failed at curvature fraction 0.952637; residual 0.00545`.
No Jacobian actor loss or sensitivity calls were active. The failure was in
the forward equilibrium BVP, not in policy-gradient integration.

The reported joint vector, ordered as translations in metres then rotations
in radians, is:

```text
[-0.21087420446586644, -0.14110996395226102, -0.010941076481799957,
 -0.874262339815487, -3.5503030133542666, -0.15839325809198898]
```

The old zero-guess shooting solve stagnates at a nonzero boundary residual.
Its curvature continuation also stalls below full curvature. Increasing the
shooting budget alone is not a demonstrated repair for that path.

## Explicit numerical strategy

New training and study plans default to `--shooting-strategy hybr_restarts`.
The option is saved inside `solver_options` in each checkpoint configuration.
Evaluation reconstructs those options. Old configurations without this field
retain `legacy`; simply loading old weights does not silently change their
numerical strategy. `--shooting-strategy legacy` remains available for prior
experiments. Use new output directories when comparing the updated plant.

For `hybr_restarts`:

1. Try the original zero-guess shooting solve. An already accepted primary
   solution is unchanged.
2. If it fails the residual check, try a fixed sequence of alternative shooting
   initial guesses at **full prescribed curvature**.
3. Accept the first candidate meeting the original boundary tolerance, then
   validate it again through the original full-shape IVP integration.
4. If the bounded restart search finds none, try the existing curvature
   continuation using the remaining shared budget.
5. If no validated solution is obtained, raise the numerical error. Do not
   invent a stationary transition, discard the action, or resample the task.

An explicit `initial_torsion` supplied by another caller is not replaced by this
search. In particular, local derivative/reference checks can continue to specify
the equilibrium being tested.

The dimensionless shooting variable is `z_i = L * eta_i(0)`, where `L` is the
longest tube length and `eta_i` is torsional strain. The initial guesses are
signed coordinate unit vectors in deterministic tube-index order, negative
then positive. Each is adjusted as

\[
z^{(0)}=v-\frac{\sum_i GJ_i v_i}{\sum_i GJ_i}\mathbf{1},
\qquad v=\pm e_j,
\]

so the guess has zero total base torque. This condition is compatible with the
unloaded equations, but is not sufficient to satisfy all distal boundary
conditions. The unit amplitude and ordering define a reproducible numerical
search; they are not physical control limits or a proof of root uniqueness.
The guesses depend on fixed tube stiffnesses, not a goal, policy, past state
or previous equilibrium. No branch tracker or action safeguard is introduced.

For three tubes there are at most six restart guesses. Each root search uses
the existing `hybr` method with nominal `maxfev=45`; an explicit wrapper caps
**all restart residual evaluations combined** at `max_restart_evaluations=200`.
Every call also consumes the original global shooting budget (500 by default
in training) and RHS budget. Counters never reset between attempts. Passing a
SciPy success flag is insufficient: the actual residual must pass, as must the
full-shape reintegration. Existing `rtol`, `atol`, `boundary_tolerance` and
physical parameters are unchanged.

Successful solve diagnostics include `shooting_algorithm`, restart attempts,
accepted seed, evaluation count and message. Environment cost reports include
`root_recovery_solves`, while all restart work remains included in the normal
equilibrium/RHS counts and elapsed time. The analytical variational ODE used
for the policy Jacobian is unchanged.

## What recovery does not establish

Diagnostic solves found three distinct numerical equilibria at the reported
joint vector. Thus a small residual is not proof of a unique configuration,
elastic stability, physical branch continuity, or real-robot fidelity. The
deterministic root-selection rule makes the simulation convention explicit;
it does not remove the underlying mechanical ambiguity. In configurations
with multiple roots, changing the numerical strategy can change the selected
equilibrium, including where the old continuation previously converged.
Primary zero-guess successes remain unchanged.

The recovered root supports the existing analytical positional Jacobian on its
local equilibrium branch. That derivative does not predict a finite jump to a
different root. Neither root recovery nor RL-gradient conflict handling can
be reported as a closed-loop physical-stability guarantee.

## Validation and next run

The regression suite checks the reported state and twelve neighbors formed by
adding/subtracting one full command increment on each joint (1 mm translations,
0.05 rad rotations). It checks finite shape, free-end residual, orthogonal
rotations, shared budgets, repeated-solve determinism, tighter integration, and
the analytical tip Jacobian against a finite-difference test oracle on the
same equilibrium. It also checks legacy configuration restoration and that
invalid results cannot be accepted based only on SciPy's success flag.
Passing these checks fixes the reported case and tested neighborhood; it does
not guarantee convergence across all future exploration states.

```powershell
git pull --ff-only
python -m pytest tests/test_solver_recovery.py tests/test_reach_hold.py tests/test_simple_jacobian_rl.py -q
python train_physics_ddpg_her.py --task-profile generalized_reach --shooting-strategy hybr_restarts --total-timesteps 10000 --physics-weight 0 --seed 7101 --checkpoint-freq 1000 --output-dir runs/reach_ddpg7101_recovery
```

Use the same solver strategy for the guided arm:

```powershell
python train_physics_ddpg_her.py --task-profile generalized_reach --shooting-strategy hybr_restarts --total-timesteps 10000 --physics-weight 0.1 --physics-integration rl_priority --seed 7101 --checkpoint-freq 1000 --output-dir runs/reach_priority7101_recovery
```

Retain the interrupted run for the failure record. Its `interrupted_model.zip`
and completed checkpoints are not a finished 10,000-transition result. The
trainer does not currently save the HER replay buffer for exact continuation,
so these commands start new runs rather than claim to resume the old experiment.
