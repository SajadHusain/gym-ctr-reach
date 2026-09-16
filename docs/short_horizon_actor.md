# Short-horizon IVP actor returns

This optional mode is a **SHAC-inspired, off-policy DDPG/HER adaptation**, not a
reimplementation of SHAC. It replaces the tracking/progress auxiliary loss with
a short model return. The ordinary DDPG critic and RL actor updates remain.
Existing tracking/progress runs and checkpoints retain their original behavior.

For a sampled source state and its real or HER goal, the actor controls a fresh
nonlinear IVP rollout of up to `h` control steps. Each control step retains the
plant's original `n_substeps` constrained joint increments. The auxiliary is

```text
L_window = -mean_valid_windows[
    sum(k=0..K-1, gamma^k * r(s[k+1], goal))
    + (gamma^h * Q_target(s[h], actor_target(s[h]), goal) if nonterminal)
] / h
```

`K` is `h` unless success terminates the window earlier. Rewards remain exactly
the original sparse task: zero inside tolerance, minus one outside. Goals and
source-state tolerances remain fixed throughout each window. Source states that
already satisfy their HER goal are excluded. True success has no bootstrap;
the artificial window boundary bootstraps and does not reset the robot. Like
the existing time-limit-aware replay update, the tail represents continuation,
not a fictitious terminal at a window cut. No imagined transitions enter replay
or the critic loss. The critic still uses ordinary one-step real/HER targets.

The target actor and target critic weights are frozen for this derivative, but
their **input derivatives remain active**. In particular, SB3's `q1_forward`
cannot be used for the tail because its feature extraction suppresses state
gradients. The new terminal-value helper extracts features with gradients and
then calls the Q network directly.

Each new configuration gets a fresh nonlinear forward IVP solve and a fresh
analytical sensitivity solve. The custom backward multiplies the tip covector
by this Jacobian. Joint constraints and egocentric observations retain their
derivatives between steps. The bridge supports first-order gradients only.
It never chains finite steps through a Jacobian frozen at the window's start.
Model rollouts use a private plant, so collection state and exploration RNG are
not modified.

## What this can and cannot improve

The multi-step derivative accounts for how the actor's early actions change
later states and actions, and how the learned critic values the resulting tail.
It may provide a more useful update than demanding immediate geometric progress.
It still depends on critic accuracy. Sparse rewards and success indicators are
locally constant: this implementation does **not** differentiate discontinuous
success boundaries. For a nonterminal sparse-reward window, the pathwise signal
comes from the terminal value. The 10,000-step default delay allows initial
critic training; it does not certify that the critic is accurate.

Segment switches or unavailable derivatives invalidate the whole affected
window. Such windows do not contribute artificial zero-Jacobian gradients;
ordinary DDPG still updates from the complete minibatch. Conditioning on valid
windows can itself bias coverage, so inspect validity and rejection reasons.
Longer windows can have fewer valid samples and higher cost. Begin with `h=2`,
then compare `h=1` and `h=4` at matched seeds and budgets.

On scheduled updates, the existing `rl_priority` rule projects conflicting
auxiliary gradients and caps their weighted parameter-gradient norm relative
to the RL gradient. Its fixed-minibatch Q check remains. The cap does not bound
the Adam parameter displacement or certify return improvement. The nominal
weighted loss is only a reference scalar, not the exact projected update rule.
No old tracking/progress objective is also added in this mode.

## Train on the same plant and task

Use the prior original-IVP config to preserve its task, architecture, action
scales, tolerance schedule, and training schedule. `--config` loads those
settings, not trained weights and not the old physics flags. Start a new run:

```powershell
python train_jacobian_ddpg_her.py --profile original `
  --config runs/ivp_guided10_progress_v1/config.json `
  --segment-mode continuous --physics-observation none `
  --seed 10 --total-timesteps 200000 `
  --physics-loss short_horizon `
  --short-horizon-steps 2 `
  --short-horizon-batch-size 4 `
  --short-horizon-every 20 `
  --short-horizon-start-steps 10000 `
  --physics-weight 0.1 --physics-final-weight 0.1 `
  --physics-integration rl_priority --physics-max-aux-ratio 0.1 `
  --checkpoint-freq 10000 --progress-every 1000 `
  --output-dir runs/ivp_short_h2_seed10_v1
```

Do not add `--physics-diagnostics`: that flag recomputes the old local loss
after each actor update and is rejected for this mode. Window validity and
cost diagnostics are logged automatically. Physics observation features are
not supported in this first implementation, because differentiating a
Jacobian-valued observation would require additional derivative handling.
Continuous segment geometry is required. Neither restriction changes the
previously evaluated continuous, no-physics-observation baseline.

The defaults schedule at most four windows of two steps every 20 **gradient
updates**, after 10,000 **environment steps**. Thus each scheduled update costs
at most eight forward/sensitivity pairs, plus one forward solve to initialize
the private plant once per training process. Invalid/terminal windows can
reduce this. At 50 updates per 100 environment steps, this is at most 0.2 extra
solve pairs per environment step once enabled. Source Jacobians are no longer
computed during collection because this mode does not use them. If extra cost
is excessive, increase `--short-horizon-every`; doing so also reduces guidance
frequency. Model calls and time are additional costs, not free interactions.

`updates.csv` identifies `physics_loss_kind=short_horizon` and includes scheduled,
attempted, valid, invalid, already-terminal and terminal window counts;
`short_horizon_loss_active_mean` averages over scheduled updates with valid
windows. Generic integration statistics average all gradient updates, including
unscheduled ones, so a mean auxiliary norm ratio below the cap is expected.
`short_horizon_totals` contains cumulative model-call counts, timings, valid
coverage and invalid reasons. `summary.json` separates collection `costs` from
`short_horizon_costs`; add the two to report total computational work.
Sensitivity RHS counts include successful sensitivity solves only; attempted
call counts and elapsed time include failed attempts.

Saved checkpoints retain the mode, horizon, budget, counters and full config.
The private numerical backend is reconstructed lazily when training resumes.
Inference/evaluation remains the same actor-only policy, without IVP gradients
or Jacobian observations. Loading a model does not restore its replay buffer;
resuming with replay requires separately saving/loading that buffer as in SB3.

## Verify and evaluate

The regression suite checks the custom IVP backward against finite differences,
the full feedback-actor derivative across 1, 2 and 4 real IVP steps, plant action
and observation parity, retained terminal state gradients, success/invalid
window handling, model budgets, and public CLI train/save/load/resume behavior.
It also runs existing tracking/progress tests:

```powershell
python -m pytest tests/test_short_horizon.py tests/test_progress_loss.py tests/test_original_ivp.py -q
```

These checks verify the implementation at smooth states, not learning efficacy.
The older `diagnose_original_jacobian.py` remains a **local-loss** diagnostic;
it explicitly rejects probing this new objective rather than reporting a
tracking-loss result as a rollout diagnostic.

Evaluate checkpoints at their actual training budget (the earlier supplied
results identified 100,000 steps even though the pasted command said 200,000):

```powershell
python evaluate_original_ddpg_her.py `
  runs/ivp_short_h2_seed10_v1/checkpoints/step_000200000/model.zip `
  --episodes 500 --seed 920000 --tolerance-m 0.0015 --max-steps 200 `
  --output-dir runs/ivp_short_h2_seed10_v1/eval_200k_seed920000
```

Compare with baseline/progress **200k** checkpoints on identical evaluation
seeds. Report success, final error, steps, model calls and wall time; repeat with
independent training seeds before concluding the method improves learning.

Reference: Xu et al., *Accelerated Policy Learning with Parallel Differentiable
Simulation*, ICLR 2022, [paper](https://arxiv.org/abs/2204.07137). Unlike that
paper's on-policy SHAC and TD(lambda) value fitting, this adaptation uses replay
source states, HER and the existing DDPG critic with a bounded auxiliary update.
