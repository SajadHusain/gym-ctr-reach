# Step 5: equilibrium DDPG/HER integration

The Step 4 Windows results support testing RL integration. They do not show that
the learned actor improves: the eight cases used local goals and synthetic
proposals. The supplied rows reconcile with the summary: 8/8 goals, 54 accepted
moves, 30 Jacobian fallbacks and zero reported positive error-function changes.
Refined final errors range from 0.516 to 0.975 mm. Control used 162 equilibrium,
116 sensitivity and 116 stability calls; including reset/witness/reference
work there were 274 equilibrium calls. No failed call lacked RHS accounting.
Only the summary and episode rows were supplied for this review, not the full
Step 4 transition records.

## What is implemented

This step adds an experimental `EquilibriumReachEnv`, guided rollout wrapper,
executed-action HER replay buffer, and training/evaluation entry points. The
registered original environment, baseline scripts and paper configuration are
unchanged. Old baseline checkpoints have different state/action semantics and
must not be used with these new scripts.

`EquilibriumReachEnv.step()` is the goal-independent mechanical transition. It
passes a bounded physical joint displacement to the branch tracker. Joint
projection, mechanics checks and backtracking may change the displacement. The
goal affects reward and task termination, but does not select the plant action.

`GuidedRolloutWrapper` instead offers the actor's proposal to the Step 4
controller. It tries the proposal, damped Jacobian control and gradient fallback
with measured decrease checks. It commits one accepted state or holds. A stall
outside tolerance does not end the episode: it is a holding transition. Success
terminates; the episode budget truncates and permits bootstrapping. There is no
physical dynamics timestep in this quasi-static model.

## Actions, branch state and HER

Actions are normalized vectors in `[-1, 1]`, ordered as all translations then all
rotations. Fixed physical scales are 1 mm and 0.05 rad per call by default. These
are caps on the total displacement, not the old environment's repeated substeps.
The replay label is

```text
executed_action = (q_after - q_before) / action_scales
```

`ExecutedActionHerReplayBuffer.add()` validates the full vector batch, records
the executed label, and retains the proposed action, physical joint increment
and branch coordinates in copied information. Missing execution data fails
before modifying replay. Real and hindsight samples both use the executed
label. A held transition has a zero executed action.

This makes the replay interpretation `(s, executed_action, next_s)` under the
goal-independent plant, even when collection used a goal-dependent safeguard.
Tests replay filtered displacements through the plain plant with a different
goal, for a straight analytical robot and a three-tube equilibrium. Floating
point replay coordinates are checked with explicit tolerances; this is a tested
numerical correspondence, not a proof for every configuration.

The observation dictionary contains:

- `observation`: normalized egocentric joint triples `(cos, sin, translation)`,
  base torsional strain times tube length, and tolerance divided by length;
- `achieved_goal` and `desired_goal`: Cartesian positions in metres.

The first tube uses absolute coordinates; subsequent tubes use differences from
the preceding tube. Base torsional strain is included because the same joints
can correspond to different equilibrium roots. The feature extractor computes
the scaled Cartesian error anew on every forward pass. HER therefore cannot
leave a stale goal error in the policy input.

The repository's `GoalTerminationHerReplayBuffer` recomputes both reward and
termination after relabelling using the recorded tolerance. An original timeout
does not suppress success at a hindsight goal. An original success is not
terminal for a new goal that was not reached. Goal-dependent collection flags
in `info` are diagnostic only; reward/termination do not use them. Terminal
observations come from SB3's terminal-observation handling, not the automatic
reset that follows an episode.

See the primary [SB3 collection source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/off_policy_algorithm.html)
and [HER implementation](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/her/her_replay_buffer.html)
for the interfaces extended here. The new `physics-train` dependency extra
targets SB3 2.9.x, because the custom buffer extends an internal sampling method.

## What physics guides in this step

Physics guides **behavior/action collection** and rejection decisions. Actor
and critic updates are standard DDPG with HER. The actor has the repository's
ReLU MLP structure (three layers of 256 by default), and the critic uses the
existing action injection after its first hidden layer. New branch features
change the input size. Default learning rate is 0.0005, gamma 0.95 and tau 0.001.
One gradient step follows each collection step after warmup; future HER uses
four sampled goals. These settings are recorded in `config.json`.

The Bellman target uses the **unfiltered target actor** acting through the
goal-independent plant. Thus the critic targets the ordinary actor's return;
the safeguard is an off-policy behavior controller. This does not evaluate the
full safeguarded controller's return inside the Bellman target. We do not claim
to differentiate through the safeguard or to have added a Jacobian actor loss.
This distinction must stay explicit in later policy-optimization experiments.

In particular, a weak actor may appear successful because the Jacobian fallback
does the work. A tiny descending policy action may pass the safeguard without
making useful progress on the episode timescale. Neither outcome demonstrates
sample efficiency or improved policy precision. The evaluator therefore reports
the deterministic actor, safeguarded actor and Jacobian-only controller
separately on identical seeded goals.

Step 4's runtime decrease property applies to accepted safeguarded transitions
for the original fixed goal. It does not apply to unfiltered actor evaluation,
HER's newly relabelled goal, changing waypoints, or gradient descent on network
weights. There are no global convergence or training-stability certificates.

## Task distribution and costs

The initial integration task uses seeded aligned starts and goals generated by
four local witness commands, as in Step 4. Held witness commands are retained.
The tolerance is fixed at 1 mm. Failed resets raise and are counted; they are
not replaced with easier configurations. This is not the paper's workspace
distribution or tolerance curriculum and cannot be compared directly with the
old 600,000-step result.

Costs include initialization, witness generation, rejected probes, sensitivity
and elastic assessments. SB3 may reset automatically after the final collected
episode, so its already-incurred reset cost is included. Completed transitions
are only one cost metric. Per-call elapsed time remains in lifetime cost totals
and does not contaminate deterministic step information.

Step 5 originally left branch torsion unbounded, producing Gymnasium Box-bound
warnings. Step 6 derives a finite unloaded-model envelope and uses a finite
tolerance bound without changing observation values. See
[`MECHANICS_STEP6.md`](MECHANICS_STEP6.md) for the derivation and compatibility
handling for saved Step 5 checkpoints.

## Windows commands

On branch `physics-equilibrium-step1`, in the activated virtual environment:

```powershell
git pull --ff-only
python -m pip install -e ".[physics-train,test]"
python -m pytest tests/test_mechanics_rl.py -q
python train_physics_ddpg_her.py --total-timesteps 64 --learning-starts 16 --episode-steps 16 --progress-every 8 --output-dir runs/step5_smoke
```

The new test file contains 11 tests, including actual updates to both networks,
model save/load, multi-tube action replay and hindsight success at an original
timeout. Training creates `config.json`, `summary.json`, `episodes.csv` and
`final_model.zip`. An interrupted/failed run saves an interrupted checkpoint and
reports the failure. Existing run directories are not overwritten.

The smoke run is successful as an integration test when it completes with
positive `gradient_updates`, changed actor and critic parameters, finite
parameters, and zero `maximum_executed_action_replay_error`. Its reaching rate
is not a useful policy-performance estimate after only 64 transitions.

For a paired diagnostic of all three behavior modes:

```powershell
python evaluate_physics_ddpg_her.py runs/step5_smoke/final_model.zip --episodes 4 --max-steps 30 --seed 800000 --output-dir runs/step5_modes
```

All modes use the same per-episode seeds, model parameters and goal generator.
Jacobian-only uses a zero proposal to invoke the fallback without an actor.
Reset/step failures remain in the denominator and are reported. These local
diagnostics can be expensive; the command prints after each episode.

For the later full mechanics-benchmark comparison, both guided and unguided
training must use this same plant, branch observations, goal distribution,
network, replay convention and matched budgets. `--guidance none` provides the
unguided training arm. A proposed actor physics loss, its ablations and a
workspace-wide curriculum remain separate research changes. Do not infer their
benefit from this integration check.

Checked local evidence is in
[`validation/step5_seed7005.json`](validation/step5_seed7005.json). GitHub CI runs
the integration tests and short three-tube training/evaluation on Linux and
Windows with Python 3.11.
