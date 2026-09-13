# Matched tolerance curriculum for the mechanics experiment

The current mechanics comparison uses a fixed 1-mm tolerance. This opt-in
experiment gives both DDPG+HER arms the same episode-level curriculum:

| Setting | Value |
|---|---:|
| Training budget | 50,000 interactions per arm |
| Initial tolerance | 5 mm |
| Final tolerance | 1 mm |
| Decay | Exponential over 25,000 collected transitions |
| Minimum initial goal distance | 6 mm (fixed for the whole run) |
| Episode horizon | 200 steps |
| Checkpoints | Every 5,000 interactions |
| Guided Jacobian weight | 0.1 throughout |

For an episode starting at `t` collected transitions:

```text
epsilon(t) = 0.005 * (0.001 / 0.005) ** min(t / 25000, 1)
```

The tolerance is sampled only when an episode starts, then remains fixed until
that episode ends. Episodes crossing the 25,000-transition boundary keep their
starting tolerance. HER uses the stored historical `position_tolerance` for
relabelled rewards and terminal flags. The observation space is bounded by 5 mm
for the complete run, so it does not change as the curriculum progresses.

The 6-mm goal floor prevents a relaxed 5-mm tolerance from making a sampled goal
successful at reset. The floor is held fixed as the tolerance tightens, so goal
sampling is not a second hidden curriculum. This is a local-goal development
experiment; it is not the preserved paper profile (which remains 20 mm to 1 mm
over 1.5 million interactions).

Evaluation never activates the curriculum. It reconstructs the checkpoint plant
with a fixed 1-mm tolerance and the saved 5-mm observation bound. The frozen
actor makes no Jacobian calls and receives no exploration noise. Use identical
evaluation seeds for both arms. Do not select a checkpoint using the final
1,000-episode test set.

## Windows PowerShell

From the repository root, with `.venv` activated:

```powershell
git status --short
git switch simple-jacobian-rl
git pull --ff-only
python -m pip install -e ".[train,test]"
python -m pytest tests/test_mechanics_curriculum.py tests/test_clean_training.py tests/test_reach_hold.py tests/test_simple_jacobian_rl.py tests/test_her_replay.py -q
```

If Git reports local edits, preserve them first and inspect the stash before
applying this update. The training scripts refuse to overwrite nonempty output
directories, and they start a new actor and HER buffer rather than resuming.

Check the resolved settings without training:

```powershell
python train_ddpg_her.py --profile mechanics --total-timesteps 50000 --tolerance-curriculum exponential --initial-tolerance-m 0.005 --tolerance-m 0.001 --curriculum-steps 25000 --minimum-goal-distance-m 0.006 --checkpoint-freq 5000 --seed 7101 --output-dir runs/curriculum_ddpg7101_50k --dry-run
python train_jacobian_ddpg_her.py --total-timesteps 50000 --tolerance-curriculum exponential --initial-tolerance-m 0.005 --tolerance-m 0.001 --curriculum-steps 25000 --minimum-goal-distance-m 0.006 --physics-weight 0.1 --physics-final-weight 0.1 --physics-anneal-steps 25000 --checkpoint-freq 5000 --seed 7101 --output-dir runs/curriculum_guided7101_50k --dry-run
```

Then run the two commands without `--dry-run`:

```powershell
python train_ddpg_her.py --profile mechanics --total-timesteps 50000 --tolerance-curriculum exponential --initial-tolerance-m 0.005 --tolerance-m 0.001 --curriculum-steps 25000 --minimum-goal-distance-m 0.006 --checkpoint-freq 5000 --seed 7101 --output-dir runs/curriculum_ddpg7101_50k

python train_jacobian_ddpg_her.py --total-timesteps 50000 --tolerance-curriculum exponential --initial-tolerance-m 0.005 --tolerance-m 0.001 --curriculum-steps 25000 --minimum-goal-distance-m 0.006 --physics-weight 0.1 --physics-final-weight 0.1 --physics-anneal-steps 25000 --checkpoint-freq 5000 --seed 7101 --output-dir runs/curriculum_guided7101_50k
```

The equal guided weights are deliberate: the Jacobian term stays active, while
the tolerance schedule is the only changing curriculum. The two runs still use
the same paper-like exploration mixture; approximately 29% of post-warmup
proposals are uniform random actions. Training time is likely roughly 2–3 hours
per arm on the previously measured CPU, but solver restarts can increase this.

## Fixed-tolerance evaluation

First use 100 development episodes:

```powershell
python evaluate_physics_ddpg_her.py runs/curriculum_ddpg7101_50k/final_model.zip --episodes 100 --max-steps 200 --tolerance-m 0.001 --seed 810000 --record-trajectories --output-dir runs/curriculum_ddpg7101_50k/development_100

python evaluate_physics_ddpg_her.py runs/curriculum_guided7101_50k/final_model.zip --episodes 100 --max-steps 200 --tolerance-m 0.001 --seed 810000 --record-trajectories --output-dir runs/curriculum_guided7101_50k/development_100
```

After the checkpoint-selection rule is fixed, reserve a different seed block
for the final 1,000-episode evaluation:

```powershell
python evaluate_physics_ddpg_her.py runs/curriculum_ddpg7101_50k/final_model.zip --episodes 1000 --max-steps 200 --tolerance-m 0.001 --seed 910000 --output-dir runs/curriculum_ddpg7101_50k/test_1000

python evaluate_physics_ddpg_her.py runs/curriculum_guided7101_50k/final_model.zip --episodes 1000 --max-steps 200 --tolerance-m 0.001 --seed 910000 --output-dir runs/curriculum_guided7101_50k/test_1000
```

Report deterministic success, mean/median/p95 error, steps, action-change RMS,
second-difference RMS, constraint alterations, solver failures and elapsed
physics cost. Changing tolerance during training does not itself demonstrate
sample-efficiency improvement. Independent training seeds are still required.

## Sampling a goal outside a larger initial tolerance

The witness generator repeats a signed, bounded joint command for a sampled
number of steps. `--goal-steps-max` defaults to 8. A distance floor of 9 mm can
exhaust its 32 proposals for a given start: for a straight tube, eight 1 mm
translations cannot displace the tip by 9 mm, regardless of rotation. This is
a reset sampling limitation, not an optimizer failure. Exhausting a random
budget alone does not prove that no qualifying goal exists.

The following explicit settings support experiments with an 8.5 mm initial
tolerance and a 9 mm goal floor:

```powershell
--goal-steps-max 20 --max-goal-sampling-attempts 32 --max-reset-sampling-attempts 8
```

Use these settings in both training arms. They are saved in `task_settings`,
and fixed-tolerance evaluation restores them automatically. The default reset
budget remains 1 for compatibility with existing configurations/checkpoints.
For a budget greater than 1, an exhausted endpoint distance budget rejects the
start and samples another start, up to the configured limit. A solver error
still propagates immediately. Explicitly supplied initial joints are never
replaced. The distance floor is never relaxed and retries create no replay
transitions. Every equilibrium solve, rejected endpoint and rejected start is
counted, including in evaluation. An impossible or unlucky task can still
exhaust the total budget; it raises a diagnostic error rather than hanging.

This defines a conditional start/goal distribution: starts from which the
configured witness sampler rarely finds a qualifying endpoint are less likely
to be accepted. A longer witness range also changes the goal distribution.
Compare newly trained arms with identical settings and evaluation seeds; do not
pool them with the earlier 2..8-command, single-start protocol. Report rejection
counts and reset computation costs alongside policy performance. Endpoint
reachability is not a certificate of intermediate equilibrium stability.

For a 150,000-transition run with decay_steps=50,000, the episode tolerance is
8.5 mm initially, approximately 2.915 mm at 25,000 transitions, and 1 mm for
episodes starting at or after 50,000 transitions. It stays fixed within an
episode. Use a fresh output directory; the trainer does not resume a checkpoint
merely because its output directory already exists.
