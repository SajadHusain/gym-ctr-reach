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
