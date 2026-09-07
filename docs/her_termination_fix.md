# Goal-dependent termination in HER

The baseline (reported in `baseline_500k.json`) achieved 188 successes in 1,000
episodes at 1 mm. This is one checkpoint from one training seed. The confidence
interval describes sampled evaluation episodes, not variation between training
seeds. No claimed accuracy improvement from the correction has been measured yet.

The environment ends an episode on success. Standard SB3 2.9.0 HER recomputes
rewards when it substitutes a goal, but retains the original terminal label.
For example, a transition that reaches its new HER goal should have target
`reward` rather than `reward + gamma * next_Q`. The former integration used
the latter whenever the original goal had not been reached.

`GoalTerminationHerReplayBuffer` recomputes both rewards and goal-dependent
termination with the transition's saved position tolerance, before observation
or reward normalization. Timeout transitions continue to bootstrap unless the
new goal is reached. Real transitions and HER episode boundaries retain SB3's
handling. Its `_get_virtual_samples` override uses SB3's internal API; run the
replay tests when changing SB3 versions.

Reference: https://github.com/DLR-RM/stable-baselines3/blob/v2.9.0/stable_baselines3/her/her_replay_buffer.py

The tests cover relabelled success/failure, timeouts, historical tolerances,
normalization, untouched real samples, and actual DDPG updates/save/load. Optional
trajectory recording is tested against identical evaluation without recording.

## Validate on Windows

```powershell
git pull --ff-only
python -m pytest -q
python train_ddpg_her.py --total-timesteps 200 --learning-starts 150 --buffer-size 5000 --batch-size 64 --eval-freq 10000 --checkpoint-freq 10000 --seed 0 --output-dir runs/her_fix_smoke
python evaluate.py runs/her_fix_smoke/final_model.zip --episodes 2 --output-dir runs/her_fix_smoke/evaluation
```

Existing checkpoints remain evaluable. Loading a model trained with the old
buffer does not correct its learned weights or automatically replace its saved
replay class. Use a fresh training run for the comparison, with the same robot,
action scale, DDPG hyperparameters, curriculum and total step count:

```powershell
python train_ddpg_her.py --total-timesteps 500000 --learning-starts 10000 --buffer-size 500000 --batch-size 256 --eval-freq 25000 --periodic-eval-episodes 25 --checkpoint-freq 50000 --seed 0 --output-dir runs/her_fix_seed0
```

Use distinct output directories to retain the original run. Select the new
checkpoint using periodic evaluation, then compare it on the same 1,000 seeds:

```powershell
python evaluate.py runs/her_fix_seed0/best/best_model.zip --episodes 1000 --tolerance-m 0.001 --seed 300000 --output-dir runs/her_fix_seed0/comparison_1000
```

This compares implementations for one training seed; repeat across training
seeds before claiming a reliable algorithm-level improvement. Use a new held-out
seed range for a final report if these comparison seeds guide further tuning.

## Inspect the old policy near its goals

```powershell
python evaluate.py runs/full_seed0/best/best_model.zip --episodes 20 --tolerance-m 0.001 --seed 100000 --record-trajectories --output-dir runs/full_seed0/diagnostic_20
```

`trajectories.csv` records the tip before/after each step, target, error,
normalized actions, joint positions, and the actual joint changes after
constraints. It does not change the policy's actions or evaluation semantics.
Inspect errors over time, repeated motion past the goal, and nonzero commands
that produce negligible joint motion. These distinguish candidate precision
problems; a low terminal error alone does not identify a cause. Stop-on-success
evaluation cannot measure the ability to hold position after reaching a goal.

The prior maximum action still repeats ten times per policy step. This is a
maximum displacement scale, not a minimum resolution: continuous small commands
are possible. Do not infer a 1 mm precision limit from action bounds alone.
