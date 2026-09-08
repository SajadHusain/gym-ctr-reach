# Author-archive reproduction with Gymnasium

This isolated implementation runs the author's archived TensorFlow DDPG/HER
learner. It does not use SB3. The user selected the released implementation as
the reference where it conflicts with the printed paper.

`bootstrap.py` retrieves 115 files at pinned commits, verifies their Git blob
hashes, keeps original copies, and makes explicit compatibility copies. The
manifest is `sources.json`. No runtime source is fetched from a moving branch.
The reference environment is `keshaviyengar/gym-ctm-ros` at
`6ae2d59719e8c7a25bf51ba91ea1d1fed7c06d57`; the learner is the author's
`stable-baselines` fork at `2fea7ee093b46f4ed4e6d90729dd09bf9d115b38`, as pinned
in that environment's requirements. Downloaded code retains its original
notices. ROS and rendering are not needed for headless experiments.

## Two discoveries relevant to retraining

The saved `cras_exp_6` checkpoint has a running observation count of
**9,500,000.01**. The original learner counts **500,000 steps per MPI worker**;
19 workers therefore collect **9.5 million real transitions**. The .01 is the
normalizer's initial epsilon. A single-worker 500,000-step SB3 run has a very
different data budget and optimizer setup.

The accompanying configuration says `relative_q: true`, but the checkpoint's
mean extension features are approximately [-0.1013, -0.0599, -0.0290] metres.
They strongly indicate absolute extension features. A fixed 10-episode
diagnostic at seed 100000 produced 10/10 successes with absolute features and
0/10 with relative features. This is evidence of an archive/configuration
mismatch, not evidence of 100% general performance. We do not relabel that
checkpoint as a proven egocentric-decay Table II result.

For the first reference-matching retraining run below, use **proprioceptive
(absolute) joints**. Egocentric remains an explicit option and the training
CLI default, reflecting the archived YAML and allowing all six paper
comparisons. Always specify representation in training and evaluation commands
so this discrepancy is visible in the saved experiment metadata.

## Initial reference result

The existing author checkpoint was evaluated for 1,000 episodes with absolute
joint features and a 1 mm tolerance, starting at seed 300000:

- Success: **917/1000 (91.7%)**.
- Mean final error: **1.383 mm**; median **0.672 mm**.
- Mean episode length: **17.676 steps**.

This is an evaluation of the author's saved weights, **not a retraining result**.
The machine-readable result and diagnostic runtime versions are in
[`reference_results.json`](reference_results.json). The original table's mean
error and success rate are close, but its variance column and uncertain
checkpoint identity prevent treating this as exact Table II replication.
Full TensorFlow/pinned-solver and MPI training checks run in the included CI
workflow; consult that run before beginning full training.

## Windows setup in the current workspace

Use Docker Desktop with Linux containers; see the
[official Windows installation instructions](https://docs.docker.com/desktop/setup/install/windows-install/).
The container isolates Python 3.7, TensorFlow 1.15.4, NumPy 1.18.5, SciPy 1.3.0
and Gymnasium 0.26.3. Your existing Python 3.11 environment can stay installed.
Gymnasium 0.26.3 supports the old NumPy/Python combination while providing the
five-return API. The archived requirements contain inconsistent TensorFlow
entries; 1.15.4 is the later explicit entry used here. MPICH/mpi4py 3.1.2 is a
packaged MPI runtime choice, not a recovered historical MPI version.

From PowerShell in `D:\CTR_gym_RL\CTR-RL\gym-ctr-reach`:

```powershell
git pull --ff-only
docker build -f icra2021/Dockerfile -t ctr-icra2021 .
docker run --rm ctr-icra2021 python -m icra2021.check --tensorflow
New-Item -ItemType Directory -Force runs | Out-Null
$ctrRuns = (Resolve-Path runs).Path
```

The build downloads the author sources and 500,000-step checkpoint. It does not
copy the existing virtual environment or large saved-policy folders into the
image. The source and checkpoint hashes are checked before use. The evaluator
reads JSON and numeric arrays; it does not unpickle checkpoint objects.

## Validate the author's saved policy first

```powershell
docker run --rm --mount "type=bind,source=$ctrRuns,target=/runs" ctr-icra2021 python -m icra2021.evaluate icra2021/_checkpoints/author_500000.zip --representation proprioceptive --episodes 1000 --seed 300000 --output-dir /runs/author_reference_1000
```

This evaluates existing author weights; it is not new training. It establishes
a reference under the selected implementation and explicit evaluation seeds.
The evaluator uses random starting joints and reachable goals, 150 steps per
episode, a 1 mm tolerance and deterministic actions. It reports mean, median,
95th percentile and variance of final Cartesian errors, plus success rate.
Output directories must be empty. Numerical library changes and unarchived
original seeds prevent a bitwise historical-results claim.

## Verify training before the full run

```powershell
docker run --rm --mount "type=bind,source=$ctrRuns,target=/runs" ctr-icra2021 python -m mpiexec -n 2 python -m icra2021.train --expected-workers 2 --representation proprioceptive --total-timesteps 600 --progress-every 200 --output-dir /runs/archive_smoke
```

Check `runs/archive_smoke/training_health.json`: optimizer steps must be
positive, parameters finite, and the two worker hashes identical. This checks
actual updates, MPI synchronization and checkpoint saving. It does not test
learning performance. The CI workflow runs this smoke test and reloads its
checkpoint in the evaluator.

## Full reference-matching run

```powershell
docker run --rm --mount "type=bind,source=$ctrRuns,target=/runs" ctr-icra2021 python -m mpiexec -n 19 python -m icra2021.train --representation proprioceptive --curriculum decay --seed 0 --output-dir /runs/archive_absolute_decay_seed0
docker run --rm --mount "type=bind,source=$ctrRuns,target=/runs" ctr-icra2021 python -m icra2021.evaluate /runs/archive_absolute_decay_seed0/final_model.zip --representation proprioceptive --episodes 1000 --seed 300000 --output-dir /runs/archive_absolute_decay_seed0/evaluation_1000
```

Defaults are 500,000 steps per worker, 20 mm to 1 mm tolerance over the first
200,000 curriculum steps, three 128-unit layers, 19 inputs, observation
normalization, a 10,000-entry replay buffer, batch size 256, learning rates
0.0005, gamma .95, tau .001, 100 rollout steps then 50 optimizer updates,
future HER with four synthetic goals, Gaussian action noise
[.00065, .00065, .00065, .025, .025, .025] in normalized action coordinates,
and random-action probability .294. There is one replay buffer per MPI worker.
The robot starts extended; ordinary training resets preserve joint state.

The run writes the environment and algorithm settings, dependency versions,
worker count, seeds, and both per-worker and global transition counts to
`run_config.json`. Checkpoints are inference checkpoints; resuming replay and
optimizer state is not implemented. Do not resume modern SB3 checkpoints.

For the six paper comparisons, run each combination of
`--representation proprioceptive|egocentric` and
`--curriculum constant|linear|decay` in its own output directory, retaining 19
workers and the same per-worker budget. Constant means 1 mm throughout.
Train multiple independent seeds when measuring training variability. The
`--noisy` option selects the author's encoder/tracker-noise environment for
the additional robust-policy experiment; use it in evaluation only when the
experiment calls for noisy observations.

## Preserved behavior and compatibility changes

The original DDPG losses, TensorFlow policy initialization, action injection
after the critic's first layer, MPI Adam updates and global running observation
statistics are retained. So are the original per-episode HER insertion,
future sampling, current-tolerance reward recomputation, unchanged embedded
goal-error fields on relabelling, and false synthetic terminal flags. These are
historical behaviors, not recommendations for a new RL implementation.

Compatibility changes are explicit:

- Import references use Gymnasium and a local API bridge. The Gym package is
  not installed or imported. Unused algorithm/Atari/video eager imports are removed.
- The public environment returns `(obs, info)` and five step values. The private
  learner boundary reconstructs the original `done = terminated or truncated`,
  including terminal treatment of episode time limits.
- Invalid observation-space bounds are replaced; observation values retain
  their original float64 precision. Rotation clipping, coupled-joint handling,
  the solver's segmentation rules and substeps are preserved.
- A logging alias provides `goal_tolerance`, which the pinned learner expects
  but the published environment calls `position_tolerance`.
- Explicit scalar extraction supports NumPy 2 for diagnostic evaluation. The
  pinned runtime compares the original and compatibility solver numerically.
- Action and joint spaces receive explicit seeds, since the original space
  seeds were not saved. Noise retains the process-wide NumPy stream. New seeds
  are recorded; they cannot be claimed as the original seeds.
- A collective replay-readiness check prevents one worker entering MPI Adam
  while another still lacks a completed episode. This necessary synchronization
  guard is the one additional training-control repair; loss equations and the
  original update cadence after all workers are ready are unchanged.

The curriculum callback uses the pinned learner's `locals['step']`, as in the
released Zoo callback. That counter is also reused inside optimizer logging;
it is not silently replaced with SB3's transition counter.

The publication/code discrepancies and unknown historical seeds mean this is
an audited archive-based reproduction attempt, not a guarantee of the printed
Table II numbers. The archived checkpoint is not sufficient evidence that
its configuration file exactly describes the corresponding training run.
