# Reproducing the documented 2024 experiment

Use `train_paper_ddpg_her.py` for the system-0, free-rotation, egocentric-decay
experiment in Table II of *Deep Reinforcement Learning for Concentric Tube Robot
Path Following* (2024). This is a modern reproduction of the documented settings,
not an exact rerun of the historical software stack. It does not guarantee the
paper's success rate.

The earlier 3.38 +/- 0.15 mm / 0.89 result is the egocentric **linear** row of
Table I, which summarizes previous work. The 3 x 256 network and 3-million-step
budget are specified for the later experiments. Combining these with the earlier
500,000-step linear schedule would mix two protocols. The profile below targets
the later experiment with an accompanying saved configuration.

## Settings and provenance

The source files are the original saved
[algorithm configuration](../ctr_reach_envs/saved_policies/rotation_experiments/free_rotation/tro_free_0/her/CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0/config.yml)
and [environment configuration](../ctr_reach_envs/saved_policies/rotation_experiments/free_rotation/tro_free_0/her/CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0/env_parameters.yml).
Immutable source URLs and all resolved values are recorded by
`ctr_reach_envs/paper_config.py` and written into each run's `run_config.json`.

| Setting | Value | Evidence |
| --- | --- | --- |
| Robot | System 0; supplied geometry and material parameters | Saved environment / repository registration |
| Joint features | Egocentric cosine, sine, extension | Paper equations (3), (11), (12) |
| Policy input | 9 joint features + achieved-minus-desired position + tolerance (13 values) | Paper equation (6) |
| Actor / critic | Three 256-neuron ReLU hidden layers | Paper section III-E / saved algorithm |
| Critic action input | After the first hidden layer | Legacy Stable Baselines MlpPolicy |
| Actor output | Six tanh values, converted to physical increments | Legacy policy / repository actions |
| Training budget | 3,000,000 transitions | Paper / saved algorithm |
| Curriculum | Exponential decay: 20 mm to 1 mm over 1,500,000 steps, then fixed | Paper / saved environment |
| Rotations | Free (no clipping to +/- pi) | Saved environment |
| Episode cap | 200 steps | Saved environment |
| Increment limits | 1 mm extension; 5 degrees rotation | Repository registration |
| Substeps | 10 applications of the physical increment per policy action | Repository registration / step implementation |
| Learning rates | Actor and critic both 0.0005 | Saved algorithm |
| Discount gamma | 0.95 | Saved algorithm |
| Batch / replay capacity | 256 / 500,000 | Saved algorithm |
| HER | Future goals; 4 sampled goals per real transition | Saved algorithm |
| Gaussian action-noise std | [0.0018, 0.0018, 0.0018, 0.025, 0.025, 0.025] | Saved algorithm |
| Uniform random-action probability | 0.294, throughout training | Saved algorithm |
| Observation / return normalization | Disabled | Saved algorithm |
| Target update tau | 0.001 | Legacy DDPG default; absent from saved YAML |
| Update cadence | 100 rollout steps, then 50 optimizer updates | Legacy DDPG defaults; absent from saved YAML |
| Final evaluation | 1,000 episodes, deterministic, 1 mm success tolerance | Paper |

Noise is applied in **normalized action coordinates**, before conversion to joint
units, in both the legacy DDPG implementation and SB3. Do not divide these noise
values by the extension/rotation limits. The earlier proposal to interpret these
values as physical-unit noise was incorrect. Exploration noise is distinct from
sensor noise; this profile disables sensor noise and domain randomization.

The saved environment YAML contains `set_tol: 0.001`. In the current environment
a positive `set_tol` overrides the curriculum, so the training profile uses zero
and evaluation uses 0.001. The observation-space tolerance bound remains 0.02,
allowing the same checkpoint to be evaluated at different runtime tolerances.

The critic implementation follows the documented legacy
[MlpPolicy structure and initialization](https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ddpg/policies.html).
The actor has a tanh output; the critic has a linear scalar output. Hidden weights
use Glorot uniform initialization, output weights use uniform [-0.003, 0.003],
and biases start at zero. There is no layer normalization.

## Run on Windows

From the repository root with your existing virtual environment active:

```powershell
git pull --ff-only
python -m pytest -q
python train_paper_ddpg_her.py --dry-run
python train_paper_ddpg_her.py --total-timesteps 1000 --seed 0 --output-dir runs/paper_smoke_seed0
```

The smoke run checks collection, HER learning, and saving. Its success rate is not
a performance test. After it finishes, start a **fresh** full run:

```powershell
python train_paper_ddpg_her.py --seed 0 --output-dir runs/paper_2024_seed0
```

This command defaults to 3,000,000 steps. Do not resume an old baseline checkpoint:
the state encoder and critic structure have changed. Existing output directories
must be empty to prevent overwriting previous experiments. The older
`train_ddpg_her.py` still runs the previous baseline.

The saved metadata contains the complete configuration, dependency versions,
seed, and declared reproduction differences. Periodic evaluation uses 25 episodes
at 1 mm; this count is a local monitoring choice. Checkpoints are saved every
50,000 steps. Ctrl+C saves an inference checkpoint named `interrupted_model.zip`;
this script does not restore replay/optimizer/curriculum state for resumption.

## Evaluate and follow a path

Use the explicit profile for the new checkpoints. A mismatched profile raises an
error even when the numerical observation spaces happen to match.

```powershell
python evaluate.py runs/paper_2024_seed0/final_model.zip --profile paper-2024 --episodes 1000 --tolerance-m 0.001 --seed 300000 --progress-every 50 --output-dir runs/paper_2024_seed0/final_1000
python follow_path.py runs/paper_2024_seed0/final_model.zip paths/example_path.csv --profile paper-2024 --output runs/paper_2024_seed0/path_result.csv
```

Path following retains joint state between waypoints and uses the paper's cap of
20 policy actions per waypoint. The example CSV is illustrative; success is not
guaranteed for arbitrary Cartesian paths. Final reaching evaluation permits 200
steps per episode. Use a separate output directory for each checkpoint/tolerance.
The final checkpoint gives a fixed-budget comparison; a checkpoint selected by
periodic validation can be evaluated separately, with its selection rule reported.

## Differences that prevent an exact replication claim

- The original implementation used TensorFlow-era Stable Baselines and MPI.
  This port uses SB3/PyTorch and one training environment. Creating 19 SB3 vector
  environments would not reproduce MPI gradient averaging or per-worker replay.
- The saved YAML does not pin the complete historical software stack or random
  seeds. Tau and update cadence are documented legacy defaults, not values
  explicitly reported in the paper's saved config.
- The modern HER buffer stores real transitions and relabels on sampling. Legacy
  HER stored synthetic transitions too, so a capacity of 500,000 has different
  effective history length. Future-goal sampling details also differ by version.
- Modern HER recomputes goal-dependent termination and uses each transition's
  historical tolerance. Goal error is reconstructed after relabelling. These
  correctness fixes are retained; possible historical HER inconsistencies are
  not reintroduced.
- Time limits bootstrap in the modern Gymnasium implementation. Training updates
  wait for at least one batch of completed-episode real transitions. Exploration
  uses the configured policy/noise/random mixture from the first step.
- The active solver is the corrected quasi-static Model/Segment/Tube path. Its
  numerical/runtime fixes and local random-number generators remain in place.

Matching the listed settings is an evidence-based starting point, not proof of
matching the paper's implementation or outcomes. Report multiple independent
training seeds after the integration run; 1,000 evaluation episodes from one
policy do not measure variation across training seeds.

## Integration validation

Validated on Linux with Python 3.12, Gymnasium 1.3.0, SB3 2.9.0 and
PyTorch 2.14.0 CPU: all 19 tests passed. A 1,000-step run performed optimizer
updates, periodic evaluation and checkpoint saving; the saved checkpoint loaded
in the standalone evaluator and waypoint runner. These checks establish runtime
integration, not trained-policy performance or Windows-specific validation.
