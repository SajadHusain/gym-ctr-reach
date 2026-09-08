# CTR reaching with DDPG and HER

For the **ICRA 2021 goal-based curriculum paper**, first read the
[source audit](docs/icra_2021_audit.md). Its archived experiment differs from
the 2024 profile below, and the paper and released code disagree about the
robot and policy inputs. An exact 2021 trainer has not yet been implemented.

For the documented 2024 paper configuration, use
[the paper reproduction instructions](docs/paper_reproduction.md) and
`train_paper_ddpg_her.py`. That profile uses 3 million training steps, an
egocentric decay curriculum over 1.5 million steps, and the saved system-0
free-rotation settings. It includes the legacy critic structure and a
HER-safe reconstruction of the paper's goal-error input. The instructions
separate verified settings from remaining software-stack differences.

The `modern-ddpg-her` branch turns the original quasi-static CTR code into a Gymnasium goal environment and adds:

- DDPG with a `MultiInputPolicy`
- hindsight experience replay (HER), using the `future` strategy and four relabelled goals
- goal-dependent terminal flags recomputed alongside HER rewards
- egocentric trigonometric joint observations
- a linear goal-tolerance curriculum from 20 mm to 1 mm over 200,000 transitions
- deterministic final evaluation over 1,000 independently seeded episodes
- Cartesian path following by passing consecutive waypoints to the same reaching policy

It deliberately uses the active simplified `Model`/`Segment`/`Tube` path. It does not import the broken experimental `CTR_Model.py`.

The previous standard-HER baseline and the terminal-label correction are described
in [docs/her_termination_fix.md](docs/her_termination_fix.md). That page includes
fresh-run comparison commands and optional per-step trajectory diagnostics for
existing checkpoints. The correction has not yet been evaluated for learning
performance; it should not be assumed to remove all precision errors.

## Important design boundary

HER and waypoint progression should not share an episode-level waypoint index. HER replaces a transition's desired goal, but it cannot reconstruct what the later path index should have been. This implementation therefore trains one goal-conditioned reaching policy and advances waypoints only during path execution.

The curriculum implemented here is a success-tolerance curriculum because that is the curriculum mechanism present in the supplied files. If “goal-based curriculum” instead means expanding the sampled goal radius, add that as a separate experiment; do not silently combine both curricula in the baseline.

## Environment corrections

The original environment could not be passed directly to current Stable-Baselines3. This bundle corrects the following blockers:

- uses the Gymnasium reset and five-return step API
- separates task success (`terminated`) from time/solver limits (`truncated`)
- returns `float32` observations matching the declared spaces
- uses local seeded random generators rather than global NumPy state
- makes `compute_reward()` vectorized for HER
- stores each transition's tolerance in `info`, so HER recomputes historical rewards consistently
- removes the goal-error vector from the `observation` key; otherwise HER would change `desired_goal` while leaving the old goal embedded in the state
- fixes length-weighted system sampling and makes `close()` safe
- raises on invalid tube parameters and handles ODE failure explicitly

The observation dictionary is:

- `observation`: nine egocentric joint features (`cos`, `sin`, extension for each tube), followed by the current tolerance and, for multi-system runs, the system index
- `achieved_goal`: current 3D tip position in metres
- `desired_goal`: target 3D tip position in metres

The internal joint vector remains `[beta_0, beta_1, beta_2, alpha_0, alpha_1, alpha_2]`, with metres and radians.

## Install

Use Python 3.10 or newer in a fresh virtual environment. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,test,render]"
python -m pytest
```

## Train

```powershell
python train_ddpg_her.py --total-timesteps 300000 --seed 0 --output-dir runs/ddpg_her_seed0
```

Periodic evaluation uses 25 episodes at the final 1 mm tolerance. Running 1,000 episodes at every checkpoint would dominate training time, so the requested 1,000-episode evaluation is a separate final command.

The supplied solver is slow enough that training will take hours. Start with a short integration run such as `--total-timesteps 12000`, verify that the replay buffer begins updating after 10,000 transitions, and only then launch the full run.

## Final 1,000-episode evaluation

```powershell
python evaluate.py runs/ddpg_her_seed0/final_model.zip --episodes 1000 --seed 100000 --output-dir runs/ddpg_her_seed0/final_evaluation
```

The evaluator writes a per-episode CSV and a JSON summary containing success rate, a Wilson 95% interval, final-position errors, step count, and solver failures. Evaluation is deterministic and always uses the final 1 mm tolerance.

## Follow a path

Waypoint files must have an `x,y,z` header and use metres:

```powershell
python follow_path.py runs/ddpg_her_seed0/final_model.zip paths/example_path.csv --output runs/ddpg_her_seed0/path_result.csv
```

The robot starts from zero joints, attempts each waypoint for at most 150 policy steps, and preserves its joint state when moving to the next waypoint. It stops at the first failed waypoint unless `--continue-on-failure` is passed. Rendering is optional via `--render`.

## What this baseline does not establish

This is DDPG because that was requested, but DDPG is sensitive to seeds and hyperparameters. Report results over multiple training seeds, not only 1,000 evaluation rollouts from one trained seed. TD3 should be the first algorithmic comparison because it directly addresses common DDPG failure modes.

The model is quasi-static and unloaded. It supports a position-reaching baseline, not claims about dynamic motion, contact, force control, or hardware fidelity. Noise remains disabled because the supplied noise values were stored but never physically applied. Add each physics or noise mechanism only with a corresponding validation test.
