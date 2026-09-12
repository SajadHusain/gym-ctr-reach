# CTR reaching: DDPG+HER and analytical Jacobian guidance

There are two supported training entry points:

| Command | Purpose |
|---|---|
| `python train_ddpg_her.py --profile paper` | Preserved documented paper reproduction; the default profile |
| `python train_ddpg_her.py --profile mechanics` | Ordinary DDPG+HER on the equilibrium plant, for a matched comparison |
| `python train_jacobian_ddpg_her.py` | Same equilibrium experiment with analytical task-space actor guidance |

The paper and equilibrium plants are different. Use the **mechanics baseline**
to measure the effect of Jacobian guidance. The paper profile preserves the
existing reproduction; its provenance and software differences are documented
in [paper_reproduction.md](docs/paper_reproduction.md).

The guided actor uses mechanical sensitivities from the analytical variational
ODE and implicit equilibrium derivative. It receives no inverse-Jacobian
controller action labels. Rewards and HER critic targets remain unchanged.
The default update prioritizes the RL surrogate and fades the auxiliary weight
to zero halfway through training. Frozen-actor evaluation uses no Jacobian.

A task-space tracking loss is still locally related to differential inverse
kinematics. It does not prove that RL adds value, improves smoothness, or provides
physical stability. The experiment includes mechanics-only and checked-RL actor
ablations, plus an evaluation-only constrained Jacobian controller.

## Setup and tests

From the repository root in your activated Python environment:

```powershell
python -m pip install -e ".[train,test]"
python -m pytest tests/test_clean_training.py tests/test_actor_gradients.py tests/test_solver_recovery.py -q
```

## Matched training

Use fresh directories. These commands share the same task and DDPG settings:

```powershell
python train_ddpg_her.py --profile mechanics --total-timesteps 10000 --seed 7101 --output-dir runs/clean_ddpg7101
python train_jacobian_ddpg_her.py --total-timesteps 10000 --seed 7101 --output-dir runs/clean_guided7101
```

The 10,000-step default is a development budget, not a promise of convergence.
The mechanics task uses fixed 1-mm first-hit reaching; holding is not required.
The paper profile retains its original tolerance curriculum and 3M-step default:

```powershell
python train_ddpg_her.py --profile paper --dry-run
python train_ddpg_her.py --profile paper --seed 0 --output-dir runs/paper_2024_seed0
```

## Evaluation

For each mechanics checkpoint, use the same evaluation seed block and horizon:

```powershell
python evaluate_physics_ddpg_her.py runs/clean_guided7101/final_model.zip --episodes 1000 --seed 910000 --output-dir runs/clean_guided7101/final_1000
```

Use `--modes jacobian` in a separate output directory for the classical control
comparison. `--controller-audit` adds diagnostic controller calls at actor states
without changing actor actions; its timing includes those extra calls.
For the paper checkpoint, use `evaluate.py --profile paper-2024` instead.

[The experiment guide](docs/clean_experiment.md) contains the exact objectives,
gradient derivation, assumptions, loss parameters, ablation commands, multi-seed
study commands, smoothness metrics, cost accounting, and interpretation rules.

## Repository organization

- `ctr_reach_envs/training/`: shared trainer and callback implementation.
- `ctr_reach_envs/mechanics/`: equilibrium solver, analytical sensitivities, actor losses and joint constraints.
- `evaluate_physics_ddpg_her.py`: frozen actor and classical controller evaluation.
- `run_physics_study.py`: frozen multi-seed plans and paired reports.
- `evaluate.py`, `follow_path.py`: preserved paper/original-plant evaluation tools.
- `tools/legacy/`: the earlier generic trainer, retained for provenance.
- `docs/`: current guide plus clearly marked historical implementation notes.
- `icra2021/`, saved policies and original utilities: author archive, retained.

The former root scripts `train_paper_ddpg_her.py` and `train_physics_ddpg_her.py`
have moved into the training package. Use the two entry points above for new runs.
The old mechanics invocation remains available for historical configurations as
`python -m ctr_reach_envs.training.mechanics`; it is not the new matched protocol.
Existing policy-class import paths remain available for checkpoint loading.
Do not resume a frozen study across this cleanup or pool its results with the
new protocol: the mechanics comparison now adopts the documented paper's
100-rollout/50-update cadence and uses a decaying guidance weight.

The simulator is unloaded and quasi-static. Numerical equilibrium recovery does
not establish physical stability or uniqueness, and none of these scripts
certifies global convergence, hardware fidelity, or publication novelty.
