> Historical implementation notes. For current training commands and defaults, use [clean_experiment.md](clean_experiment.md). The old root training entry points have been consolidated.

# Step 1: validated unloaded equilibrium backend

This branch adds an **opt-in** mechanics backend. It does not change
`CTR-Reach-v1`, the reproduced DDPG+HER trainer, or any saved policy.

## What is implemented

`ctr_reach_envs.mechanics` contains:

- `TubeParameters`: validated SI tube geometry and constitutive properties.
- `segment_tubes`: event-based segmentation at deployed tips and curved-section
  starts. It does not round boundaries to a fixed grid.
- `JointConstraints`: simultaneous Euclidean projection onto all extension
  ordering inequalities and optional rotation limits.
- `EquilibriumSolver`: an unloaded, geometrically exact orientation integration
  with nonlinear torsional equilibrium solved by shooting. The free-end
  torsional strain of every tube is checked explicitly.
- `EquilibriumReachEnv`: an opt-in Gymnasium adapter. Register it with
  `register_equilibrium_env()`; importing the package does not replace the
  baseline registration.

The model assumes Euler--Bernoulli bending, torsion, inextensible tubes,
perfect concentric contact, no friction, no gravity, no external wrench, and a
zero intrinsic torsion. These assumptions describe a quasi-static unloaded
reference model; they are not a dynamic or contact simulator.

## Numerical checks

The 19 mechanics tests cover:

- invalid geometry and constitutive parameters;
- simultaneous constraint projection and convex feasible interpolation;
- preservation of arbitrarily small segment intervals;
- straight and single-tube closed-form shapes;
- aligned multi-tube zero-strain cases;
- an independent two-tube torsional boundary-value calculation;
- orientation orthogonality, torque balance, and solver refinement;
- derivative smoothness across perturbations below 10 micrometres;
- explicit failure on solver-budget exhaustion;
- pure solving without mutating rendered state;
- baseline registration remaining unchanged.

Run the tests from the repository root:

```powershell
python -m pytest tests/test_mechanics.py -q
```

Run the seeded audit across all four configurations:

```powershell
python validate_mechanics.py --samples-per-system 16 `
  --seed 7001 --output-dir runs/step1_validation
```

The audit compares the standard solver (`rtol=1e-8`, `atol=1e-10`) with a
refined reference (`rtol=1e-11`, `atol=1e-13`, `max_step=0.002`). On the
development run, 64/64 cases passed a 20-micrometre refinement threshold; the
largest observed tip discrepancy was approximately 1.01e-12 m. This is a
numerical refinement result, **not a rigorous global error bound**.

## What is deliberately not claimed

The solver reports `elastic_stability_certified=False` and
`global_uniqueness_certified=False`. A small boundary residual proves only that
the numerical equations were solved to the configured tolerance. It does not
prove that the equilibrium is the unique solution or a strict local minimum of
the CTR's elastic energy. Snap-through and branch changes require a separate
elastic-stability analysis.

Likewise, the solver's refinement comparison does not prove agreement with a
physical robot. The next mechanics gate is to add an energy Hessian/second
variation or a validated stability metric, then expose equilibrium sensitivities
only on branches that pass that gate.

## Expected cost

On the development CPU, a standard solve took roughly 0.09--0.26 s depending
on the repository system and configuration. This is substantially slower than
the existing simplified forward model. The future policy method must count
equilibrium solves, shooting evaluations, rejected actions, and wall-clock time
in addition to environment transitions.
