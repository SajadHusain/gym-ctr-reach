# Step 2: local equilibrium sensitivity and elastic second variation

This stage adds analysis to `unloaded_equilibrium_bvp_v1`. It does not change
the DDPG baseline, rewards, training configuration, policy, or environment
registration. It does not yet implement physics-guided policy optimization.

## Scope and decisions

The analyzed model has inextensible, unshearable, concentric rods, circular
isotropic bending sections, independent positive EI and GJ, piecewise intrinsic
curvature, zero intrinsic torsion, and a straight proximal guide. Tube centers
share one centerline. There are no external loads, contact, friction, inertia,
or actuator dynamics. The Bishop-frame backbone integration admits finite
rotations. These assumptions are narrower than a general loaded Cosserat model.

An equilibrium residual tests stationarity, not elastic stability. Step 2
therefore computes two independent quantities for the **same supplied root**:

1. A positional Jacobian from analytical variational ODEs and implicit
   differentiation of the shooting boundary equations.
2. A mesh-refined finite-element second-variation diagnostic at fixed clamp
   rotations and insertions.

Neither operation silently changes the equilibrium, projects the joints, or
selects a more convenient branch. A geometry fingerprint prevents mixing
equilibria from different tube systems. Analysis reintegrates the supplied base
twist, checks the distal boundary residual and tip agreement, and preserves the
input result. An explicit base-twist warm start is still a numerical branch
tracking device, not a global uniqueness guarantee.

## Jacobian calculation

Let `q = [beta_1, ..., beta_n, alpha_1, ..., alpha_n]`, with translations in
metres and rotations in radians. Write `z = L * eta_base`, where `L` is the
longest total tube length, and let `R(z,q)` be the distal torsional strains
multiplied by L. On a regular local equilibrium branch:

\[
R(z(q),q)=0,\quad z_q=-R_z^{-1}R_q,\qquad
J=P_q-P_zR_z^{-1}R_q.
\]

The implementation uses a linear solve, not an explicit inverse. `P` is tip
position. Partial derivatives are propagated together with the mechanical ODE;
no finite-difference approximation or additional equilibrium root solve is
used to compute J. It also returns the derivative of base torsional strain.

Tube-tip and curved-section events move with beta. Each interval is integrated
on `t in [0,1]` with `dy/dt = (b-a) F(y)`. For any normalized parameter u:

\[
\frac{dS}{dt}=(b-a)F_yS+F(y)\frac{d(b-a)}{du}.
\]

Differentiating the initial condition `theta(0)=alpha-beta*eta_base` includes
the straight guide. Within a fixed event ordering, this accounts for moving
interfaces without quantizing their locations. Translation parameters are
normalized by L and rotations by one radian during integration; the returned
J is converted back to physical joint units.

The routine rejects coincident or near-coincident events with different event
gradients. At those locations a full joint Jacobian can be ambiguous; a future
directional derivative implementation is required. It also rejects an
ill-conditioned or nearly singular shooting derivative and failed integrations.
The shooting condition number **is not an elastic stability test**.

J is `dp/dq` before projection. For a recorded transition use the actual applied
`q_after-q_before`, including constraints. To differentiate a projected policy
action, the projection's own derivative must be included separately. J is local;
`J*dq` is not an exact finite displacement or a global error bound.

## Elastic diagnostic and derivation

At an arc-length location define the active-tube stiffness sum B and rotated
intrinsic curvatures c_i:

\[
B=\sum_i EI_i,\quad
c_i=R_2(\theta_i)\bar\kappa_i,\quad
\kappa=B^{-1}\sum_i EI_i c_i.
\]

Eliminating the common unloaded bending curvature gives bending energy density

\[
W_b=\tfrac12\sum_i EI_i\|\bar\kappa_i\|^2
-\frac{1}{2B}\left\|\sum_i EI_i c_i\right\|^2.
\]

Its angular Hessian is a graph Laplacian with pair weights
`w_ij = EI_i EI_j (c_i dot c_j) / B`: off-diagonal entries are `-w_ij`,
and diagonal entries sum the other weights. Anti-aligned intrinsic curvatures
can make this Hessian indefinite.

For angular perturbations v_i with clamped proximal rotations, the reduced
second variation is

\[
Q[v]=\int\left[\sum_{i\;active}GJ_i(v_i')^2+v^TH_bv\right]ds
+\sum_{\beta_i<0}\frac{GJ_i}{-\beta_i}v_i(0)^2.
\]

When beta_i=0, impose v_i(0)=0 instead. Each perturbation ends at its own tube
tip with a natural free-end condition. The guide term is exact static
condensation of straight-guide torsion. Suppressed intrinsic bending energy in
the guide is independent of tube angle and contributes no angular Hessian.

Continuous piecewise-linear trial functions are assembled on meshes aligned
with all mechanical events, with three-point Gauss integration of H_b evaluated
from dense ODE output. The smallest generalized eigenpair uses the positive
mass form `integral sum_i GJ_i/L^2 v_i^2 ds` over deployed portions. This makes
the reported eigenvalues dimensionless. The guide contributes stiffness but
no mass; eigenvalue magnitude is a diagnostic scale, not a vibration frequency.

Default meshes have 8, 16, and 32 elements per mechanical segment. Classification
requires the last change to be <= `0.01 * max(1, abs(lambda_min))`; the sign
margin is `max(0.001, 2 * last_change)`:

| Status | Meaning |
|---|---|
| `negative_second_variation` | A converged numerical negative mode; evidence of an elastically unstable equilibrium. |
| `positive_second_variation_on_tested_meshes` | Positive on the tested trial spaces with refinement agreement. Not a continuum certificate. |
| `inconclusive` | Near-zero sign margin or insufficient mesh convergence. |

Positive Rayleigh-Ritz eigenvalues can miss a continuum negative mode. Mesh
agreement is not a rigorous eigenvalue bound. For this reason
`elastic_stability_certified`, `controller_stability_certified`, and
`global_uniqueness_certified` remain false. A future rigorous certificate would
need verified continuum bounds or a validated conjugate-point calculation.

## Independent validation

The tests include a two-tube constant-curvature case whose relative angle
obeys `delta'' = C sin(delta)` with

\[
C=\frac{EI_1EI_2}{EI_1+EI_2}\kappa_1\kappa_2
\left(\frac1{GJ_1}+\frac1{GJ_2}\right).
\]

At the anti-aligned equilibrium with no guide, the lowest relative-mode
eigenvalue is `L^2 * [(pi/(2*l))^2-C]`, where l is deployed length.
Tests exercise below, at, and above the analytical threshold. With a guide of
length g, the threshold wave number solves `cos(k*l)-g*k*sin(k*l)=0`; separate
tests check the shifted threshold. Other checks cover aligned and straight
rods, actual energy decrease along a negative mode, the general bending-energy
Hessian, guide-sensitive tip derivatives, finite rotations, and common-rotation
covariance. Complete perturbed BVP solutions independently check tip and base
twist derivatives; local Taylor errors decrease quadratically in a regular case.

The seeded audit visits all four repository tube systems. It records every
sample, including failed solves, unstable equilibria, and unsupported
sensitivities. J is compared to full tighter-tolerance equilibrium solves at
two finite-difference step sizes with explicit local branch predictors. It also
records nonlinear prediction errors for three decreasing joint increments.
Sampling uses a separate random generator from probing, so probe changes do
not alter the sampled configurations.

Both the normalized Frobenius error and the largest normalized column error
must be at most 0.001. Column denominators have a 1e-6 m floor to avoid dividing
by essentially zero sensitivities; the Frobenius denominator has a 1e-8 m floor.
The two finite-difference estimates must also agree to normalized Frobenius
error 0.001. These are numerical acceptance tolerances, not physical accuracy
claims.

`PASS` in this audit means numerical derivative agreement and a completed
stability diagnostic. **An unstable equilibrium can pass this numerical audit.**
Read `stability_status_counts` and the individual cases before interpreting it
as evidence that a state is appropriate for control.

The initial local run passed all 34 combined Step 1/2 tests. At seed 7002 with
four samples per system, all 16 cases passed the numerical audit. Five had
negative second variation; eleven were positive on the tested meshes. The
largest normalized Frobenius Jacobian error was 1.30e-9. This small seeded set
does not estimate the fraction of unstable states over the entire workspace.
Machine and per-case results are in `docs/validation/step2_seed7002.json`.

## Windows commands

Remain on `physics-equilibrium-step1`; this stage is committed to that branch.
The changes do not touch `paper_config.py`, `train_ddpg_her.py`, or
`train_paper_ddpg_her.py`, so there is no need to discard your local edits.

```powershell
git pull --ff-only
python -m pytest tests/test_mechanics.py tests/test_mechanics_step2.py -q
python validate_mechanics_step2.py --samples-per-system 4 --seed 7002 --output-dir runs/step2_validation
```

Outputs are `summary.json`, `cases.csv`, and `details.json`. The details include
J, base-twist root identifiers, eigenvalues across meshes, numerical solver
settings, finite-difference steps, per-analysis timings, and ODE/shooting counts.
Partial outputs are saved after every case. No additional dependencies beyond
the existing numerical stack and `[test]` extra are needed.

The full audit is deliberately more expensive than computing J: each successful
three-tube sample uses one base equilibrium, two analysis integrations, 24
perturbed reference equilibria, and three local prediction equilibria.
Those reference solves belong to validation, not the intended future training
loop. Analysis timings and solve counts must still be included in any eventual
sample-efficiency comparison.

## Gate before policy optimization

Do not retrain yet. Review the numerical audit, identify any negative or
inconclusive states, and define explicit equilibrium-branch continuation and
recovery behavior. Simply filtering successful roots would change the task
distribution and could hide the problem.

The following stage can use regular-branch J for local action guidance and use
the elastic diagnostic to screen candidate equilibria. It must separately
validate finite-action prediction error, count all model queries, and compare
against DDPG under matched tasks. Closed-loop Lyapunov decrease is a separate
condition on the selected action and actual next state. Neither elastic
positivity nor an accurate Jacobian supplies that condition by itself.

This is model-analysis infrastructure, not a claim of a new stability theorem,
sample-efficiency improvement, or publishable novelty.

Background references: Dupont et al., *Design and Control of Concentric-Tube
Robots* (IEEE TRO, 2010), https://pmc.ncbi.nlm.nih.gov/articles/PMC3022350/;
Gilbert, Hendrick and Webster, *Elastic Stability of Concentric Tube Robots:
A Stability Measure and Design Test* (IEEE TRO, 2016),
https://pmc.ncbi.nlm.nih.gov/articles/PMC4814113/.
The finite-element diagnostic here is not presented as an implementation of
the latter paper's full stability test.
