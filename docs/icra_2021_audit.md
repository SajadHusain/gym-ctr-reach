# ICRA 2021 reproduction audit

Status: source audit completed on 2026-09-08; an exact ICRA 2021 training
implementation has **not** been delivered. The existing `paper-2024` profile
targets a different publication and must not be presented as this experiment.

The target is *Deep Reinforcement Learning for Concentric Tube Robot Control
with a Goal-Based Curriculum*, Iyengar and Stoyanov, ICRA 2021,
DOI [10.1109/ICRA48506.2021.9561620](https://doi.org/10.1109/ICRA48506.2021.9561620).
The attached published PDF was checked against its rendered Table I. An
[author manuscript](https://discovery.ucl.ac.uk/10132537/1/ICRA_2021_Submission.pdf)
is also available.

## Evidence recovered

The author's ROS policy repository contains an archived `cras_exp_6` run:

- [Run configuration](https://github.com/keshaviyengar/ctr_policy_ros/blob/2a220e37ab14b98b89bb7ce9fa234b357a94e952/example_model/cras_exp_6/config.yml).
- [Saved 500,000-step checkpoint](https://github.com/keshaviyengar/ctr_policy_ros/blob/2a220e37ab14b98b89bb7ce9fa234b357a94e952/example_model/cras_exp_6/learned_policy/500000_saved_model.pkl).
- [ICRA evaluation script](https://github.com/keshaviyengar/gym-ctm-ros/blob/6ae2d59719e8c7a25bf51ba91ea1d1fed7c06d57/ctm_envs/tests/evaluation.py), which explicitly names `cras_exp_6` among the six ICRA experiments.

Despite its `.pkl` extension, the checkpoint is a ZIP containing JSON metadata
and NumPy parameter arrays. The audit read the JSON and numeric arrays with
`allow_pickle=False`; it did not unpickle objects, evaluate YAML constructors,
load the policy into a runtime, or execute downloaded code. Checkpoint SHA-256:
`f4f19d922d9cce52876bbc3256c65b6d606e6c2731317f69d1554f87bc05be1b`.

This establishes a strong connection to the experiment, but does not prove that
the public checkpoint produced the exact Table II numbers.

## Settings verified in the archived run

| Setting | Archived value | Evidence |
| --- | --- | --- |
| Actor and critic hidden layers | 128, 128, 128 | Configuration, checkpoint metadata and weight shapes |
| Actor input width | 19 | First actor matrix is 19 x 128 |
| Critic action insertion | After first hidden layer | Second critic matrix is 134 x 128: 128 features plus 6 actions |
| Observation normalization | Enabled | Configuration, checkpoint flag, 19-element running statistics |
| Return normalization | Disabled | Configuration and checkpoint |
| Replay capacity | 10,000 | Configuration and checkpoint |
| Batch size | 256 | Configuration and checkpoint |
| Actor / critic learning rates | Both 0.0005 | Configuration and checkpoint |
| Discount / target update | Gamma 0.95 / tau 0.001 | Checkpoint |
| Rollout / optimizer cadence | 100 rollout steps / 50 training updates | Checkpoint |
| HER | Future strategy, 4 sampled goals | Configuration and checkpoint |
| Gaussian action noise | 0.00065 for each extension component; 0.025 for each rotation component | Configuration and checkpoint |
| Random exploration probability | 0.294 | Configuration and checkpoint |
| Representation | Relative (egocentric), trigonometric | Configuration |
| Training reset behavior | `resample_joints: false`; initial joints zero | Configuration; environment preserves joints between episodes |
| Training budget | 500,000 | Configuration and paper |
| Decay curriculum | 20 mm to 1 mm over 200,000 steps; then 1 mm | Configuration and paper |

The noise entries are algorithm action-noise settings, not sensor-noise values.
The archived checkpoint does not provide a reproducible training seed: its
`seed` entry is null. The paper reports 19 parallel workers. Worker count and
whether the budget is counted per worker must be preserved and documented;
19 SB3 vector environments are not equivalent to legacy MPI DDPG.

Later files in the author's `rl-baselines-zoo` repository disagree with this
archive about normalization and training resets. The archived configuration and
checkpoint are stronger evidence for this particular saved run than later YAML
defaults.

## Conflicts that prevent an exact paper-and-code claim

### Robot parameters

| Parameter (tube 1, 2, 3) | Published Table I | Released `icra_2021` environment |
| --- | --- | --- |
| Precurvature, 1/m | 21.3, 13.1, 0.1 | 15.82, 11.8, 20.04 |
| Inner diameter, mm | 0.7, 1.4, 2.0 | 1.0, 3.0, 4.4 |
| Outer diameter, mm | 1.1, 1.8, 2.2 | 2.4, 3.8, 5.4 |
| Straight length, mm | 431, 332, 10 | 200.1, 98.6, 39.7 (derived as total minus curved) |
| Curved length, mm | 103, 113, 5 | 14.9, 21.6, 8.8 |
| Total length, mm | 534, 445, 15 (literal straight plus curved) | 215, 120.2, 48.5 |
| Young's modulus, GPa | 6.4, 5.3, 4.7 | 50, 50, 50 |
| Shear modulus, GPa | 2.5, 2.2, 3.0 | 23, 23, 23 |

The released values come from the pinned
[environment registration](https://github.com/keshaviyengar/gym-ctm-ros/blob/6ae2d59719e8c7a25bf51ba91ea1d1fed7c06d57/ctm_envs/__init__.py).
Its [forward model](https://github.com/keshaviyengar/gym-ctm-ros/blob/6ae2d59719e8c7a25bf51ba91ea1d1fed7c06d57/ctm_envs/envs/exact_model.py)
uses the supplied tubes; it does not substitute the paper's dimensions.
The saved checkpoint's extension bounds are consistent with the released total
lengths, but checkpoint spaces alone do not certify every physical parameter.

Consequently, choosing the paper's robot and claiming to have rerun the archived
robot would be incorrect. The paper labels its lengths as **straight** lengths;
silently treating them as total lengths would be another undocumented change.

### State and software

Paper equation (5) defines 13 state values, including achieved-minus-desired
position. The released environment forms desired-minus-achieved error, and the
legacy [HER wrapper](https://github.com/keshaviyengar/stable-baselines/blob/2fea7ee093b46f4ed4e6d90729dd09bf9d115b38/stable_baselines/her/utils.py)
concatenates observation, achieved goal and desired goal. The archived network
therefore has 19 inputs, confirmed directly by its first weight matrix. A
13-input reconstruction is not the archived network.

The original stack is TensorFlow-era Stable Baselines with MPI, not SB3/PyTorch.
A Gymnasium-only port must preserve the original learner, normalization,
initialization, replay insertion, relabelling and bootstrap behavior. The
current branch deliberately changed several of these. Reusing that trainer
with different hyperparameters would still be a different implementation.

Gymnasium's `terminated`/`truncated` interface also needs an explicit adapter
at the old learner boundary; adopting modern timeout bootstrapping silently
would change the learning algorithm. Invalid historical observation bounds
need API-compatible replacements while preserving the actual observation
values. Those necessary compatibility edits must be documented and tested.

## Reproduction decision and validation gates

There are two defensible targets, and they must be named separately:

1. **Archived implementation reproduction:** preserve the public ICRA code and
   saved run settings, including its robot and 19-input network, with a minimal
   Gymnasium interface port. This is the recommended first target because the
   saved weights provide a concrete numerical reference for policy outputs.
2. **Published-method reconstruction:** use the paper's Table I robot and
   equation (5), filling unreported training details from the archive. This is
   an interpretation of the paper, not an unchanged rerun of released code.

The source priority needs to be resolved before an implementation is advertised
as exact. Author confirmation of the experiment geometry would resolve the most
important discrepancy. No claim of reproducing Table II is currently justified.

After that decision, the necessary checks are: compare fixed-configuration
forward kinematics and fixed-observation policy outputs with the reference;
verify replay, update and MPI accounting; then train and evaluate the six
representation/curriculum combinations. Final reaching evaluation uses 1,000
episodes with random initial joints and reachable goals, at 1 mm. The noisy
policy and path experiments are additional experiments, not covered by a single
reaching run. Record evaluation seeds and independent training runs separately.

For the egocentric-decay row, Table II reports mean error 1.29 mm and success
rate 0.93. The 3.38 mm / 0.89 row is egocentric-linear. The adjacent column is
labelled **variance**; it should not be rewritten as a standard deviation or
confidence interval. Matching architecture alone does not guarantee either row.

This audit changes documentation only. It does not supply a new runnable 2021
trainer, change existing checkpoint compatibility, or establish learning
performance. Existing tests of the modern implementation cannot validate the
historical reproduction.
