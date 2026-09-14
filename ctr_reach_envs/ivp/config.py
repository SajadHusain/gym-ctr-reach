"""Save the complete experiment rather than infer its plant from a filename."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from ctr_reach_envs.paper_config import paper_configuration, env_kwargs_for_profile, PAPER_PROFILE
from .env import OriginalIVPEnv
from .observations import observation_settings

PROFILE = "original-ivp-comparison-v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_spec(path=None):
    if path is None:
        return paper_configuration(), None, "repository paper defaults with explicit CLI overrides"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("profile") == PROFILE:
        if fingerprint(data["environment"]) != data["environment_fingerprint"]:
            raise ValueError("Configuration environment fingerprint mismatch")
        return deepcopy(data["spec"]), deepcopy(data["environment"]), str(Path(path))
    if data.get("profile") == PAPER_PROFILE and set(paper_configuration()).issubset(data):
        return data, None, str(Path(path))
    raise ValueError("--config requires this profile's config.json or a complete paper-2024 run_config.json")


def resolve(spec, environment, *, segment_mode, seed, physics, provenance):
    spec = deepcopy(spec)
    spec["physics_observation"] = observation_settings(spec.get("physics_observation"))
    integer_fields = ("total_timesteps", "curriculum_steps", "max_steps_per_episode", "n_substeps",
                      "buffer_size", "batch_size", "n_sampled_goal")
    for key in integer_fields:
        if isinstance(spec[key], bool) or int(spec[key]) != spec[key] or spec[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if spec["buffer_size"] <= spec["max_steps_per_episode"] or spec["batch_size"] > spec["buffer_size"]:
        raise ValueError("Replay capacity must exceed episode length and contain a batch")
    if not 0 < spec["final_tolerance_m"] <= spec["initial_tolerance_m"] < 1:
        raise ValueError("Require 0 < final tolerance <= initial tolerance < 1 metre")
    if not 0 < spec["actor_lr"] == spec["critic_lr"] < 1 or not 0 < spec["gamma"] <= 1:
        raise ValueError("Invalid learning rate or discount")
    if not 0 < spec["legacy_defaults"]["tau"] <= 1 or min(spec["legacy_defaults"]["rollout_steps"], spec["legacy_defaults"]["gradient_steps"]) < 1:
        raise ValueError("Invalid update schedule")
    if spec["activation"] != "relu" or spec["critic_action_injection"] != "after_first_hidden_layer":
        raise ValueError("This profile requires the saved paper MLP structure")
    if len(spec["hidden_layers"]) < 2 or any(not isinstance(x, int) or x < 1 for x in spec["hidden_layers"]):
        raise ValueError("At least two positive hidden widths are required")
    if spec["normalize_observations"] or spec["normalize_returns"] or spec["joint_representation"] != "egocentric":
        raise ValueError("This profile uses unnormalized paper egocentric features")
    sigma = np.asarray(spec["normalized_action_noise_std"])
    if sigma.shape != (6,) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0) or not 0 <= spec["random_exploration"] <= 1:
        raise ValueError("Invalid exploration configuration")
    env = deepcopy(environment) if environment is not None else env_kwargs_for_profile(PAPER_PROFILE)
    if env.get("domain_rand", 0) != 0:
        raise ValueError("This matched profile uses fixed tube parameters")
    env.update(max_steps_per_episode=spec["max_steps_per_episode"], n_substeps=spec["n_substeps"],
        constrain_alpha=spec["constrain_alpha"], joint_representation=spec["joint_representation"],
        select_systems=[spec["system_idx"]], resample_joints=True,
        extension_action_limit=spec["extension_action_limit_m"], rotation_action_limit=spec["rotation_action_limit_deg"],
        evaluation=False, render_mode=None)
    env["goal_tolerance_parameters"] = dict(initial_tol=spec["initial_tolerance_m"],
        final_tol=spec["final_tolerance_m"], N_ts=spec["curriculum_steps"], function=spec["curriculum_function"], set_tol=0.)
    mode = segment_mode or env.get("model_options", {}).get("segment_mode", "legacy")
    if mode not in ("legacy", "continuous"):
        raise ValueError("Unknown segment mode")
    # Continuous is an explicit numerical variant, not claimed bitwise legacy.
    env["model_options"] = dict(segment_mode=mode, integration_options=(
        {} if mode == "legacy" else dict(method="DOP853", rtol=1e-8, atol=1e-10)))
    env = json.loads(json.dumps(env, default=lambda x: x.tolist()))
    return dict(profile=PROFILE, seed=seed, spec=spec, environment=env, physics=physics,
                environment_fingerprint=fingerprint(env), configuration_source=provenance,
                plant="original zero-initial-torsion IVP; no free-tip torque shooting",
                segment_mode=mode, action_semantics="normalized proposal; original constrained increments repeated n_substeps",
                physical_stability_guarantee=False, sample_efficiency_demonstrated=False)


def make_env(config, *, evaluation=False, compute_jacobian=False, tolerance=None):
    if config.get("profile") != PROFILE or fingerprint(config["environment"]) != config["environment_fingerprint"]:
        raise ValueError("Unrecognized or modified environment configuration")
    env = deepcopy(config["environment"])
    if evaluation:
        tol = config["spec"]["final_tolerance_m"] if tolerance is None else float(tolerance)
        if not np.isfinite(tol) or not 0 < tol <= env["goal_tolerance_parameters"]["initial_tol"]:
            raise ValueError("Evaluation tolerance must be positive and within saved observation bounds")
        env["evaluation"] = True
        env["goal_tolerance_parameters"].update(function="constant", set_tol=tol)
    return OriginalIVPEnv(**env, compute_jacobian=compute_jacobian,
                          physics_observation=config["spec"].get("physics_observation"))
