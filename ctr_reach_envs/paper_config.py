"""Auditable modern reproduction of the 2024 paper's system-0 free-rotation run."""

from copy import deepcopy

from ctr_reach_envs.config import default_env_kwargs


PAPER_PROFILE = "paper-2024"
SOURCE_ROOT = (
    "https://github.com/SajadHusain/gym-ctr-reach/blob/"
    "17c0b9dd0e3faa7704b7ba704b08d2dbf9e7d210/ctr_reach_envs/saved_policies/"
    "rotation_experiments/free_rotation/tro_free_0/her/"
    "CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0/"
)

# YAML values transcribed as data, without executing the YAML's Python tags.
PAPER_CONFIG = {
    "profile": PAPER_PROFILE,
    "experiment": "2024 paper Table II, system 0, constraint-free egocentric decay",
    "total_timesteps": 3_000_000,
    "curriculum_steps": 1_500_000,
    "curriculum_function": "decay",
    "initial_tolerance_m": 0.02,
    "final_tolerance_m": 0.001,
    "max_steps_per_episode": 200,
    "system_idx": 0,
    "constrain_alpha": False,
    "joint_representation": "egocentric",
    "extension_action_limit_m": 0.001,
    "rotation_action_limit_deg": 5.0,
    "n_substeps": 10,
    "hidden_layers": [256, 256, 256],
    "activation": "relu",
    "critic_action_injection": "after_first_hidden_layer",
    "actor_lr": 0.0005,
    "critic_lr": 0.0005,
    "gamma": 0.95,
    "buffer_size": 500_000,
    "batch_size": 256,
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    # Both legacy SB2 and SB3 add noise BEFORE unscaling the tanh action.
    # These are dimensionless standard deviations. Do not divide by joint limits.
    "normalized_action_noise_std": [0.0018, 0.0018, 0.0018, 0.025, 0.025, 0.025],
    "random_exploration": 0.294,
    "normalize_observations": False,
    "normalize_returns": False,
    "final_evaluation_episodes": 1000,
    # Not stated in the saved YAML: explicitly selected legacy SB2 defaults.
    "legacy_defaults": {"tau": 0.001, "rollout_steps": 100, "gradient_steps": 50},
    "sources": {
        "algorithm": SOURCE_ROOT + "config.yml",
        "environment": SOURCE_ROOT + "env_parameters.yml",
        "legacy_algorithm": "https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ddpg/ddpg.html",
        "legacy_network": "https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ddpg/policies.html",
    },
    "reproduction_limits": [
        "Modern SB3/PyTorch port, not the original TensorFlow/MPI implementation.",
        "One training environment; original MPI workers and gradient averaging are not reproduced.",
        "Corrected modern HER recomputes historical reward and goal-dependent termination.",
        "SB3 replay capacity counts real transitions; legacy HER stored relabelled transitions too.",
        "Paper equation (6) is used as the 13-feature policy input; raw goals remain in replay only.",
        "Training starts after at least one batch of completed-episode real transitions is available.",
        "Time-limit truncations bootstrap, unlike some legacy Gym done handling.",
    ],
}


def paper_configuration():
    return deepcopy(PAPER_CONFIG)


def env_kwargs_for_profile(profile="current", *, evaluation=False):
    kwargs = default_env_kwargs(evaluation=evaluation)
    if profile == "current":
        return kwargs
    if profile != PAPER_PROFILE:
        raise ValueError(f"Unknown profile: {profile!r}")
    spec = PAPER_CONFIG
    kwargs.update(
        max_steps_per_episode=spec["max_steps_per_episode"],
        constrain_alpha=spec["constrain_alpha"],
        joint_representation=spec["joint_representation"],
        select_systems=[spec["system_idx"]],
        resample_joints=True,
        extension_action_limit=spec["extension_action_limit_m"],
        rotation_action_limit=spec["rotation_action_limit_deg"],
        n_substeps=spec["n_substeps"],
        domain_rand=0.0,
        noise_parameters={"rotation_std": 0.0, "extension_std": 0.0, "tracking_std": 0.0},
    )
    kwargs["goal_tolerance_parameters"] = {
        "initial_tol": spec["initial_tolerance_m"],
        "final_tol": spec["final_tolerance_m"],
        "N_ts": spec["curriculum_steps"],
        "function": "constant" if evaluation else spec["curriculum_function"],
        # Legacy set_tol is an evaluation value. In the modern environment any
        # positive set_tol overrides the schedule, so training must use zero.
        "set_tol": spec["final_tolerance_m"] if evaluation else 0.0,
    }
    return kwargs
