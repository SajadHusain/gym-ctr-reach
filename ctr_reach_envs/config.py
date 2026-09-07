"""Robot and task parameters transcribed from the supplied ``__init__(2).py``."""

from copy import deepcopy

import numpy as np


CTR_SYSTEMS_PARAMETERS = {
    "ctr_0": {
        "tube_0": {
            "length": 431e-3,
            "length_curved": 103e-3,
            "diameter_inner": 0.7e-3,
            "diameter_outer": 1.10e-3,
            "stiffness": 10.25e10,
            "torsional_stiffness": 18.79e10,
            "x_curvature": 21.3,
            "y_curvature": 0.0,
        },
        "tube_1": {
            "length": 332e-3,
            "length_curved": 113e-3,
            "diameter_inner": 1.4e-3,
            "diameter_outer": 1.8e-3,
            "stiffness": 68.6e10,
            "torsional_stiffness": 11.53e10,
            "x_curvature": 13.1,
            "y_curvature": 0.0,
        },
        "tube_2": {
            "length": 174e-3,
            "length_curved": 134e-3,
            "diameter_inner": 2e-3,
            "diameter_outer": 2.4e-3,
            "stiffness": 16.96e10,
            "torsional_stiffness": 14.25e10,
            "x_curvature": 3.5,
            "y_curvature": 0.0,
        },
    },
    "ctr_1": {
        "tube_0": {
            "length": 370e-3,
            "length_curved": 45e-3,
            "diameter_inner": 0.3e-3,
            "diameter_outer": 0.4e-3,
            "stiffness": 50e10,
            "torsional_stiffness": 2.3e10,
            "x_curvature": 15.8,
            "y_curvature": 0.0,
        },
        "tube_1": {
            "length": 305e-3,
            "length_curved": 100e-3,
            "diameter_inner": 0.7e-3,
            "diameter_outer": 0.9e-3,
            "stiffness": 50e10,
            "torsional_stiffness": 2.3e10,
            "x_curvature": 9.27,
            "y_curvature": 0.0,
        },
        "tube_2": {
            "length": 170e-3,
            "length_curved": 100e-3,
            "diameter_inner": 1.2e-3,
            "diameter_outer": 1.5e-3,
            "stiffness": 50e10,
            "torsional_stiffness": 2.3e10,
            "x_curvature": 4.37,
            "y_curvature": 0.0,
        },
    },
    "ctr_2": {
        "tube_0": {
            "length": 309e-3,
            "length_curved": 145e-3,
            "diameter_inner": 0.7e-3,
            "diameter_outer": 1.1e-3,
            "stiffness": 75e9,
            "torsional_stiffness": 25e9,
            "x_curvature": 13.52,
            "y_curvature": 0.0,
        },
        "tube_1": {
            "length": 275e-3,
            "length_curved": 114e-3,
            "diameter_inner": 1.4e-3,
            "diameter_outer": 1.8e-3,
            "stiffness": 75e9,
            "torsional_stiffness": 25e9,
            "x_curvature": 11.68,
            "y_curvature": 0.0,
        },
        "tube_2": {
            "length": 173e-3,
            "length_curved": 173e-3,
            "diameter_inner": 1.83e-3,
            "diameter_outer": 2.39e-3,
            "stiffness": 75e9,
            "torsional_stiffness": 25e9,
            "x_curvature": 10.8,
            "y_curvature": 0.0,
        },
    },
    "ctr_3": {
        "tube_0": {
            "length": 150e-3,
            "length_curved": 100e-3,
            "diameter_inner": 1.0e-3,
            "diameter_outer": 2.4e-3,
            "stiffness": 5e10,
            "torsional_stiffness": 2.3e10,
            "x_curvature": 15.82,
            "y_curvature": 0.0,
        },
        "tube_1": {
            "length": 100e-3,
            "length_curved": 21.6e-3,
            "diameter_inner": 3.0e-3,
            "diameter_outer": 3.8e-3,
            "stiffness": 5e10,
            "torsional_stiffness": 2.3e10,
            "x_curvature": 11.8,
            "y_curvature": 0.0,
        },
        "tube_2": {
            "length": 70e-3,
            "length_curved": 8.8e-3,
            "diameter_inner": 4.4e-3,
            "diameter_outer": 5.4e-3,
            "stiffness": 5e10,
            "torsional_stiffness": 2.3e10,
            "x_curvature": 20.04,
            "y_curvature": 0.0,
        },
    },
}


def default_env_kwargs(*, evaluation: bool = False) -> dict:
    """Return an independent configuration dictionary in SI units."""

    final_tolerance = 1e-3
    tolerance = {
        "initial_tol": 20e-3,
        "final_tol": final_tolerance,
        "N_ts": 200_000,
        "function": "linear" if not evaluation else "constant",
        "set_tol": final_tolerance if evaluation else 0.0,
    }
    return {
        "ctr_systems_parameters": deepcopy(CTR_SYSTEMS_PARAMETERS),
        "extension_action_limit": 1e-3,
        "rotation_action_limit": 5.0,
        "max_steps_per_episode": 150,
        "n_substeps": 10,
        "goal_tolerance_parameters": tolerance,
        "noise_parameters": {
            "rotation_std": 0.0,
            "extension_std": 0.0,
            "tracking_std": 0.0,
        },
        "select_systems": [0],
        "constrain_alpha": False,
        "initial_joints": np.zeros(6, dtype=np.float64),
        "joint_representation": "egocentric",
        "resample_joints": True,
        "evaluation": evaluation,
        "length_based_sample": False,
        "domain_rand": 0.0,
    }

