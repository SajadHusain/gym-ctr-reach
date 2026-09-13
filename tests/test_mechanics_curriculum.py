import io
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from ctr_reach_envs.mechanics.curriculum import ToleranceCurriculum
from ctr_reach_envs.mechanics.geometry import TubeParameters
from ctr_reach_envs.mechanics.simple_rl_env import JointConstrainedReachEnv, make_reach_env
from ctr_reach_envs.training.cli import mechanics_defaults
from ctr_reach_envs.training.mechanics import arguments


def env(**kwargs):
    return JointConstrainedReachEnv(
        tubes=[TubeParameters(.2, 0., .001, .002, 50e9, 23e9, 0.)],
        task_profile="generalized_reach", max_episode_steps=4, **kwargs)


def test_exponential_schedule_and_cli_settings_are_explicit():
    schedule = ToleranceCurriculum()
    assert schedule.value(0) == .005
    assert schedule.value(12500) == pytest.approx(np.sqrt(.005*.001))
    assert schedule.value(25000) == schedule.value(500000) == .001
    parsed = arguments(
        ["--total-timesteps", "50000", "--tolerance-curriculum", "exponential",
         "--initial-tolerance-m", ".005", "--tolerance-m", ".001",
         "--curriculum-steps", "25000", "--minimum-goal-distance-m", ".006"],
        mechanics_defaults(True), baseline=True)
    assert parsed.curriculum == asdict(schedule)
    assert parsed.minimum_goal_distance_m == .006


@pytest.mark.parametrize("values", [(.001, .005, 25), (0., .001, 25),
                                     (float("nan"), .001, 25), (.005, .001, 0)])
def test_invalid_schedule_is_rejected(values):
    with pytest.raises(ValueError):
        ToleranceCurriculum(*values)


def test_tolerance_changes_only_between_episodes_and_goal_floor_is_fixed():
    schedule = ToleranceCurriculum(decay_steps=2)
    plant = env(compute_jacobian=False, tolerance_curriculum=schedule)
    try:
        options = {"joints": [-.03, 0.]}
        _, first = plant.reset(seed=7, options=options)
        assert first["position_tolerance"] == .005 and first["error"] >= .006
        for _ in range(4):
            _, _, _, _, info = plant.step([0., 0.])
            assert info["position_tolerance"] == .005
        _, second = plant.reset(options=options)
        assert second["position_tolerance"] == .001 and second["error"] >= .006
        assert plant.observation_space.contains(plant._observation())
    finally:
        plant.close()


def test_seeded_reset_restarts_curriculum_clock():
    plant = env(compute_jacobian=False, tolerance_curriculum=ToleranceCurriculum(decay_steps=2))
    try:
        options = {"joints": [-.03, 0.]}
        plant.reset(seed=9, options=options)
        for _ in range(4): plant.step([0., 0.])
        _, progressed = plant.reset(options=options)
        assert progressed["position_tolerance"] == .001
        _, restarted = plant.reset(seed=9, options=options)
        assert restarted["position_tolerance"] == .005
    finally:
        plant.close()


def test_fixed_checkpoint_environment_uses_final_tolerance_and_bounds():
    schedule = ToleranceCurriculum()
    training = JointConstrainedReachEnv(task_profile="generalized_reach", compute_jacobian=False,
                                        tolerance_curriculum=schedule)
    config = {"system": "ctr_0", "task_profile": "generalized_reach",
              "tolerance_m": .001, "evaluation_tolerance_m": .001,
              "observation_tolerance_bound_m": .005,
              "episode_steps": 4, "task_settings": training.task_settings,
              "solver_options": {**training.solver.options.__dict__},
              "action_scales": training.action_scales.tolist()}
    evaluation = make_reach_env(config)
    try:
        assert evaluation.tolerance_m == .001
        assert evaluation.tolerance_curriculum is None
        assert evaluation.observation_space == training.observation_space
        _, info = evaluation.reset(seed=10)
        assert info["error"] >= .006 and evaluation.costs["sensitivity_calls"] == 0
    finally:
        training.close(); evaluation.close()


def test_curriculum_env_contract():
    from gymnasium.utils.env_checker import check_env
    plant = env(compute_jacobian=False, tolerance_curriculum=ToleranceCurriculum(decay_steps=4))
    try:
        check_env(plant, skip_render_check=True)
    finally:
        plant.close()
