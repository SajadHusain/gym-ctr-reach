import csv
import json
import sys

import numpy as np
from stable_baselines3 import DDPG

from evaluate import main as evaluate_main, wilson_interval
from follow_path import load_waypoints
from rl_utils import make_env


def test_waypoint_csv_loader(tmp_path):
    path = tmp_path / "path.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "z"])
        writer.writeheader()
        writer.writerow({"x": 0.01, "y": -0.02, "z": 0.2})
    points = load_waypoints(path)
    np.testing.assert_allclose(points, [[0.01, -0.02, 0.2]])


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(750, 1000)
    assert low < 0.75 < high
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0


def test_evaluation_tolerance_override():
    default_env = make_env(evaluation=True, seed=7)
    env = make_env(evaluation=True, seed=7, position_tolerance=0.0181)
    assert env.unwrapped.get_goal_tolerance() == 0.0181
    assert env.observation_space == default_env.observation_space
    env.close()
    default_env.close()


def test_recording_trajectories_preserves_evaluation(tmp_path, monkeypatch):
    env = make_env(evaluation=True, seed=7)
    checkpoint = tmp_path / "model.zip"
    try:
        DDPG("MultiInputPolicy", env, seed=0, device="cpu", buffer_size=20,
             policy_kwargs={"net_arch": [16, 16]}).save(checkpoint)
    finally:
        env.close()
    episode_results = []
    for record in (False, True):
        output = tmp_path / ("trace" if record else "plain")
        arguments = ["evaluate.py", str(checkpoint), "--episodes", "2", "--seed", "100000",
                     "--output-dir", str(output)]
        if record:
            arguments.append("--record-trajectories")
        monkeypatch.setattr(sys, "argv", arguments)
        evaluate_main()
        with (output / "episodes.csv").open(encoding="utf-8") as handle:
            episode_results.append(list(csv.DictReader(handle)))
        assert json.loads((output / "summary.json").read_text())["complete"]
    assert episode_results[0] == episode_results[1]
    with (tmp_path / "trace" / "trajectories.csv").open(encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    assert len(trace) == sum(int(row["steps"]) for row in episode_results[1])
    for row in trace:
        np.testing.assert_allclose(
            float(row["delta_beta_0_m"]),
            float(row["after_beta_0_m"]) - float(row["before_beta_0_m"]), atol=1e-12,
        )
