import csv

import numpy as np

from evaluate import wilson_interval
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
    env = make_env(evaluation=True, seed=7, position_tolerance=0.0181)
    assert env.unwrapped.get_goal_tolerance() == 0.0181
    env.close()
