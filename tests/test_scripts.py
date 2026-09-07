import csv

import numpy as np

from evaluate import wilson_interval
from follow_path import load_waypoints


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
