from __future__ import annotations

import numpy as np

from ctr_reach_envs.envs.CTR_Python import Tube


def _randomize_value(value: float, fraction: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(value * (1.0 - fraction), value * (1.0 + fraction)))


def sample_parameters(tube: Tube, fraction: float, rng: np.random.Generator) -> Tube:
    """Randomize selected tube properties without changing kinematic limits."""

    if not 0.0 <= fraction < 1.0:
        raise ValueError("domain_rand must be in [0, 1)")
    diameter_scale = _randomize_value(1.0, fraction, rng)
    return Tube(
        length=tube.L,
        length_curved=tube.L_c,
        # A shared scale preserves wall thickness ordering.
        diameter_inner=tube.diameter_inner * diameter_scale,
        diameter_outer=tube.diameter_outer * diameter_scale,
        stiffness=_randomize_value(tube.E, fraction, rng),
        torsional_stiffness=_randomize_value(tube.G, fraction, rng),
        x_curvature=_randomize_value(tube.U_x, fraction, rng),
        y_curvature=tube.U_y,
    )
