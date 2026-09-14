"""Goal-independent mechanical features; no controller or learning objective."""
import numpy as np
from ctr_reach_envs.envs.obs import EXT_TOL


FEATURE_COUNTS = {"none": 0, "jacobian": 19, "jacobian_limits": 29, "zeros": 29}
DEFAULT_OBSERVATION = dict(mode="none", scale_m=.002, version=1)


def observation_settings(settings=None):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_OBSERVATION):
        raise ValueError("Unknown physics observation settings")
    resolved = {**DEFAULT_OBSERVATION, **settings}
    if resolved["mode"] not in FEATURE_COUNTS:
        raise ValueError("Unknown physics observation mode")
    if type(resolved["version"]) is not int or resolved["version"] != 1:
        raise ValueError("Unsupported physics observation version")
    if not np.isfinite(resolved["scale_m"]) or resolved["scale_m"] <= 0:
        raise ValueError("Physics observation scale must be finite and positive")
    return resolved


def extension_margins(joints, lengths):
    """Six box slacks and four adjacent-tube slacks, normalized to [0, 1].

    These are extension margins only. Unconstrained rotations have no finite
    distance to a limit; do not invent one from their periodic representation.
    """
    beta, lengths = np.asarray(joints)[:3], np.asarray(lengths)
    spans, differences = lengths - EXT_TOL, lengths[:-1] - lengths[1:]
    if np.any(spans <= 0) or np.any(differences <= 0):
        raise ValueError("Physics margins require strictly decreasing positive tube lengths")
    gaps = beta[1:] - beta[:-1]
    return np.clip(np.r_[(beta + spans) / spans, -beta / spans,
                         gaps / differences, (differences - gaps) / differences], 0., 1.)


def physics_features(context, lengths, n_substeps, settings):
    """Encode J(q) n_substeps D / scale using bounded signed compression.

    n_substeps D is the NOMINAL unconstrained action-to-joint scaling. The
    feature is not the derivative of the plant's clipped multi-substep update.
    Separate limit margins let the policy observe proximity to constraints.
    """
    mode = settings["mode"]
    result = np.zeros(FEATURE_COUNTS[mode], dtype=np.float32)
    if mode in ("none", "zeros"):
        return result
    if context["jacobian_valid"]:
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (np.asarray(context["jacobian"], dtype=np.float64)
                      * (n_substeps * np.asarray(context["action_scales"])) / settings["scale_m"])
        if scaled.shape == (3, 6) and np.all(np.isfinite(scaled)):
            result[:18] = (scaled / (1. + np.abs(scaled))).ravel()
            result[18] = 1.
    if mode == "jacobian_limits":
        result[19:] = extension_margins(context["joints"], lengths)
    return result
