"""Episode-local action variation; no physical timestep or dynamics is assumed."""
import numpy as np


class ReachHoldMetrics:
    """First-hit reaching and final-window holding, measured independently.

    Holding requires every error in the final K executed steps to satisfy the
    tolerance. One late hit or a short/interrupted trajectory cannot count.
    This is an empirical criterion, not a stability certificate.
    """
    def __init__(self, tolerance_m, hold_steps=10):
        if not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("Tolerance must be positive and finite")
        if isinstance(hold_steps, bool) or not isinstance(hold_steps, int) or hold_steps < 1:
            raise ValueError("Holding window must be a positive integer")
        self.tolerance_m, self.hold_steps = float(tolerance_m), hold_steps
        self.errors = []

    def add(self, error_m):
        if not np.isfinite(error_m) or error_m < 0:
            raise ValueError("Error must be finite and nonnegative")
        self.errors.append(float(error_m))

    def summary(self, *, episode_complete=True):
        errors = np.asarray(self.errors)
        hits = np.flatnonzero(errors <= self.tolerance_m)
        full = len(errors) >= self.hold_steps
        window = errors[-self.hold_steps:]
        return dict(reached=bool(len(hits)),
            first_success_step=int(hits[0])+1 if len(hits) else None,
            final_success=bool(len(errors) and errors[-1] <= self.tolerance_m),
            sustained_success=bool(episode_complete and full and np.all(window <= self.tolerance_m)),
            hold_window_steps=self.hold_steps, hold_window_observed=full,
            hold_window_max_error_m=float(window.max()) if full else None,
            hold_window_rms_error_m=float(np.sqrt(np.mean(window**2))) if full else None,
            within_tolerance_fraction=float(np.mean(errors <= self.tolerance_m)) if len(errors) else None,
            post_first_hit_max_error_m=float(errors[hits[0]:].max()) if len(hits) else None)


class EpisodeMotion:
    def __init__(self, joint_count):
        self.n = joint_count
        self.proposed, self.executed, self.increments = [], [], []

    def add(self, info):
        values = [np.asarray(info[k], dtype=float) for k in
                  ("proposed_action", "executed_action", "applied_delta")]
        if any(v.shape != (2*self.n,) or not np.all(np.isfinite(v)) for v in values):
            raise ValueError("Invalid action motion record")
        for records, value in zip((self.proposed, self.executed, self.increments), values):
            records.append(value.copy())

    def summary(self):
        count = len(self.executed)
        result = {"motion_steps": count, "action_difference_pairs": max(0, count-1),
                  "action_second_difference_triples": max(0, count-2)}
        for name, values in (("proposed", self.proposed), ("executed", self.executed)):
            a = np.asarray(values).reshape(-1, 2*self.n)
            d = np.diff(a, axis=0)
            dd = np.diff(a, n=2, axis=0)
            # RMS is per joint and per pair, not a sum that grows with duration.
            result[name+"_action_rms"] = float(np.sqrt(np.mean(a*a))) if count else None
            result[name+"_action_change_rms"] = float(np.sqrt(np.mean(d*d))) if len(d) else None
            result[name+"_action_total_variation"] = float(np.linalg.norm(d, axis=1).sum())
            result[name+"_action_second_difference_rms"] = float(np.sqrt(np.mean(dd*dd))) if len(dd) else None
        dq = np.asarray(self.increments).reshape(-1, 2*self.n)
        change = np.diff(dq, axis=0)
        for name, part in (("translation", change[:, :self.n]), ("rotation", change[:, self.n:])):
            unit = "m" if name == "translation" else "rad"
            result[f"{name}_increment_change_rms_{unit}"] = float(np.sqrt(np.mean(part*part))) if len(part) else None
        result["held_action_fraction"] = float(np.mean(np.max(np.abs(dq), axis=1) < 1e-12)) if count else None
        return result


def summarize_motion(rows):
    """Episode-weighted means with explicit counts; single-action episodes stay missing."""
    names = EpisodeMotion(1).summary()
    result = {}
    for name in names:
        values = [float(r[name]) for r in rows if r.get(name) is not None]
        result[name] = {"mean": float(np.mean(values)) if values else None, "episodes": len(values)}
    return result
