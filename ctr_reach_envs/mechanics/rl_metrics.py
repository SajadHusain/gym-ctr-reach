"""Episode-local action variation; no physical timestep or dynamics is assumed."""
import numpy as np


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
