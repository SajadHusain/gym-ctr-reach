"""Numerical summaries for local-motion diagnostics (all distances in metres)."""
import numpy as np


def motion_metrics(source_tip, goal, actual_delta, predicted_delta=None, epsilon_m=1e-6):
    source_tip, goal, actual_delta = [np.asarray(x, dtype=float) for x in
                                    (source_tip, goal, actual_delta)]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (source_tip, goal, actual_delta)):
        raise ValueError("Motion diagnostics require finite 3-vectors")
    initial = float(np.linalg.norm(goal-source_tip))
    actual_norm = float(np.linalg.norm(actual_delta))
    actual_error = float(np.linalg.norm(goal-source_tip-actual_delta))
    result = dict(initial_error_m=initial, actual_step_m=actual_norm,
                  actual_error_m=actual_error, actual_progress_m=initial-actual_error,
                  predicted_step_m=None, predicted_progress_m=None,
                  linearization_error_m=None, relative_linearization_error=None,
                  motion_cosine=None, progress_sign_agrees=None)
    if predicted_delta is not None:
        predicted_delta = np.asarray(predicted_delta, dtype=float)
        if predicted_delta.shape != (3,) or not np.isfinite(predicted_delta).all():
            raise ValueError("Prediction must be a finite 3-vector")
        predicted_norm = float(np.linalg.norm(predicted_delta))
        predicted_progress = initial-float(np.linalg.norm(goal-source_tip-predicted_delta))
        mismatch = float(np.linalg.norm(predicted_delta-actual_delta))
        cosine = (float(np.clip(predicted_delta.dot(actual_delta)/(predicted_norm*actual_norm), -1., 1.))
                  if min(predicted_norm, actual_norm) > epsilon_m else None)
        # Motions/progress below the numerical floor do not count as sign tests.
        sign = (bool(predicted_progress*result["actual_progress_m"] > 0)
                if min(abs(predicted_progress), abs(result["actual_progress_m"])) > epsilon_m else None)
        result.update(predicted_step_m=predicted_norm, predicted_progress_m=predicted_progress,
                      linearization_error_m=mismatch,
                      relative_linearization_error=mismatch/max(actual_norm, epsilon_m),
                      motion_cosine=cosine, progress_sign_agrees=sign)
    return result


def summarize_motion(rows):
    success = [r for r in rows if not r.get("failure")]
    result = dict(rows=len(rows), failures=len(rows)-len(success),
                  valid_jacobians=sum(bool(r["jacobian_valid"]) for r in rows))
    for field in ("actual_progress_m", "linearization_error_m", "relative_linearization_error",
                  "motion_cosine", "projection_error_inf", "actual_progress_change_m"):
        values = [r[field] for r in success if r.get(field) is not None]
        result[field] = dict(count=len(values), mean=float(np.mean(values)) if values else None,
                             median=float(np.median(values)) if values else None,
                             p95=float(np.percentile(values, 95)) if values else None)
    predicted_positive = [r for r in success if r.get("predicted_progress_m") is not None
                          and r["predicted_progress_m"] > 1e-6]
    result["predicted_positive_count"] = len(predicted_positive)
    result["actual_positive_given_predicted_positive"] = (
        sum(r["actual_progress_m"] > 1e-6 for r in predicted_positive)/len(predicted_positive)
        if predicted_positive else None)
    return result
