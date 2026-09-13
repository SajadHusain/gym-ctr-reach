"""Training instrumentation around the original transition, never a controller."""
from collections import Counter
import time
import numpy as np
from ctr_reach_envs.envs.ctr_reach_env import CtrReachEnv
from .sensitivity import tip_sensitivity, IVPSensitivityError


class OriginalIVPEnv(CtrReachEnv):
    def __init__(self, *args, compute_jacobian=False, **kwargs):
        self.compute_jacobian = bool(compute_jacobian)
        self.costs = dict(forward_calls=0, forward_seconds=0., sensitivity_calls=0,
                          sensitivity_seconds=0., sensitivity_rhs=0, invalid_jacobians=0)
        self.invalid_reasons = Counter()
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        # Constructor performs one forward solve; time includes environment setup.
        self.costs["forward_calls"] = 1
        self.costs["forward_seconds"] = time.perf_counter() - started
        forward = self.model.forward_kinematics

        def measured_forward(*a, **kw):
            started = time.perf_counter()
            self.costs["forward_calls"] += 1
            try:
                return forward(*a, **kw)
            finally:
                self.costs["forward_seconds"] += time.perf_counter() - started

        self.model.forward_kinematics = measured_forward

    def source_physics(self):
        context = dict(joints=self.trig_obj.joints.copy(), action_scales=self.action_scale.copy(),
                       jacobian=np.zeros((3, 6)), jacobian_valid=False)
        if not self.compute_jacobian:
            return context
        self.costs["sensitivity_calls"] += 1
        started = time.perf_counter()
        try:
            result = tip_sensitivity(self.model, context["joints"], self.system)
            self.costs["sensitivity_rhs"] += result.rhs_evaluations
            if np.linalg.norm(result.tip - self.achieved_goal) > 2e-5:
                raise IVPSensitivityError("forward/sensitivity tip discrepancy exceeds 20 micrometres")
            context.update(jacobian=result.jacobian, jacobian_valid=True)
        except (IVPSensitivityError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.costs["invalid_jacobians"] += 1
            self.invalid_reasons[str(exc)] += 1
        finally:
            self.costs["sensitivity_seconds"] += time.perf_counter() - started
        return context

    def step(self, action):
        source = self.source_physics()
        source_tip = self.achieved_goal.copy()
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(q_before=source["joints"], q_after=self.trig_obj.joints.copy(),
                    proposed_action=np.clip(np.asarray(action, dtype=np.float32), -1., 1.),
                    applied_delta_q=self.trig_obj.joints - source["joints"],
                    physics=source, action_source="plant")
        if source["jacobian_valid"] and not info["solver_failure"]:
            # Logged only: this diagnostic never changes the reward or transition.
            predicted = source["jacobian"] @ info["applied_delta_q"]
            info["jacobian_predicted_delta_m"] = predicted
            info["jacobian_linearization_error_m"] = float(np.linalg.norm(predicted - (self.achieved_goal - source_tip)))
        return observation, reward, terminated, truncated, info
