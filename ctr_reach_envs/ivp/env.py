"""Original transition with optional sensitivity diagnostics and observations."""
from collections import Counter
import time
import gymnasium as gym
import numpy as np
from ctr_reach_envs.envs.ctr_reach_env import CtrReachEnv
from .sensitivity import tip_sensitivity, IVPSensitivityError
from .observations import FEATURE_COUNTS, observation_settings, physics_features


class OriginalIVPEnv(CtrReachEnv):
    def __init__(self, *args, compute_jacobian=False, physics_observation=None, **kwargs):
        self.physics_observation = observation_settings(physics_observation)
        self.observation_uses_jacobian = self.physics_observation["mode"] in ("jacobian", "jacobian_limits")
        self.compute_jacobian = bool(compute_jacobian or self.observation_uses_jacobian)
        self._observation_physics_cache = None
        self.costs = dict(forward_calls=0, forward_seconds=0., sensitivity_calls=0,
                          sensitivity_seconds=0., sensitivity_rhs=0, invalid_jacobians=0)
        self.invalid_reasons = Counter()
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        count = FEATURE_COUNTS[self.physics_observation["mode"]]
        if count:
            self.observation_space = gym.spaces.Dict({
                **self.observation_space.spaces,
                "physics": gym.spaces.Box(
                    low=np.r_[np.full(18, -1.), np.zeros(count - 18)].astype(np.float32),
                    high=np.ones(count, dtype=np.float32), dtype=np.float32),
            })
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

    def reset(self, *, seed=None, options=None):
        # Parameters are reset/randomized by the parent, even if q is unchanged.
        self._observation_physics_cache = None
        return super().reset(seed=seed, options=options)

    def _observation(self):
        observation = super()._observation()
        if self.physics_observation["mode"] != "none":
            context = self.source_physics() if self.observation_uses_jacobian else None
            observation["physics"] = physics_features(
                context, self.trig_obj.tube_lengths[self.system], self.n_substeps,
                self.physics_observation)
        return observation

    def source_physics(self):
        context = dict(joints=self.trig_obj.joints.copy(), action_scales=self.action_scale.copy(),
                       jacobian=np.zeros((3, 6)), jacobian_valid=False)
        if not (self.compute_jacobian or self.observation_uses_jacobian):
            return context
        cached = self._observation_physics_cache
        if (self.observation_uses_jacobian and cached is not None
                and cached[0] == self.system and np.array_equal(cached[1], context["joints"])):
            context.update(jacobian=cached[2].copy(), jacobian_valid=cached[3])
            return context
        self.costs["sensitivity_calls"] += 1
        started = time.perf_counter()
        try:
            result = tip_sensitivity(self.model, context["joints"], self.system)
            self.costs["sensitivity_rhs"] += result.rhs_evaluations
            if (np.shape(result.jacobian) != (3, 6) or not np.all(np.isfinite(result.jacobian))
                    or np.shape(result.tip) != (3,) or not np.all(np.isfinite(result.tip))):
                raise IVPSensitivityError("nonfinite or malformed sensitivity result")
            if np.linalg.norm(result.tip - self.achieved_goal) > 2e-5:
                raise IVPSensitivityError("forward/sensitivity tip discrepancy exceeds 20 micrometres")
            context.update(jacobian=result.jacobian, jacobian_valid=True)
        except (IVPSensitivityError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.costs["invalid_jacobians"] += 1
            self.invalid_reasons[str(exc)] += 1
        finally:
            self.costs["sensitivity_seconds"] += time.perf_counter() - started
        if self.observation_uses_jacobian:
            self._observation_physics_cache = (self.system, context["joints"].copy(),
                                               context["jacobian"].copy(), context["jacobian_valid"])
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
