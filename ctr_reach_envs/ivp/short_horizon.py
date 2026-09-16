"""Budgeted, SHAC-inspired actor returns on the original IVP plant.

This is an off-policy DDPG/HER adaptation, not the on-policy SHAC algorithm.
Every imagined transition runs the nonlinear plant. Its analytical Jacobian
provides a first-order backward pass at that transition's new configuration.
Sparse rewards and success branches are locally constant; their discontinuous
boundary derivatives are NOT estimated by this pathwise gradient.
"""
from contextlib import contextmanager
import time

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .config import make_env
from .sensitivity import IVPSensitivityError, tip_sensitivity


class UnavailableWindow(RuntimeError):
    """A numerical solve or analytical derivative invalidates the whole window."""


def validate_short_horizon(config):
    p = config["physics"]
    if p.get("loss_kind") != "short_horizon":
        return
    if config["segment_mode"] != "continuous":
        raise ValueError("short_horizon requires --segment-mode continuous")
    if config["spec"].get("physics_observation", {}).get("mode", "none") != "none":
        raise ValueError("short_horizon currently requires --physics-observation none")
    if p["objective"] != "hybrid" or p["integration"] != "rl_priority":
        raise ValueError("short_horizon requires the hybrid actor and rl_priority integration")
    if p.get("diagnostics", False):
        raise ValueError("--physics-diagnostics probes local losses only; short_horizon logs window diagnostics automatically")
    for name, default in (("short_horizon_steps", 2), ("short_horizon_batch_size", 4), ("short_horizon_every", 20)):
        value = p.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if p.get("short_horizon_steps", 2) > 8:
        raise ValueError("short_horizon supports at most eight steps; validate small windows first")
    start = p.get("short_horizon_start_steps", 10000)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("short_horizon_start_steps must be a nonnegative integer")


class IVPBackwardBackend:
    """Private plant: imagined states never overwrite the collection environment."""
    def __init__(self, config, totals=None):
        validate_short_horizon(config)
        self.totals = {} if totals is None else totals
        for name in ("forward_calls", "sensitivity_calls", "sensitivity_rhs", "invalid_windows",
                     "attempted_windows", "valid_windows", "terminal_windows", "initial_terminal_windows",
                     "rollout_steps", "scheduled_updates"):
            self.totals.setdefault(name, 0)
        for name in ("forward_seconds", "sensitivity_seconds", "rollout_seconds"):
            self.totals.setdefault(name, 0.)
        self.totals.setdefault("invalid_reasons", {})
        self.env = make_env(config, compute_jacobian=False)
        # Include the one constructor solve in the separate model budget.
        self.totals["forward_calls"] += self.env.costs["forward_calls"]
        self.totals["forward_seconds"] += self.env.costs["forward_seconds"]

    def forward_and_jacobian(self, q):
        self.totals["forward_calls"] += 1
        started = time.perf_counter()
        try:
            tip = np.asarray(self.env.model.forward_kinematics(q, 0), dtype=np.float64)
            if tip.shape != (3,) or not np.isfinite(tip).all():
                raise ValueError("nonfinite or malformed forward tip")
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise UnavailableWindow("forward: " + str(exc)) from exc
        finally:
            self.totals["forward_seconds"] += time.perf_counter() - started
        self.totals["sensitivity_calls"] += 1
        started = time.perf_counter()
        try:
            result = tip_sensitivity(self.env.model, q, 0)
            self.totals["sensitivity_rhs"] += result.rhs_evaluations
            if result.jacobian.shape != (3, 6) or not np.isfinite(result.jacobian).all():
                raise IVPSensitivityError("nonfinite or malformed Jacobian")
            if not np.isfinite(result.tip).all() or np.linalg.norm(result.tip - tip) > 2e-5:
                raise IVPSensitivityError("forward/sensitivity tip discrepancy exceeds 20 micrometres")
        except (IVPSensitivityError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise UnavailableWindow("sensitivity: " + str(exc)) from exc
        finally:
            self.totals["sensitivity_seconds"] += time.perf_counter() - started
        return tip, result.jacobian

    def close(self):
        self.env.close()


class _IVPTip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, joints, backend):
        if joints.shape != (1, 6):
            raise ValueError("The CPU IVP bridge expects one six-joint configuration")
        tip, jacobian = backend.forward_and_jacobian(joints.detach().double().cpu().numpy()[0])
        ctx.save_for_backward(torch.as_tensor(jacobian, dtype=joints.dtype, device=joints.device))
        return torch.as_tensor(tip, dtype=joints.dtype, device=joints.device).reshape(1, 3)

    @staticmethod
    @once_differentiable
    def backward(ctx, tip_gradient):
        (jacobian,) = ctx.saved_tensors
        return tip_gradient @ jacobian, None


def differentiable_tip(joints, backend):
    """First-order derivatives only; no Jacobian is reused at a different q."""
    return _IVPTip.apply(joints, backend)


def observation_from_state(joints, tip, goal, tolerance):
    """Exact egocentric observation encoding, retaining input derivatives."""
    beta, alpha = joints[:, :3], joints[:, 3:]
    beta = torch.cat((beta[:, :1], beta[:, 1:] - beta[:, :-1]), dim=1)
    alpha = torch.cat((alpha[:, :1], alpha[:, 1:] - alpha[:, :-1]), dim=1)
    features = torch.stack((alpha.cos(), alpha.sin(), beta), dim=-1).flatten(1)
    # The actual plant exposes float32 observations, also for float64 joints.
    return {"observation": torch.cat((features, tolerance.reshape(-1, 1)), dim=1).float(),
            "achieved_goal": tip.float(), "desired_goal": goal.float()}


@contextmanager
def frozen_parameters(*modules):
    parameters = list(dict.fromkeys(p for m in modules for p in m.parameters()))
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def terminal_value(actor_target, critic_target, observation):
    # q1_forward uses no_grad for feature extraction in SB3. Calling its
    # feature extractor and Q network directly retains dQ/ds as well as dQ/da.
    # Freeze weights, never the forward pass: V_bar(s) = Q_bar(s, mu_bar(s)).
    with frozen_parameters(actor_target, critic_target):
        action = actor_target(observation)
        features = critic_target.extract_features(observation, critic_target.features_extractor)
        return critic_target.q_networks[0](torch.cat((features, action), dim=1)).reshape(())


class ShortHorizonReturn:
    def __init__(self, backend, projection, horizon, gamma):
        self.backend, self.projection = backend, projection
        self.horizon, self.gamma = horizon, gamma

    def loss(self, actor, actor_target, critic_target, first_actions, sample, indices):
        started = time.perf_counter()
        values = []
        record = dict(attempted_windows=len(indices), valid_windows=0, invalid_windows=0,
                      initial_terminal_windows=0, terminal_windows=0, rollout_steps=0)
        totals = self.backend.totals
        totals["scheduled_updates"] += 1
        for i in indices:
            q = sample.joints[i:i+1].detach().double()
            scales = sample.action_scales[i:i+1].detach().double()
            goal = sample.observations["desired_goal"][i:i+1].detach()
            tip = sample.observations["achieved_goal"][i:i+1].detach()
            tolerance = sample.observations["observation"][i:i+1, 9:10].detach()
            # Some HER source states already satisfy the relabeled goal.
            if bool(torch.linalg.vector_norm(tip - goal) <= tolerance.reshape(())):
                record["initial_terminal_windows"] += 1
                continue
            action = first_actions[i:i+1]
            value = action.sum() * 0.  # Sparse terminal returns may have zero gradient.
            try:
                for k in range(self.horizon):
                    q = q + self.projection.projected_delta(action, q, scales, detach_source=False)
                    tip = differentiable_tip(q, self.backend)
                    record["rollout_steps"] += 1
                    observation = observation_from_state(q, tip, goal, tolerance)
                    success = bool(torch.linalg.vector_norm(observation["achieved_goal"] - goal)
                                   <= tolerance.reshape(()))
                    value = value + (self.gamma ** k) * (0. if success else -1.)
                    if success:
                        record["terminal_windows"] += 1
                        break  # True task termination: no terminal bootstrap.
                    if k + 1 == self.horizon:
                        value = value + (self.gamma ** self.horizon) * terminal_value(
                            actor_target, critic_target, observation)
                    else:
                        action = actor(observation)
            except UnavailableWindow as exc:
                record["invalid_windows"] += 1
                reason = str(exc)
                totals["invalid_reasons"][reason] = totals["invalid_reasons"].get(reason, 0) + 1
                continue  # Discard the entire graph, never substitute a zero Jacobian.
            if not bool(torch.isfinite(value)):
                raise FloatingPointError("Nonfinite short-horizon actor return")
            values.append(-value / self.horizon)
            record["valid_windows"] += 1
        loss = torch.stack(values).mean() if values else first_actions.sum() * 0.
        # Also connects unused first actions so the parent action-gradient audit
        # is well-defined when all windows are invalid or already terminal.
        loss = loss + first_actions.sum() * 0.
        for key in record:
            totals[key] += record[key]
        totals["rollout_seconds"] += time.perf_counter() - started
        record["loss"] = float(loss.detach())
        return loss, record
