"""Source-state sensitivity data and a loss using the original joint update."""
import numpy as np
import torch
from ctr_reach_envs.her_replay_buffer import GoalTerminationHerReplayBuffer
from ctr_reach_envs.mechanics.rl_jacobian import JacobianHerReplayBuffer, JacobianDDPG
from stable_baselines3 import DDPG


class OriginalHerReplayBuffer(JacobianHerReplayBuffer):
    """Store normalized proposals; constraints belong to the plant transition."""
    def __init__(self, *args, action_semantics="proposal", **kwargs):
        if action_semantics != "proposal":
            raise ValueError("Original-IVP replay requires proposal actions")
        super().__init__(*args, action_semantics=action_semantics, **kwargs)

    def add(self, obs, next_obs, action, reward, done, infos):
        if len(infos) != self.n_envs:
            raise ValueError("One source-state record per environment is required")
        for index, info in enumerate(infos):
            proposal = np.asarray(info["proposed_action"], dtype=np.float32)
            # SB3 unscales even a [-1,1] Box, adding float32 roundoff.
            if (proposal.shape != (6,) or not np.all(np.isfinite(proposal)) or np.any(abs(proposal) > 1.)
                    or not np.allclose(proposal, action[index], atol=1e-6, rtol=0)):
                raise ValueError("Replay action differs from original-plant proposal")
            context = info["physics"]
            for key, shape in (("jacobian", (3, 6)), ("joints", (6,)), ("action_scales", (6,))):
                value = np.asarray(context[key])
                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"Invalid source-state {key}")
            if np.any(np.asarray(context["action_scales"]) <= 0):
                raise ValueError("Nonpositive action scale")
            if not np.array_equal(context["joints"], info["q_before"]):
                raise ValueError("Jacobian belongs to a different source state")
            if not isinstance(context["jacobian_valid"], (bool, np.bool_)):
                raise ValueError("jacobian_valid must be Boolean")
        # Deliberately bypass the equilibrium-specific executed-action checks.
        GoalTerminationHerReplayBuffer.add(self, obs, next_obs, action, reward, done, infos)


class OriginalJacobianLoss(torch.nn.Module):
    """Tracking or minimum-progress loss through the original joint update.

    The Jacobian is a detached local model. Only the actor's constrained joint
    displacement is differentiated; goals always come from the sampled HER batch.
    """
    def __init__(self, lengths, n_substeps=10, constrain_alpha=False,
                 cartesian_scale=.002, gain=.5, max_tip_step=.002, loss_kind="tracking"):
        super().__init__()
        if loss_kind not in ("tracking", "progress"):
            raise ValueError("Unknown original-IVP physics loss")
        self.loss_kind = loss_kind
        self.register_buffer("lengths", torch.tensor(lengths, dtype=torch.float64))
        self.n_substeps, self.constrain_alpha = int(n_substeps), bool(constrain_alpha)
        self.cartesian_scale, self.gain, self.max_tip_step = cartesian_scale, gain, max_tip_step
        if self.n_substeps < 1 or any(not np.isfinite(x) or x <= 0 for x in (cartesian_scale, gain, max_tip_step)):
            raise ValueError("Invalid local displacement loss parameters")

    def projected_delta(self, actions, joints, scales, *, detach_source=True):
        source = (joints.detach() if detach_source else joints).double()
        q = source
        increment = actions.double().clamp(-1., 1.) * scales.detach().double()
        for _ in range(self.n_substeps):
            command = q + increment
            beta_tensor = torch.minimum(torch.maximum(command[:, :3], -self.lengths + .001),
                                        torch.zeros_like(command[:, :3]))
            beta = list(beta_tensor.unbind(1))
            for i in (1, 2):
                beta[i - 1] = torch.maximum(torch.minimum(beta[i - 1], beta[i]),
                                            self.lengths[i] - self.lengths[i - 1] + beta[i])
            alpha = command[:, 3:]
            if self.constrain_alpha:
                alpha = alpha.clamp(-np.pi, np.pi)
            q = torch.cat((torch.stack(beta, 1), alpha), 1)
        return q - source

    def forward(self, actions, sample):
        dq = self.projected_delta(actions, sample.joints, sample.action_scales)
        predicted = torch.bmm(sample.jacobian.detach().double(), dq.unsqueeze(-1)).squeeze(-1)
        error = (sample.observations["desired_goal"] - sample.observations["achieved_goal"]).detach().double()
        if self.loss_kind == "progress":
            distance = torch.linalg.vector_norm(error, dim=1)
            next_distance = torch.linalg.vector_norm(error - predicted, dim=1)
            # Never request more progress than the remaining error, even if a
            # tracking-era configuration used a gain greater than one.
            required = torch.minimum(self.gain * distance, distance).clamp_max(self.max_tip_step)
            errors = torch.relu((next_distance - distance + required) / self.cartesian_scale).square()
        else:
            # Preserve the existing tracking objective for old checkpoints/runs.
            desired = self.gain * error
            desired = desired / (torch.linalg.vector_norm(desired, dim=1, keepdim=True) / self.max_tip_step).clamp_min(1.)
            errors = ((predicted - desired) / self.cartesian_scale).square().sum(1)
        valid = (torch.ones_like(errors) if sample.jacobian_valid is None
                 else sample.jacobian_valid.detach().reshape(-1).to(errors.dtype))
        return (errors * valid).sum() / valid.sum().clamp_min(1.)


class OriginalJacobianDDPG(JacobianDDPG):
    # Identical exploration, including random-number consumption at epsilon=0.
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = DDPG._sample_action(self, learning_starts, action_noise, n_envs)
        if self.num_timesteps >= learning_starts:
            random_mask = np.random.random(n_envs) < self.random_exploration
            for index in np.flatnonzero(random_mask):
                action[index] = self.action_space.sample()
                buffer_action[index] = self.policy.scale_action(action[index])
        return action, buffer_action

    def __init__(self, *args, original_n_substeps=10, original_constrain_alpha=False,
                 physics_loss_kind="tracking", short_horizon_steps=2,
                 short_horizon_batch_size=4, short_horizon_every=20,
                 short_horizon_start_steps=10000, **kwargs):
        if physics_loss_kind not in ("tracking", "progress", "short_horizon"):
            raise ValueError("Unknown original-IVP physics loss")
        for value in (short_horizon_steps, short_horizon_batch_size, short_horizon_every):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Short-horizon sizes and frequency must be positive integers")
        if short_horizon_steps > 8:
            raise ValueError("Initial short-horizon implementation supports at most eight steps")
        if isinstance(short_horizon_start_steps, bool) or not isinstance(short_horizon_start_steps, int) or short_horizon_start_steps < 0:
            raise ValueError("Short-horizon start step must be a nonnegative integer")
        self.short_horizon_steps = short_horizon_steps
        self.short_horizon_batch_size = short_horizon_batch_size
        self.short_horizon_every = short_horizon_every
        self.short_horizon_start_steps = short_horizon_start_steps
        self._short_horizon = None
        self.short_horizon_totals = {}
        self.physics_loss_kind = physics_loss_kind
        self.original_n_substeps = int(original_n_substeps)
        self.original_constrain_alpha = bool(original_constrain_alpha)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps, batch_size=100):
        short = getattr(self, "physics_loss_kind", "tracking") == "short_horizon"
        previous = self.physics_update_count
        if short:
            self._window_block = []
        super().train(gradient_steps, batch_size)
        self.last_physics_metrics["physics_loss_kind"] = getattr(self, "physics_loss_kind", "tracking")
        if short:
            records = self._window_block
            self.physics_update_count = previous + sum(r["valid_windows"] > 0 for r in records)
            # Old field names describe cached local Jacobians, not this mode.
            self.last_physics_metrics["short_horizon_loss_all_updates_mean"] = self.last_physics_metrics.pop("jacobian_loss", 0.)
            self.last_physics_metrics.pop("jacobian_valid_fraction", None)
            self.last_physics_metrics.pop("jacobian_action_gradient_norm", None)
            attempted = sum(r["attempted_windows"] for r in records)
            valid = sum(r["valid_windows"] for r in records)
            self.last_physics_metrics.update(
                short_horizon_scheduled_updates=len(records),
                short_horizon_attempted_windows=attempted,
                short_horizon_valid_windows=valid,
                short_horizon_invalid_windows=sum(r["invalid_windows"] for r in records),
                short_horizon_initial_terminal_windows=sum(r["initial_terminal_windows"] for r in records),
                short_horizon_terminal_windows=sum(r["terminal_windows"] for r in records),
                short_horizon_valid_fraction=valid / attempted if attempted else 0.,
                short_horizon_loss_active_mean=(float(np.mean([r["loss"] for r in records if r["valid_windows"]]))
                    if valid else 0.),
                short_horizon_totals=dict(self.short_horizon_totals))
            active = [r["integration"] for r in records if r["valid_windows"] and "integration" in r]
            for key in ("weighted_aux_norm_ratio_before", "weighted_aux_norm_ratio_after",
                        "gradient_conflict", "gradient_cosine", "jacobian_parameter_gradient_norm",
                        "actor_step_skipped", "actor_step_rl_only"):
                self.last_physics_metrics["short_horizon_active_" + key] = (
                    float(np.mean([r[key] for r in active])) if active else 0.)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_short_horizon", "_window_block"]

    def close_short_horizon(self):
        if self._short_horizon is not None:
            self._short_horizon.backend.close()
            self._short_horizon = None

    def _actor_auxiliary_loss(self, actions, data):
        if getattr(self, "physics_loss_kind", "tracking") != "short_horizon":
            return super()._actor_auxiliary_loss(actions, data)
        if self.num_timesteps < self.short_horizon_start_steps or self._n_updates % self.short_horizon_every:
            return actions.sum() * 0.
        if self._short_horizon is None:
            from .short_horizon import IVPBackwardBackend, ShortHorizonReturn, validate_short_horizon
            validate_short_horizon(self.original_ivp_config)
            self._short_horizon = ShortHorizonReturn(
                IVPBackwardBackend(self.original_ivp_config, self.short_horizon_totals),
                self._physics_loss, self.short_horizon_steps, self.gamma)
        # Cycle through the existing real/HER batch without another RNG draw.
        count = min(self.short_horizon_batch_size, len(actions))
        start = ((self._n_updates // self.short_horizon_every - 1) * count) % len(actions)
        indices = [(start + i) % len(actions) for i in range(count)]
        loss, record = self._short_horizon.loss(
            self.actor, self.actor_target, self.critic_target, actions, data, indices)
        record["update_index"] = self._n_updates
        self._window_block.append(record)
        return loss

    def _record_auxiliary_integration(self, metrics):
        if (getattr(self, "physics_loss_kind", "tracking") == "short_horizon" and self._window_block
                and self._window_block[-1]["update_index"] == self._n_updates):
            self._window_block[-1]["integration"] = dict(metrics)

    def _make_physics_loss(self):
        return OriginalJacobianLoss(self.physics_lengths, self.original_n_substeps,
            self.original_constrain_alpha, self.physics_cartesian_scale,
            self.physics_gain, self.physics_max_tip_step,
            loss_kind=("tracking" if getattr(self, "physics_loss_kind", "tracking") == "short_horizon"
                       else getattr(self, "physics_loss_kind", "tracking")))
