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
    """Differentiable transcription of Obs.set_action, including all substeps."""
    def __init__(self, lengths, n_substeps=10, constrain_alpha=False,
                 cartesian_scale=.002, gain=.5, max_tip_step=.002):
        super().__init__()
        self.register_buffer("lengths", torch.tensor(lengths, dtype=torch.float64))
        self.n_substeps, self.constrain_alpha = int(n_substeps), bool(constrain_alpha)
        self.cartesian_scale, self.gain, self.max_tip_step = cartesian_scale, gain, max_tip_step
        if self.n_substeps < 1 or any(not np.isfinite(x) or x <= 0 for x in (cartesian_scale, gain, max_tip_step)):
            raise ValueError("Invalid local displacement loss parameters")

    def projected_delta(self, actions, joints, scales):
        source = joints.detach().double()
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
        desired = self.gain * (sample.observations["desired_goal"] - sample.observations["achieved_goal"]).detach().double()
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

    def __init__(self, *args, original_n_substeps=10, original_constrain_alpha=False, **kwargs):
        self.original_n_substeps = int(original_n_substeps)
        self.original_constrain_alpha = bool(original_constrain_alpha)
        super().__init__(*args, **kwargs)

    def _make_physics_loss(self):
        return OriginalJacobianLoss(self.physics_lengths, self.original_n_substeps,
            self.original_constrain_alpha, self.physics_cartesian_scale,
            self.physics_gain, self.physics_max_tip_step)
