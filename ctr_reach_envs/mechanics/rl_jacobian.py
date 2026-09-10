# DDPG update adapted from Stable Baselines3, Copyright (c) 2019 Antonin Raffin.
# MIT license notice: docs/third_party/STABLE_BASELINES3_LICENSE.txt.
"""HER-consistent local Jacobian regularization of the DDPG actor.

All ODE derivatives are detached replay data. Gradients pass through the actor,
joint projection and matrix products, not through the equilibrium solver.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.utils import polyak_update
from .geometry import JointConstraints
from .rl_replay import ExecutedActionHerReplayBuffer


@dataclass
class PhysicsSamples:
    observations: dict
    actions: torch.Tensor
    next_observations: dict
    dones: torch.Tensor
    rewards: torch.Tensor
    jacobian: torch.Tensor
    joints: torch.Tensor
    action_scales: torch.Tensor
    discounts: torch.Tensor | None = None
    jacobian_valid: torch.Tensor | None = None


class JacobianHerReplayBuffer(ExecutedActionHerReplayBuffer):
    """Attach source-state mechanics to matching real/HER samples, not policy input."""
    _fields = ("jacobian", "joints", "action_scales")

    def add(self, obs, next_obs, action, reward, done, infos):
        size = self.action_space.shape[0]
        for info in infos:
            context = info.get("physics",{})
            for key,shape in (("jacobian",(3,size)),("joints",(size,)),("action_scales",(size,))):
                value = np.asarray(context.get(key),dtype=float)
                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"Missing or invalid source-state physics: {key}")
            if np.any(np.asarray(context["action_scales"]) <= 0):
                raise ValueError("Action scales must be positive")
            if not np.allclose(context["joints"],info["q_before"],rtol=0,atol=1e-12):
                raise ValueError("Physics context belongs to a different source state")
            if not isinstance(context.get("jacobian_valid", True), (bool, np.bool_)):
                raise ValueError("jacobian_valid must be a Boolean")
        super().add(obs,next_obs,action,reward,done,infos)

    def _attach(self, data, batch_indices, env_indices):
        obs = dict(data.observations)
        shapes = {"jacobian":(3,self.action_space.shape[0]),
                  "joints":self.action_space.shape,"action_scales":self.action_space.shape}
        for key in self._fields:
            values = np.array([self.infos[b,e]["physics"][key] for b,e in zip(batch_indices,env_indices)],dtype=np.float64)
            values = values.reshape((len(batch_indices),)+shapes[key])
            obs["__physics_"+key] = self.to_torch(values)
        obs["__physics_jacobian_valid"] = self.to_torch(np.array([
            [self.infos[b,e]["physics"].get("jacobian_valid", True)]
            for b,e in zip(batch_indices,env_indices)], dtype=bool))
        return data._replace(observations=obs)

    def _get_real_samples(self,batch_indices,env_indices,env=None):
        return self._attach(super()._get_real_samples(batch_indices,env_indices,env),batch_indices,env_indices)

    def _get_virtual_samples(self,batch_indices,env_indices,env=None):
        # Parent relabels goals/reward/done; the underlying q and J stay fixed.
        return self._attach(super()._get_virtual_samples(batch_indices,env_indices,env),batch_indices,env_indices)

    def sample(self,batch_size,env=None):
        if env is not None:
            raise ValueError("Jacobian actor loss currently requires unnormalized replay coordinates (no VecNormalize)")
        data = super().sample(batch_size,env)
        obs = dict(data.observations)
        context = {key:obs.pop("__physics_"+key) for key in self._fields}
        valid = obs.pop("__physics_jacobian_valid")
        return PhysicsSamples(obs,data.actions,data.next_observations,data.dones,data.rewards,
                              **context,jacobian_valid=valid)


class ProjectedJacobianLoss(torch.nn.Module):
    """Differentiate exact polytope projection within its selected active face.

    This accounts for extension projection and uniform joint-step limiting.
    It does not differentiate branch rejection/backtracking or certify a finite
    nonlinear decrease for the actor's current proposal.
    """
    def __init__(self,lengths,minimum_deployed=.001,cartesian_scale=.002,gain=.5):
        super().__init__()
        if not np.isfinite(cartesian_scale) or cartesian_scale<=0 or not np.isfinite(gain) or gain<=0:
            raise ValueError("Cartesian scale and gain must be positive and finite")
        constraint = JointConstraints(lengths,minimum_deployed)
        self.n = constraint.n
        self.cartesian_scale,self.gain = float(cartesian_scale),float(gain)
        for key,value in (("faces",constraint._projections),("offsets",constraint._offsets),
                          ("inequalities",constraint.A),("limits",constraint.b)):
            self.register_buffer(key,torch.as_tensor(value,dtype=torch.float64))

    def projected_delta(self, actions, joints, scales):
        # Double precision preserves narrow feasible faces; casts retain the
        # actor gradient. Source-state mechanics and scales are constants.
        q,scales = joints.detach().double(),scales.detach().double()
        desired = q+scales*actions.double()
        beta = desired[:,:self.n]
        candidates = torch.einsum("pij,bj->bpi",self.faces,beta)+self.offsets
        feasible = torch.all(candidates@self.inequalities.T <= self.limits+1e-13,dim=-1)
        if not bool(torch.all(torch.any(feasible,dim=1))):
            raise RuntimeError("No feasible actor projection face")
        distances = ((candidates-beta[:,None,:])**2).sum(-1)
        face = distances.masked_fill(~feasible,float("inf")).argmin(1)
        projected = candidates[torch.arange(len(q),device=q.device),face]
        dq = torch.cat((projected-q[:,:self.n],desired[:,self.n:]-q[:,self.n:]),dim=1)
        cap = (dq/scales).abs().amax(1,keepdim=True).clamp_min(1.)
        return dq/cap

    def forward(self,actions,sample):
        dq = self.projected_delta(actions,sample.joints,sample.action_scales)
        predicted = torch.bmm(sample.jacobian.detach().double(),dq[:,:,None]).squeeze(-1)/self.cartesian_scale
        desired = self.gain*(sample.observations["desired_goal"]-sample.observations["achieved_goal"]).detach().double()
        norm = torch.linalg.vector_norm(desired,dim=1,keepdim=True)
        desired = desired/torch.clamp(norm/self.cartesian_scale,min=1.)/self.cartesian_scale
        errors = ((predicted-desired)**2).sum(1)
        if sample.jacobian_valid is None:
            return errors.mean()
        valid = sample.jacobian_valid.detach().reshape(-1).to(errors.dtype)
        # Missing derivatives remove only their auxiliary contribution. The
        # transition and its real/HER critic update remain in replay.
        return (errors*valid).sum()/valid.sum().clamp_min(1.)


class JacobianDDPG(DDPG):
    """DDPG actor objective -Q(s,mu(s)) + weight * local kinematic tracking loss."""
    def __init__(self,*args,physics_lengths=None,physics_weight=.1,physics_final_weight=.01,
                 physics_anneal_steps=100000,**kwargs):
        if physics_lengths is None and kwargs.get("_init_setup_model",True):
            raise ValueError("Supply fixed tube lengths for the actor's joint projection")
        for value in (physics_weight,physics_final_weight):
            if not np.isfinite(value) or value<0:raise ValueError("Physics weights must be finite and nonnegative")
        if isinstance(physics_anneal_steps,bool) or not isinstance(physics_anneal_steps,int) or physics_anneal_steps<1:
            raise ValueError("physics_anneal_steps must be a positive integer")
        self.physics_lengths = list(physics_lengths) if physics_lengths is not None else []
        self.physics_weight,self.physics_final_weight = float(physics_weight),float(physics_final_weight)
        self.physics_anneal_steps = physics_anneal_steps
        self.physics_update_count = 0
        self.last_physics_metrics = {}
        self._physics_loss = None
        super().__init__(*args,**kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+["_physics_loss"]

    @property
    def current_physics_weight(self):
        fraction = min(1.,self.num_timesteps/self.physics_anneal_steps)
        return (1-fraction)*self.physics_weight+fraction*self.physics_final_weight

    def train(self,gradient_steps,batch_size=100):
        weight = self.current_physics_weight
        if weight == 0:
            # Preserve the SB3 update exactly for the zero-weight ablation.
            self.last_physics_metrics = {"weight":0.,"jacobian_loss":0.,"extra_equilibrium_calls":0}
            return super().train(gradient_steps,batch_size)
        if not isinstance(self.replay_buffer,JacobianHerReplayBuffer):
            raise ValueError("JacobianDDPG requires JacobianHerReplayBuffer")
        if self._physics_loss is None:
            self._physics_loss = ProjectedJacobianLoss(self.physics_lengths).to(self.device)
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer,self.critic.optimizer])
        actor_values,critic_values,physics_values,physics_gradients,valid_fractions = [],[],[],[],[]
        for _ in range(gradient_steps):
            self._n_updates += 1
            data = self.replay_buffer.sample(batch_size,env=self._vec_normalize_env)
            with torch.no_grad():
                # DDPG clips this noise to zero; keep SB3's RNG consumption.
                noise = data.actions.clone().normal_(0,self.target_policy_noise).clamp(-self.target_noise_clip,self.target_noise_clip)
                next_action = (self.actor_target(data.next_observations)+noise).clamp(-1,1)
                next_q = torch.cat(self.critic_target(data.next_observations,next_action),dim=1).min(1,keepdim=True).values
                target = data.rewards+(1-data.dones)*self.gamma*next_q
            critic_loss = sum(F.mse_loss(q,target) for q in self.critic(data.observations,data.actions))
            if not bool(torch.isfinite(critic_loss)):raise FloatingPointError("Nonfinite critic loss")
            self.critic.optimizer.zero_grad();critic_loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.critic.parameters()):
                raise FloatingPointError("Nonfinite critic gradient")
            self.critic.optimizer.step()
            actions = self.actor(data.observations)
            rl_loss = -self.critic.q1_forward(data.observations,actions).mean()
            physics_loss = self._physics_loss(actions,data)
            actor_loss = rl_loss+weight*physics_loss
            if not bool(torch.isfinite(actor_loss)):raise FloatingPointError("Nonfinite actor loss")
            physics_gradient = torch.autograd.grad(physics_loss,actions,retain_graph=True)[0]
            self.actor.optimizer.zero_grad();actor_loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.actor.parameters()):
                raise FloatingPointError("Nonfinite actor gradient")
            self.actor.optimizer.step()
            polyak_update(self.critic.parameters(),self.critic_target.parameters(),self.tau)
            polyak_update(self.actor.parameters(),self.actor_target.parameters(),self.tau)
            polyak_update(self.critic_batch_norm_stats,self.critic_batch_norm_stats_target,1.)
            polyak_update(self.actor_batch_norm_stats,self.actor_batch_norm_stats_target,1.)
            actor_values.append(float(rl_loss.detach()));critic_values.append(float(critic_loss.detach()))
            physics_values.append(float(physics_loss.detach()));self.physics_update_count += 1
            physics_gradients.append(float(torch.linalg.vector_norm(physics_gradient.detach())))
            valid_fractions.append(1. if data.jacobian_valid is None else float(data.jacobian_valid.float().mean()))
        self.last_physics_metrics = {"weight":weight,"jacobian_loss":float(np.mean(physics_values)),
                                     "rl_actor_loss":float(np.mean(actor_values)),
                                     "total_actor_loss":float(np.mean(actor_values)+weight*np.mean(physics_values)),
                                     "critic_loss":float(np.mean(critic_values)),
                                     "jacobian_action_gradient_norm":float(np.mean(physics_gradients)),
                                     "jacobian_valid_fraction":float(np.mean(valid_fractions)),
                                     "extra_equilibrium_calls":0}
        self.logger.record("train/n_updates",self._n_updates,exclude="tensorboard")
        self.logger.record("train/actor_loss",self.last_physics_metrics["total_actor_loss"])
        self.logger.record("train/critic_loss",float(np.mean(critic_values)))
        self.logger.record("physics/jacobian_loss",self.last_physics_metrics["jacobian_loss"])
        self.logger.record("physics/weight",weight)
