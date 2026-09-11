"""DDPG + HER with joint constraints and an optional Jacobian actor loss."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from ctr_reach_envs.paper_policy import PaperMlpPolicy
from ctr_reach_envs.mechanics.simple_rl_env import JointConstrainedReachEnv, TASK_PROFILES
from ctr_reach_envs.mechanics.rl_policy import EquilibriumStateExtractor
from ctr_reach_envs.mechanics.rl_replay import ExecutedActionHerReplayBuffer
from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG, JacobianHerReplayBuffer
from ctr_reach_envs.mechanics.rl_exploration import ExplorationDDPG, exploration_settings
from ctr_reach_envs.mechanics.rl_metrics import EpisodeMotion, ReachHoldMetrics


class AuditCallback(BaseCallback):
    def __init__(self, plant, stream, progress_every, update_stream=None,
                 checkpoint_freq=0, output_dir=None, config=None):
        super().__init__()
        self.plant, self.stream, self.progress_every = plant, stream, progress_every
        self.rows = []
        self.replaced = self.decrease_verified = self.fallbacks = 0
        self.writer = None
        self.started = time.perf_counter()
        self.update_stream,self.update_writer,self.last_logged_update = update_stream,None,-1
        self.motion=EpisodeMotion(plant.n)
        self.hold_steps = (config or {}).get("hold_steps", 10)
        self.reaching = ReachHoldMetrics(plant.tolerance_m, self.hold_steps)
        self.checkpoint_freq,self.output_dir,self.config=checkpoint_freq,output_dir,config
        self.saved_steps=set()

    def save_checkpoint(self, force=False):
        step=self.model.num_timesteps
        if not self.checkpoint_freq or step in self.saved_steps:
            return
        if not force and step % self.checkpoint_freq:
            return
        folder=self.output_dir/"checkpoints"/f"step_{step:09d}"
        folder.mkdir(parents=True,exist_ok=False)
        self.model.save(folder/"model.zip")
        (folder/"config.json").write_text(json.dumps(self.config,indent=2)+"\n",encoding="utf-8")
        budget={"timesteps":step,"gradient_updates":self.model._n_updates,
                "exploration_proposal_counts":dict(self.model.exploration_counts),
                "training_seconds":time.perf_counter()-self.started,
                "physics_costs_including_resets_and_witnesses":dict(self.plant.costs),
                "replaced_actions":self.replaced,"jacobian_fallbacks":self.fallbacks,
                "all_parameters_finite":all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters())}
        (folder/"budget.json").write_text(json.dumps(budget,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        self.saved_steps.add(step)

    def _on_rollout_start(self):
        # SB3 has finished the previous optimizer update here. The final step
        # is saved after learn(), so all arms include identical update counts.
        self.save_checkpoint()

    def record_update(self):
        if not hasattr(self,"model"):return
        metrics = getattr(self.model,"last_physics_metrics",{})
        count = self.model._n_updates
        if count<1 or count==self.last_logged_update or self.update_stream is None:return
        logged=self.model.logger.name_to_value
        row={"timesteps":self.model.num_timesteps,"gradient_updates":count,
             "weight":metrics.get("weight",0.),"jacobian_loss":metrics.get("jacobian_loss",0.),
             "rl_actor_loss":metrics.get("rl_actor_loss",logged.get("train/actor_loss")),
             "total_actor_loss":metrics.get("total_actor_loss",logged.get("train/actor_loss")),
             "critic_loss":metrics.get("critic_loss",logged.get("train/critic_loss")),
             "jacobian_action_gradient_norm":metrics.get("jacobian_action_gradient_norm",0.),
             "jacobian_valid_fraction":metrics.get("jacobian_valid_fraction",0.),
             "extra_equilibrium_calls":metrics.get("extra_equilibrium_calls",0)}
        if self.update_writer is None:
            self.update_writer=csv.DictWriter(self.update_stream,fieldnames=list(row));self.update_writer.writeheader()
        self.update_writer.writerow(row);self.update_stream.flush();self.last_logged_update=count

    def _on_step(self):
        self.record_update()
        info = self.locals["infos"][0]
        # Only actions are used here; terminal-observation resets cannot leak
        # into action-difference statistics.
        self.motion.add(info)
        self.reaching.add(info["error"])
        self.replaced += int(info["action_replaced"])
        self.decrease_verified += int(info["numerical_decrease_verified"])
        self.fallbacks += int(info["action_source"].startswith("jacobian"))
        if self.locals["dones"][0]:
            row = {"episode": len(self.rows), "timesteps": self.num_timesteps,
                   "success": info["is_success"], "error_m": info["error"],
                   "truncated": info.get("TimeLimit.truncated",False),
                   "cumulative_equilibrium_calls": self.plant.costs["equilibrium_calls"]}
            row.update(self.motion.summary())
            row.update(self.reaching.summary())
            if not self.plant.terminate_on_success:
                row["success"] = row["sustained_success"]
            self.motion=EpisodeMotion(self.plant.n)
            self.reaching=ReachHoldMetrics(self.plant.tolerance_m, self.hold_steps)
            self.rows.append(row)
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row)); self.writer.writeheader()
            self.writer.writerow(row); self.stream.flush()
        if self.num_timesteps % self.progress_every == 0:
            print(f"Step {self.num_timesteps}: completed episodes={len(self.rows)}, "
                  f"error={info['error']*1000:.3f} mm, "
                  f"equilibrium calls={self.plant.costs['equilibrium_calls']}, "
                  f"elapsed={time.perf_counter()-self.started:.0f} s", flush=True)
        return True


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-timesteps",type=int,default=10000)
    p.add_argument("--learning-starts",type=int,default=200)
    p.add_argument("--episode-steps",type=int,default=60)
    p.add_argument("--buffer-size",type=int,default=20000)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--hidden-width",type=int,default=256)
    p.add_argument("--layers",type=int,default=3)
    p.add_argument("--learning-rate",type=float,default=.0005)
    p.add_argument("--exploration-profile",choices=["paper","gaussian"],default="paper",
                   help="paper: the earlier DDPG+HER noise and uniform-action mixture; gaussian: previous simple-run exploration")
    p.add_argument("--noise-std",type=float,default=None,
                   help="Override the profile's normalized Gaussian standard deviation on all joints")
    p.add_argument("--random-exploration",type=float,default=None,
                   help="Override the probability of a uniform random action after warmup")
    p.add_argument("--guidance",choices=["none"],default="none",
                   help="Compatibility option; this trainer uses the actor without a controller wrapper")
    p.add_argument("--system",choices=[f"ctr_{i}" for i in range(4)],default="ctr_0")
    p.add_argument("--tolerance-m",type=float,default=.001)
    p.add_argument("--task-profile", choices=TASK_PROFILES, default="generalized_hold",
                   help="generalized_hold: signed reachable goals and no success termination; legacy: reproduce the old task")
    p.add_argument("--hold-steps", type=int, default=10,
                   help="Report sustained success over the final K steps; does not change rewards or terminate an episode")
    p.add_argument("--initial-rotation-span-rad", type=float, default=.15,
                   help="Independent tube-angle offsets about a common angle sampled uniformly from [-pi, pi]")
    p.add_argument("--goal-steps-min", type=int, default=2)
    p.add_argument("--goal-steps-max", type=int, default=8)
    p.add_argument("--seed",type=int,default=7101)
    p.add_argument("--progress-every",type=int,default=100)
    p.add_argument("--physics-weight",type=float,default=.1)
    p.add_argument("--physics-final-weight",type=float,default=None)
    p.add_argument("--physics-anneal-steps",type=int,default=100000)
    p.add_argument("--checkpoint-freq",type=int,default=0)
    p.add_argument("--output-dir",type=Path,default=Path("runs/simple_jacobian_seed7101"))
    a=p.parse_args()
    for name in ("total_timesteps","learning_starts","episode_steps","buffer_size","batch_size","hidden_width","layers","progress_every"):
        if getattr(a,name)<1:p.error(f"{name} must be positive")
    if a.layers<2:p.error("Late-action critic requires at least two hidden layers")
    if a.checkpoint_freq<0:p.error("checkpoint-freq cannot be negative")
    if a.learning_starts<a.episode_steps:p.error("learning-starts must cover one complete episode")
    if a.total_timesteps<=a.learning_starts:p.error("Run beyond learning-starts to test actual updates")
    if a.buffer_size<=2*a.episode_steps:p.error("buffer-size must exceed two episode lengths")
    if a.seed<0:p.error("Invalid seed")
    if a.hold_steps < 1 or (a.task_profile == "generalized_hold" and a.hold_steps > a.episode_steps):
        p.error("Holding window must be positive and fit the generalized episode")
    if not np.isfinite(a.initial_rotation_span_rad) or a.initial_rotation_span_rad < 0:
        p.error("Initial rotation span must be finite and nonnegative")
    if not 1 <= a.goal_steps_min <= a.goal_steps_max:
        p.error("Need 1 <= goal-steps-min <= goal-steps-max")
    try: exploration_settings(a.exploration_profile,a.noise_std,a.random_exploration)
    except ValueError as exc: p.error(str(exc))
    if not np.isfinite(a.learning_rate) or a.learning_rate<=0:p.error("Invalid learning rate")
    if a.physics_final_weight is None:a.physics_final_weight=a.physics_weight
    if any(not np.isfinite(x) or x<0 for x in (a.physics_weight,a.physics_final_weight)) or a.physics_anneal_steps<1:
        p.error("Physics weights must be finite/nonnegative and anneal steps positive")
    return a


def main():
    a=arguments()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit("Output directory is not empty; choose a new run directory")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    physics_enabled = max(a.physics_weight,a.physics_final_weight)>0
    plant=JointConstrainedReachEnv(a.system,tolerance_m=a.tolerance_m,
        max_episode_steps=a.episode_steps,compute_jacobian=physics_enabled,
        task_profile=a.task_profile, initial_rotation_span_rad=a.initial_rotation_span_rad,
        goal_steps_min=a.goal_steps_min, goal_steps_max=a.goal_steps_max)
    env=plant
    replay_action_semantics = "proposal"
    exploration = exploration_settings(a.exploration_profile,a.noise_std,a.random_exploration,plant.n*2)
    config={**vars(a),"output_dir":str(a.output_dir),"algorithm":"JacobianDDPG" if physics_enabled else "DDPG",
        "gamma":.95,"tau":.001,"train_freq":1,"gradient_steps":1,
        "n_sampled_goal":4,"goal_selection_strategy":"future",
        "replay_action_semantics":replay_action_semantics,
        "plant":plant.plant_version,"action_contract_version":3,
        "action_semantics":"normalized proposal; joint projection and fixed joint increment caps",
        "action_scales":plant.action_scales.tolist(),
        "observation_semantics":"normalized egocentric joints and tolerance; reconstructed goal error",
        "equilibrium_initialization":"deterministic default; no previous torsion",
        "solver_options":asdict(plant.solver.options),"integrator":"scale_safe_dop853_v1",
        "observation_bounds_version":plant.observation_bounds_version,
        "model_fingerprint":plant.solver.model_fingerprint,
        "goal_distribution":plant.goal_distribution,"task_settings":plant.task_settings,
        "task_semantics_version":1,
        "success_definition":"first_hit" if plant.terminate_on_success else "final_hold_window",
        "reward_semantics":"0 within tolerance, -1 outside, recomputed for every real/HER transition",
        "episode_semantics":"terminate on success" if plant.terminate_on_success else
            "continuing reaching-and-holding task; fixed collection horizon truncates and bootstraps, including at success",
        "critic_target_policy":"raw actor proposal to the joint-constrained plant",
        "actor_physics_loss":physics_enabled,
        "jacobian_method":"Burgner 2014 Eqs. (6)-(7); analytical variational ODE, unloaded specialization, spatial-to-tip conversion",
        "exploration":exploration,
        "physics_loss_semantics":"projected-current-actor Jacobian tracking, target recomputed for each HER goal",
        "unavailable_jacobian":"mask auxiliary sample; keep transition and RL update",
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "torch":torch.__version__,"stable_baselines3":sb3.__version__}
    try: config["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): config["git_commit"]=None
    (a.output_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    algorithm = JacobianDDPG if physics_enabled else ExplorationDDPG
    physics_kwargs = {"physics_lengths":plant.solver.lengths.tolist(),"physics_weight":a.physics_weight,
                      "physics_final_weight":a.physics_final_weight,"physics_anneal_steps":a.physics_anneal_steps} if physics_enabled else {}
    model=algorithm(PaperMlpPolicy,env,seed=a.seed,device="cpu",**physics_kwargs,
        random_exploration=exploration["random_exploration"],
        learning_rate=a.learning_rate,gamma=.95,tau=.001,
        learning_starts=a.learning_starts,buffer_size=a.buffer_size,batch_size=a.batch_size,
        train_freq=1,gradient_steps=1,
        action_noise=NormalActionNoise(np.zeros(plant.n*2),np.asarray(exploration["normalized_action_noise_std"])),
        policy_kwargs={"net_arch":[a.hidden_width]*a.layers,
                       "features_extractor_class":EquilibriumStateExtractor,
                       "features_extractor_kwargs":{"length_scale":plant.length}},
        replay_buffer_class=JacobianHerReplayBuffer if physics_enabled else ExecutedActionHerReplayBuffer,
        replay_buffer_kwargs={"copy_info_dict":True,"n_sampled_goal":4,"goal_selection_strategy":"future",
                              "action_semantics":replay_action_semantics},verbose=0)
    actor=[p.detach().clone() for p in model.actor.parameters()]
    critic=[p.detach().clone() for p in model.critic.parameters()]
    started=time.perf_counter(); failure=None; complete=False
    with (a.output_dir/"episodes.csv").open("w",newline="",encoding="utf-8") as f, (a.output_dir/"updates.csv").open("w",newline="",encoding="utf-8") as updates:
        callback=AuditCallback(plant,f,a.progress_every,updates,a.checkpoint_freq,a.output_dir,config)
        try:
            model.learn(a.total_timesteps,callback=callback)
            complete=True
        except (KeyboardInterrupt,Exception) as exc:
            failure=f"{type(exc).__name__}: {exc}"
        callback.record_update()
        if complete:callback.save_checkpoint(force=True)
        model.save(a.output_dir/("final_model.zip" if complete else "interrupted_model.zip"))
        buffer=model.replay_buffer
        indices=range(buffer.buffer_size if buffer.full else buffer.pos)
        action_key="proposed_action" if replay_action_semantics=="proposal" else "executed_action"
        replay_errors=[float(np.max(abs(buffer.actions[i,0]-buffer.infos[i,0][action_key]))) for i in indices]
        execution_differences=[float(np.max(abs(buffer.actions[i,0]-buffer.infos[i,0]["executed_action"]))) for i in indices]
        summary={"complete":complete,"failure":failure,"timesteps":model.num_timesteps,
            "exploration":exploration,"exploration_proposal_counts":dict(model.exploration_counts),
            "gradient_updates":model._n_updates,"completed_episodes":len(callback.rows),
            "rollout_successes":sum(r["success"] for r in callback.rows),
            "rollout_reaches":sum(r["reached"] for r in callback.rows),
            "rollout_sustained_successes":sum(r["sustained_success"] for r in callback.rows),
            "task_profile":a.task_profile,"success_definition":config["success_definition"],
            "actor_parameters_changed":any(not torch.equal(x,p) for x,p in zip(actor,model.actor.parameters())),
            "critic_parameters_changed":any(not torch.equal(x,p) for x,p in zip(critic,model.critic.parameters())),
            "all_parameters_finite":all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()),
            "replay_entries":len(replay_errors),"replay_action_semantics":replay_action_semantics,
            "maximum_training_action_replay_error":max(replay_errors,default=None),
            "maximum_replay_vs_executed_action_difference":max(execution_differences,default=None),
            "replaced_actions":callback.replaced,"jacobian_fallbacks":callback.fallbacks,
            "decrease_verified_transitions":callback.decrease_verified,
            "physics_actor_updates":getattr(model,"physics_update_count",0),
            "last_physics_metrics":getattr(model,"last_physics_metrics",{}),
            "reset_attempts":plant.reset_attempts,"failed_resets":plant.failed_resets,
            "rejected_trivial_goals":plant.rejected_trivial_goals,
            "last_failed_solve_q":plant.last_failed_solve_q,
            "physics_costs_including_resets_and_witnesses":plant.costs,
            "elapsed_seconds":time.perf_counter()-started,
            "plant":plant.plant_version,"jacobian_failures":plant.jacobian_failures,
            "scope":plant.goal_distribution+"; optional Jacobian actor loss; no holding or stability guarantee",
            "global_convergence_certified":False,"sample_efficiency_improvement_demonstrated":False}
        (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    env.close()
    print(json.dumps(summary,indent=2))
    print(f"Results: {a.output_dir.resolve()}")
    if not complete or not summary["all_parameters_finite"]:
        raise SystemExit(1)


if __name__=="__main__":main()
