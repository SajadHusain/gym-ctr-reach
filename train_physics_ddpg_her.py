"""DDPG/HER integration pilot on local equilibrium goals, with optional guidance.

This establishes executable training, not sample-efficiency improvement. The
model is more costly than the original baseline; begin with a short run.
"""
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
from ctr_reach_envs.mechanics.rl_env import EquilibriumReachEnv, GuidedRolloutWrapper
from ctr_reach_envs.mechanics.rl_policy import EquilibriumStateExtractor
from ctr_reach_envs.mechanics.rl_replay import ExecutedActionHerReplayBuffer
from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG, JacobianHerReplayBuffer


class AuditCallback(BaseCallback):
    def __init__(self, plant, stream, progress_every, update_stream=None):
        super().__init__()
        self.plant, self.stream, self.progress_every = plant, stream, progress_every
        self.rows = []
        self.replaced = self.decrease_verified = self.fallbacks = 0
        self.writer = None
        self.started = time.perf_counter()
        self.update_stream,self.update_writer,self.last_logged_update = update_stream,None,-1

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
             "extra_equilibrium_calls":metrics.get("extra_equilibrium_calls",0)}
        if self.update_writer is None:
            self.update_writer=csv.DictWriter(self.update_stream,fieldnames=list(row));self.update_writer.writeheader()
        self.update_writer.writerow(row);self.update_stream.flush();self.last_logged_update=count

    def _on_step(self):
        self.record_update()
        info = self.locals["infos"][0]
        self.replaced += int(info["action_replaced"])
        self.decrease_verified += int(info["numerical_decrease_verified"])
        self.fallbacks += int(info["action_source"].startswith("jacobian"))
        if self.locals["dones"][0]:
            row = {"episode": len(self.rows), "timesteps": self.num_timesteps,
                   "success": info["is_success"], "error_m": info["error"],
                   "truncated": info.get("TimeLimit.truncated",False),
                   "cumulative_equilibrium_calls": self.plant.costs["equilibrium_calls"]}
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
    p.add_argument("--total-timesteps",type=int,default=64)
    p.add_argument("--learning-starts",type=int,default=16)
    p.add_argument("--episode-steps",type=int,default=16)
    p.add_argument("--buffer-size",type=int,default=10000)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--hidden-width",type=int,default=256)
    p.add_argument("--layers",type=int,default=3)
    p.add_argument("--learning-rate",type=float,default=.0005)
    p.add_argument("--noise-std",type=float,default=.05)
    p.add_argument("--guidance",choices=["none","safeguard"],default="safeguard")
    p.add_argument("--system",choices=[f"ctr_{i}" for i in range(4)],default="ctr_0")
    p.add_argument("--tolerance-m",type=float,default=.001)
    p.add_argument("--seed",type=int,default=7005)
    p.add_argument("--progress-every",type=int,default=8)
    p.add_argument("--physics-weight",type=float,default=0.)
    p.add_argument("--physics-final-weight",type=float,default=None)
    p.add_argument("--physics-anneal-steps",type=int,default=100000)
    p.add_argument("--output-dir",type=Path,default=Path("runs/step5_smoke"))
    a=p.parse_args()
    for name in ("total_timesteps","learning_starts","episode_steps","buffer_size","batch_size","hidden_width","layers","progress_every"):
        if getattr(a,name)<1:p.error(f"{name} must be positive")
    if a.layers<2:p.error("Late-action critic requires at least two hidden layers")
    if a.learning_starts<a.episode_steps:p.error("learning-starts must cover one complete episode")
    if a.total_timesteps<=a.learning_starts:p.error("Run beyond learning-starts to test actual updates")
    if a.buffer_size<=2*a.episode_steps:p.error("buffer-size must exceed two episode lengths")
    if a.seed<0 or not np.isfinite(a.noise_std) or a.noise_std<0:p.error("Invalid seed/noise")
    if not np.isfinite(a.learning_rate) or a.learning_rate<=0:p.error("Invalid learning rate")
    if a.physics_final_weight is None:a.physics_final_weight=.1*a.physics_weight
    if any(not np.isfinite(x) or x<0 for x in (a.physics_weight,a.physics_final_weight)) or a.physics_anneal_steps<1:
        p.error("Physics weights must be finite/nonnegative and anneal steps positive")
    return a


def main():
    a=arguments()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit("Output directory is not empty; choose a new run directory")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    plant=EquilibriumReachEnv(a.system,tolerance_m=a.tolerance_m,max_episode_steps=a.episode_steps)
    env=GuidedRolloutWrapper(plant) if a.guidance=="safeguard" else plant
    physics_enabled = max(a.physics_weight,a.physics_final_weight)>0
    config={**vars(a),"output_dir":str(a.output_dir),"algorithm":"JacobianDDPG" if physics_enabled else "DDPG",
        "gamma":.95,"tau":.001,"train_freq":1,"gradient_steps":1,
        "n_sampled_goal":4,"goal_selection_strategy":"future",
        "action_semantics":"executed delta_q divided by fixed joint caps",
        "observation_semantics":"normalized egocentric joints, branch torsion, tolerance; reconstructed goal error",
        "tracking_options":asdict(plant.tracking_options),
        "observation_bounds_version":plant.observation_bounds_version,
        "torsion_observation_bound":plant.torsion_observation_bound.tolist(),
        "model_fingerprint":plant.solver.model_fingerprint,
        "goal_distribution":"seeded aligned starts and four local witness commands; fixed tolerance",
        "critic_target_policy":"unfiltered actor under goal-independent plant",
        "actor_physics_loss":physics_enabled,
        "physics_loss_semantics":"projected-current-actor Jacobian tracking, target recomputed for each HER goal",
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "torch":torch.__version__,"stable_baselines3":sb3.__version__}
    try: config["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): config["git_commit"]=None
    (a.output_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    algorithm = JacobianDDPG if physics_enabled else sb3.DDPG
    physics_kwargs = {"physics_lengths":plant.solver.lengths.tolist(),"physics_weight":a.physics_weight,
                      "physics_final_weight":a.physics_final_weight,"physics_anneal_steps":a.physics_anneal_steps} if physics_enabled else {}
    model=algorithm(PaperMlpPolicy,env,seed=a.seed,device="cpu",**physics_kwargs,
        learning_rate=a.learning_rate,gamma=.95,tau=.001,
        learning_starts=a.learning_starts,buffer_size=a.buffer_size,batch_size=a.batch_size,
        train_freq=1,gradient_steps=1,
        action_noise=NormalActionNoise(np.zeros(plant.n*2),np.full(plant.n*2,a.noise_std)),
        policy_kwargs={"net_arch":[a.hidden_width]*a.layers,
                       "features_extractor_class":EquilibriumStateExtractor,
                       "features_extractor_kwargs":{"length_scale":plant.length}},
        replay_buffer_class=JacobianHerReplayBuffer if physics_enabled else ExecutedActionHerReplayBuffer,
        replay_buffer_kwargs={"copy_info_dict":True,"n_sampled_goal":4,"goal_selection_strategy":"future"},verbose=0)
    actor=[p.detach().clone() for p in model.actor.parameters()]
    critic=[p.detach().clone() for p in model.critic.parameters()]
    started=time.perf_counter(); failure=None; complete=False
    with (a.output_dir/"episodes.csv").open("w",newline="",encoding="utf-8") as f, (a.output_dir/"updates.csv").open("w",newline="",encoding="utf-8") as updates:
        callback=AuditCallback(plant,f,a.progress_every,updates)
        try:
            model.learn(a.total_timesteps,callback=callback)
            complete=True
        except (KeyboardInterrupt,Exception) as exc:
            failure=f"{type(exc).__name__}: {exc}"
        callback.record_update()
        model.save(a.output_dir/("final_model.zip" if complete else "interrupted_model.zip"))
        buffer=model.replay_buffer
        indices=range(buffer.buffer_size if buffer.full else buffer.pos)
        replay_errors=[float(np.max(abs(buffer.actions[i,0]-buffer.infos[i,0]["executed_action"]))) for i in indices]
        summary={"complete":complete,"failure":failure,"timesteps":model.num_timesteps,
            "gradient_updates":model._n_updates,"completed_episodes":len(callback.rows),
            "rollout_successes":sum(r["success"] for r in callback.rows),
            "actor_parameters_changed":any(not torch.equal(x,p) for x,p in zip(actor,model.actor.parameters())),
            "critic_parameters_changed":any(not torch.equal(x,p) for x,p in zip(critic,model.critic.parameters())),
            "all_parameters_finite":all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()),
            "replay_entries":len(replay_errors),"maximum_executed_action_replay_error":max(replay_errors,default=None),
            "replaced_actions":callback.replaced,"jacobian_fallbacks":callback.fallbacks,
            "decrease_verified_transitions":callback.decrease_verified,
            "physics_actor_updates":getattr(model,"physics_update_count",0),
            "last_physics_metrics":getattr(model,"last_physics_metrics",{}),
            "reset_attempts":plant.reset_attempts,"failed_resets":plant.failed_resets,
            "physics_costs_including_resets_and_witnesses":plant.costs,
            "elapsed_seconds":time.perf_counter()-started,
            "scope":"local-goal pilot; collection safeguard and actor Jacobian loss are separate options",
            "global_convergence_certified":False,"sample_efficiency_improvement_demonstrated":False}
        (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    env.close()
    print(json.dumps(summary,indent=2))
    print(f"Results: {a.output_dir.resolve()}")
    if not complete or not summary["all_parameters_finite"]:
        raise SystemExit(1)


if __name__=="__main__":main()
