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


class AuditCallback(BaseCallback):
    def __init__(self, plant, stream, progress_every):
        super().__init__()
        self.plant, self.stream, self.progress_every = plant, stream, progress_every
        self.rows = []
        self.replaced = self.decrease_verified = self.fallbacks = 0
        self.writer = None
        self.started = time.perf_counter()

    def _on_step(self):
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
    return a


def main():
    a=arguments()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit("Output directory is not empty; choose a new run directory")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    plant=EquilibriumReachEnv(a.system,tolerance_m=a.tolerance_m,max_episode_steps=a.episode_steps)
    env=GuidedRolloutWrapper(plant) if a.guidance=="safeguard" else plant
    config={**vars(a),"output_dir":str(a.output_dir),"algorithm":"DDPG",
        "gamma":.95,"tau":.001,"train_freq":1,"gradient_steps":1,
        "n_sampled_goal":4,"goal_selection_strategy":"future",
        "action_semantics":"executed delta_q divided by fixed joint caps",
        "observation_semantics":"normalized egocentric joints, branch torsion, tolerance; reconstructed goal error",
        "tracking_options":asdict(plant.tracking_options),
        "model_fingerprint":plant.solver.model_fingerprint,
        "goal_distribution":"seeded aligned starts and four local witness commands; fixed tolerance",
        "critic_target_policy":"unfiltered actor under goal-independent plant",
        "actor_physics_loss":False,
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "torch":torch.__version__,"stable_baselines3":sb3.__version__}
    try: config["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): config["git_commit"]=None
    (a.output_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    model=sb3.DDPG(PaperMlpPolicy,env,seed=a.seed,device="cpu",
        learning_rate=a.learning_rate,gamma=.95,tau=.001,
        learning_starts=a.learning_starts,buffer_size=a.buffer_size,batch_size=a.batch_size,
        train_freq=1,gradient_steps=1,
        action_noise=NormalActionNoise(np.zeros(plant.n*2),np.full(plant.n*2,a.noise_std)),
        policy_kwargs={"net_arch":[a.hidden_width]*a.layers,
                       "features_extractor_class":EquilibriumStateExtractor,
                       "features_extractor_kwargs":{"length_scale":plant.length}},
        replay_buffer_class=ExecutedActionHerReplayBuffer,
        replay_buffer_kwargs={"copy_info_dict":True,"n_sampled_goal":4,"goal_selection_strategy":"future"},verbose=0)
    actor=[p.detach().clone() for p in model.actor.parameters()]
    critic=[p.detach().clone() for p in model.critic.parameters()]
    started=time.perf_counter(); failure=None; complete=False
    with (a.output_dir/"episodes.csv").open("w",newline="",encoding="utf-8") as f:
        callback=AuditCallback(plant,f,a.progress_every)
        try:
            model.learn(a.total_timesteps,callback=callback)
            complete=True
        except (KeyboardInterrupt,Exception) as exc:
            failure=f"{type(exc).__name__}: {exc}"
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
            "reset_attempts":plant.reset_attempts,"failed_resets":plant.failed_resets,
            "physics_costs_including_resets_and_witnesses":plant.costs,
            "elapsed_seconds":time.perf_counter()-started,
            "scope":"integration pilot; behavior is safeguarded only when configured",
            "global_convergence_certified":False,"sample_efficiency_improvement_demonstrated":False}
        (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    env.close()
    print(json.dumps(summary,indent=2))
    print(f"Results: {a.output_dir.resolve()}")
    if not complete or not summary["all_parameters_finite"]:
        raise SystemExit(1)


if __name__=="__main__":main()
