"""Compare an actor, safeguarded actor and Jacobian control on identical local goals."""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import DDPG
from ctr_reach_envs.mechanics.rl_env import EquilibriumReachEnv, GuidedRolloutWrapper
from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG
from ctr_reach_envs.mechanics.rl_metrics import EpisodeMotion, summarize_motion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",type=Path)
    p.add_argument("--episodes",type=int,default=10)
    p.add_argument("--seed",type=int,default=800000)
    p.add_argument("--modes",nargs="+",choices=["actor","safeguard","jacobian"],default=["actor","safeguard","jacobian"])
    p.add_argument("--max-steps",type=int,default=60)
    p.add_argument("--record-trajectories",action="store_true")
    p.add_argument("--output-dir",type=Path,default=Path("runs/step5_evaluation"))
    a=p.parse_args()
    if a.episodes<1 or a.max_steps<1 or a.seed<0 or len(set(a.modes))!=len(a.modes):p.error("Invalid episode, seed, step or mode selection")
    config=json.loads((a.model.parent/"config.json").read_text())
    bounds_version=config.get("observation_bounds_version",1)
    if bounds_version not in (1,2):raise ValueError("Unsupported checkpoint observation bounds version")
    algorithm=JacobianDDPG if config.get("algorithm")=="JacobianDDPG" else DDPG
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit("Choose a new, empty output directory")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    rows=[]; started=time.perf_counter(); checkpoint_timesteps=None
    with ExitStack() as stack:
        stream=stack.enter_context((a.output_dir/"episodes.csv").open("w",newline="",encoding="utf-8"))
        trajectory_stream=stack.enter_context((a.output_dir/"trajectories.csv").open("w",newline="",encoding="utf-8")) if a.record_trajectories else None
        trajectory_writer=None
        writer=None
        for mode in a.modes:
            plant=EquilibriumReachEnv(config["system"],tolerance_m=config["tolerance_m"],max_episode_steps=a.max_steps,
                                     legacy_observation_bounds=bounds_version==1)
            if plant.solver.model_fingerprint!=config["model_fingerprint"]:
                raise ValueError("Checkpoint and evaluator model parameters differ")
            env=plant if mode=="actor" else GuidedRolloutWrapper(plant)
            model=algorithm.load(a.model,env=env,device="cpu") if mode!="jacobian" else None
            if model is not None:checkpoint_timesteps=model.num_timesteps
            for episode in range(a.episodes):
                costs=dict(plant.costs);t=time.perf_counter()
                motion=EpisodeMotion(plant.n)
                row={"mode":mode,"episode":episode,"seed":a.seed+episode,"success":False,
                     "failure":"","steps":0,"error_m":None,"trivial_goal":False,
                     "initial_error_m":None,"task_fingerprint":None,
                     "replaced_actions":0,"jacobian_fallbacks":0,"numerical_decrease_checks":0,
                     "equilibrium_calls":0,"sensitivity_calls":0,"stability_calls":0,
                     "failed_calls":0,"rhs_evaluations_known":0,
                     "failed_calls_without_rhs_counts":0,"call_seconds":0.,"seconds":0.}
                try:
                    obs,info=env.reset(seed=a.seed+episode)
                    row["trivial_goal"]=info["trivial_goal"]
                    row["initial_error_m"]=info["error"]
                    task=np.r_[info["initial_q"],info["initial_torsion"],obs["desired_goal"]].astype("<f8")
                    row["task_fingerprint"]=hashlib.sha256(task.tobytes()).hexdigest()
                    initial_tip=obs["achieved_goal"].astype(float)
                    previous_tip=initial_tip.copy();path_length=0.
                    for step in range(a.max_steps):
                        # A zero proposal invokes the same Jacobian fallback with no
                        # policy solve or learned-policy contribution.
                        action=np.zeros(plant.action_space.shape,dtype=np.float32) if model is None else model.predict(obs,deterministic=True)[0]
                        obs,reward,term,trunc,info=env.step(action)
                        tip=obs["achieved_goal"].astype(float)
                        path_length+=float(np.linalg.norm(tip-previous_tip));previous_tip=tip.copy()
                        motion.add(info)
                        if trajectory_stream is not None:
                            trace={"mode":mode,"episode":episode,"seed":a.seed+episode,"step":step+1,
                                   "error_m":info["error"],"action_source":info["action_source"]}
                            for name,values in (("proposed",info["proposed_action"]),("executed",info["executed_action"]),
                                                ("delta_q",info["applied_delta"]),("q",info["q_after"]),
                                                ("tip_m",tip),("goal_m",obs["desired_goal"])):
                                trace.update({f"{name}_{i}":float(v) for i,v in enumerate(values)})
                            if trajectory_writer is None:
                                trajectory_writer=csv.DictWriter(trajectory_stream,fieldnames=list(trace));trajectory_writer.writeheader()
                            trajectory_writer.writerow(trace)
                        row["steps"]=step+1;row["error_m"]=info["error"]
                        row["success"]=info["is_success"]
                        row["replaced_actions"]+=int(info["action_replaced"])
                        row["jacobian_fallbacks"]+=int(info["action_source"].startswith("jacobian"))
                        row["numerical_decrease_checks"]+=int(info["numerical_decrease_verified"])
                        if term or trunc:break
                except Exception as exc:
                    row["failure"]=f"{type(exc).__name__}: {exc}"
                    row["success"]=False
                for key in plant.costs:
                    row[key]=plant.costs[key]-costs[key]
                row["seconds"]=time.perf_counter()-t
                row.update(motion.summary())
                row["tip_path_length_m"]=path_length if row["initial_error_m"] is not None else None
                rows.append(row)
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);stream.flush()
                print(f"{mode} {episode+1}/{a.episodes}: success={row['success']}, error={row['error_m']} m, failure={row['failure']}",flush=True)
            env.close()
    summary={"complete":True,"episodes_per_mode":a.episodes,"first_seed":a.seed,"max_steps":a.max_steps,
             "tolerance_m":config["tolerance_m"],"checkpoint":str(a.model),"checkpoint_timesteps":checkpoint_timesteps,
             "goal_distribution":config["goal_distribution"],"elapsed_seconds":time.perf_counter()-started,
             "trajectories_recorded":a.record_trajectories,
             "smoothness_semantics":"per-command differences, no physical dt; excludes episode boundaries; lower values alone do not imply better reaching",
             "modes":{}}
    for mode in a.modes:
        subset=[r for r in rows if r["mode"]==mode]
        errors=[r["error_m"] for r in subset if r["error_m"] is not None]
        summary["modes"][mode]={"successes":sum(r["success"] for r in subset),"success_rate":sum(r["success"] for r in subset)/a.episodes,
            "failures":sum(bool(r["failure"]) for r in subset),"trivial_goals":sum(r["trivial_goal"] for r in subset),
            "mean_error_m_on_completed_steps":float(np.mean(errors)) if errors else None,
            "median_final_error_m":float(np.median(errors)) if errors else None,
            "p95_final_error_m":float(np.quantile(errors,.95)) if errors else None,
            "equilibrium_calls":sum(r["equilibrium_calls"] for r in subset),
            "physics_costs_including_resets":{k:sum(r[k] for r in subset) for k in plant.costs},
            "mean_initial_error_m":float(np.mean([r["initial_error_m"] for r in subset if r["initial_error_m"] is not None])) if any(r["initial_error_m"] is not None for r in subset) else None,
            "motion_all_episodes":summarize_motion(subset),
            "motion_successful_episodes":summarize_motion([r for r in subset if r["success"]]),
            "replaced_actions":sum(r["replaced_actions"] for r in subset),
            "jacobian_fallbacks":sum(r["jacobian_fallbacks"] for r in subset),
            "steps":sum(r["steps"] for r in subset)}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({**summary,"modes":{mode:{k:v for k,v in values.items() if not k.startswith("motion_")}
                                          for mode,values in summary["modes"].items()}},indent=2))
    print(f"Motion metrics and episode records: {a.output_dir.resolve()}")
    if any(r["failure"] for r in rows):raise SystemExit(1)


if __name__=="__main__":main()
