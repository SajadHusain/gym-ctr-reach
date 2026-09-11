"""Matched CTR experiments with frozen budgets, held-out tasks and seed-level reports.

Create a plan first. Subsequent stages read it, so commands need only its folder.
Completed jobs are skipped; partial/failed training is never silently restarted.
"""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from ctr_reach_envs.mechanics.rl_exploration import exploration_settings
from ctr_reach_envs.mechanics.simple_rl_env import TASK_PROFILES

ROOT = Path(__file__).resolve().parent
ARMS = {
    "ddpg": {"guidance": "none", "physics_weight": 0.},
    "jacobian": {"guidance": "none", "physics_weight": .1},
}
DEFAULTS = dict(seeds=[7100,7101,7102,7103,7104], arms=["ddpg","jacobian"],
                total_timesteps=10000, learning_starts=200, episode_steps=60,
                checkpoint_freq=500, buffer_size=20000, batch_size=128,
                hidden_width=256, layers=3, learning_rate=.0005, noise_std=None,
                exploration_profile="paper", random_exploration=None,
                task_profile="generalized_hold", hold_steps=10,
                initial_rotation_span_rad=.15, goal_steps_min=2, goal_steps_max=8,
                max_shooting_evaluations=500,
                physics_weight=.1, physics_final_weight=.1, physics_anneal_steps=10000,
                system="ctr_0", tolerance_m=.001, eval_episodes=20, eval_steps=60,
                eval_seed=810000, final_seed=910000, final_episodes=1000)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_hashes():
    files = list((ROOT/"ctr_reach_envs"/"mechanics").glob("*.py"))
    files += [ROOT/name for name in ("train_physics_ddpg_her.py", "evaluate_physics_ddpg_her.py",
             "run_physics_study.py", "ctr_reach_envs/paper_policy.py", "ctr_reach_envs/her_replay_buffer.py",
             "ctr_reach_envs/config.py", "ctr_reach_envs/paper_config.py")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def runtime_versions():
    return {"python":platform.python_version(), **{name:version(name) for name in
            ("numpy","scipy","torch","gymnasium","stable-baselines3")}}


def validate_config(c):
    if len(set(c["seeds"])) != len(c["seeds"]) or not c["seeds"] or min(c["seeds"]) < 0:
        raise ValueError("Training seeds must be unique, nonnegative and nonempty")
    if len(set(c["arms"])) != len(c["arms"]) or not c["arms"] or any(x not in ARMS for x in c["arms"]):
        raise ValueError("Unknown or duplicate arm")
    positive = ("total_timesteps", "learning_starts", "episode_steps", "checkpoint_freq", "buffer_size",
                "batch_size", "hidden_width", "layers", "physics_anneal_steps", "eval_episodes", "eval_steps", "final_episodes")
    if any(c[k] < 1 for k in positive):
        raise ValueError("Budgets and network dimensions must be positive")
    if c["layers"] < 2 or not c["total_timesteps"] > c["learning_starts"] >= c["episode_steps"]:
        raise ValueError("Need layers >= 2 and total_timesteps > learning_starts >= episode_steps")
    if c["buffer_size"] <= 2*c["episode_steps"]:
        raise ValueError("Buffer must exceed two episode lengths")
    for key in ("learning_rate", "tolerance_m", "physics_weight"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("physics_final_weight",):
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"Invalid {key}")
    exploration_settings(c.get("exploration_profile","gaussian"),c.get("noise_std"),c.get("random_exploration"))
    if c.get("task_profile", "legacy") not in TASK_PROFILES:
        raise ValueError("Unknown task profile")
    if c.get("hold_steps", 10) < 1:
        raise ValueError("Holding window must be positive")
    if c.get("task_profile", "legacy") == "generalized_hold" and c["hold_steps"] > min(c["episode_steps"],c["eval_steps"]):
        raise ValueError("Holding window must fit training and evaluation horizons")
    if not np.isfinite(c.get("initial_rotation_span_rad",.15)) or c.get("initial_rotation_span_rad",.15) < 0:
        raise ValueError("Initial rotation span must be finite and nonnegative")
    if not 1 <= c.get("goal_steps_min",2) <= c.get("goal_steps_max",8):
        raise ValueError("Invalid goal step bounds")
    if c.get("max_shooting_evaluations",500) < 1:
        raise ValueError("Shooting evaluation budget must be positive")
    if c["eval_seed"] < 0 or c["final_seed"] < 0:
        raise ValueError("Evaluation seeds must be nonnegative")
    dev = set(range(c["eval_seed"], c["eval_seed"]+c["eval_episodes"]))
    final = set(range(c["final_seed"], c["final_seed"]+c["final_episodes"]))
    if dev & final or (dev | final) & set(c["seeds"]):
        raise ValueError("Training, development and final evaluation seed blocks must be distinct")


def checkpoints(c):
    return sorted({0, c["total_timesteps"], *range(c["checkpoint_freq"], c["total_timesteps"]+1, c["checkpoint_freq"])})


def command(script, values):
    args = [sys.executable, str(ROOT/script)]
    for key, value in values.items():
        args += ["--"+key.replace("_", "-")]
        args += [str(v) for v in value] if isinstance(value, (tuple,list)) else [str(value)]
    return args


def run_command(args, result_dir):
    print("Running:", subprocess.list2cmdline(args), flush=True)
    result = subprocess.run(args, cwd=ROOT)
    summary = result_dir/"summary.json"
    if not summary.exists() or not read_json(summary).get("complete"):
        print(f"INCOMPLETE: {result_dir}. Other planned jobs will still be attempted.", flush=True)
        return False
    return result.returncode == 0


def train(plan, folder):
    c = plan["config"]
    ok = True
    for index, seed in enumerate(c["seeds"]):
        # Rotate arm order to reduce a consistent warm-cache/temperature bias.
        arms = c["arms"][index % len(c["arms"]):]+c["arms"][:index % len(c["arms"])]
        for arm in arms:
            out = folder/arm/f"seed_{seed}"
            if out.exists() and any(out.iterdir()):
                print(f"Existing training retained: {out}", flush=True)
                ok = ok and (out/"summary.json").exists() and read_json(out/"summary.json").get("complete",False)
                continue
            params = {k:c[k] for k in ("total_timesteps", "learning_starts", "episode_steps", "checkpoint_freq",
                      "buffer_size", "batch_size", "hidden_width", "layers", "learning_rate",
                      "system", "tolerance_m", "physics_anneal_steps")}
            params["exploration_profile"] = c.get("exploration_profile","gaussian")
            params["task_profile"] = c.get("task_profile", "legacy")
            for key in ("hold_steps","initial_rotation_span_rad","goal_steps_min","goal_steps_max",
                        "max_shooting_evaluations"):
                if key in c: params[key] = c[key]
            for key in ("noise_std","random_exploration"):
                if c.get(key) is not None: params[key] = c[key]
            params.update(seed=seed, guidance=ARMS[arm]["guidance"],
                          physics_weight=c["physics_weight"] if ARMS[arm]["physics_weight"] else 0.,
                          physics_final_weight=c["physics_final_weight"] if ARMS[arm]["physics_weight"] else 0.,
                          progress_every=min(100,c["total_timesteps"]), output_dir=out)
            ok = run_command(command("train_physics_ddpg_her.py", params), out) and ok
    return ok


def evaluate(plan, folder, final=False):
    c = plan["config"]
    steps = [c["total_timesteps"]] if final else checkpoints(c)
    label = "final_evaluation" if final else "evaluation"
    for arm in c["arms"]:
        for seed in c["seeds"]:
            for step in steps:
                checkpoint = folder/arm/f"seed_{seed}"/"checkpoints"/f"step_{step:09d}"
                out = checkpoint/label
                if not (checkpoint/"model.zip").exists():
                    print(f"Missing checkpoint: {checkpoint}", flush=True)
                    continue
                if out.exists() and any(out.iterdir()):
                    print(f"Existing evaluation retained: {out}", flush=True)
                    continue
                args = command("evaluate_physics_ddpg_her.py", dict(
                    episodes=c["final_episodes"] if final else c["eval_episodes"],
                    max_steps=c["eval_steps"], seed=c["final_seed"] if final else c["eval_seed"],
                    modes=["actor"], hold_steps=c.get("hold_steps",10), output_dir=out))
                args.insert(2,str(checkpoint/"model.zip"))
                args.append("--record-trajectories")
                run_command(args,out)


def paired_seed_statistics(values):
    values = np.asarray(values,dtype=float)
    if not len(values):
        return {"seeds":0,"mean_difference":None,"seed_std":None,"bootstrap_95":None}
    interval = None
    if len(values) >= 2:
        rng = np.random.default_rng(202607)
        means = rng.choice(values, size=(10000,len(values)), replace=True).mean(axis=1)
        interval = np.quantile(means,[.025,.975]).tolist()
    return {"seeds":len(values),"mean_difference":float(values.mean()),
            "seed_std":float(values.std(ddof=1)) if len(values)>1 else None,"bootstrap_95":interval}


def report(plan, folder, final=False):
    c = plan["config"]; rows=[]; missing=[]; datasets={};tasks={}
    steps = [c["total_timesteps"]] if final else checkpoints(c)
    for arm in c["arms"]:
        for seed in c["seeds"]:
            for step in steps:
                checkpoint=folder/arm/f"seed_{seed}"/"checkpoints"/f"step_{step:09d}"
                out=checkpoint/("final_evaluation" if final else "evaluation")
                if not all((out/f).exists() for f in ("summary.json","episodes.csv")) or not (checkpoint/"budget.json").exists():
                    missing.append(f"{arm}/{seed}/{step}");continue
                result=read_json(out/"summary.json");budget=read_json(checkpoint/"budget.json")
                expected=c["final_episodes"] if final else c["eval_episodes"]
                expected_seed=c["final_seed"] if final else c["eval_seed"]
                if (not result.get("complete") or result["episodes_per_mode"]!=expected or result["checkpoint_timesteps"]!=step
                        or result["first_seed"]!=expected_seed or result["max_steps"]!=c["eval_steps"]
                        or result["tolerance_m"]!=c["tolerance_m"]):
                    raise ValueError(f"Mismatched evaluation: {out}")
                if result.get("task_profile","legacy") != c.get("task_profile","legacy"):
                    raise ValueError(f"Mismatched evaluation task profile: {out}")
                if c.get("task_profile","legacy") == "generalized_hold" and (
                        result.get("success_definition") != "final_hold_window" or result.get("hold_steps") != c["hold_steps"]):
                    raise ValueError(f"Mismatched sustained-success definition: {out}")
                with (out/"episodes.csv").open(newline="",encoding="utf-8") as stream:
                    episode_rows=list(csv.DictReader(stream))
                for mode, values in result["modes"].items():
                    episodes=[e for e in episode_rows if e["mode"]==mode]
                    if len(episodes)!=expected:
                        raise ValueError(f"Wrong episode count: {out}")
                    if {int(e["seed"]) for e in episodes}!=set(range(expected_seed,expected_seed+expected)):
                        raise ValueError(f"Wrong or duplicate task seeds: {out}")
                    for e in episodes:
                        task_key=int(e["seed"])
                        fp=e["task_fingerprint"]
                        if fp and task_key in tasks and fp!=tasks[task_key]:
                            raise ValueError(f"Evaluation tasks differ at seed {task_key}")
                        if fp:tasks[task_key]=fp
                    datasets[(arm,seed,step,mode)]=episodes
                    row={"arm":arm,"training_seed":seed,"timesteps":step,"mode":mode,
                         "success_rate":values["success_rate"],"final_error_m":values["mean_error_m_on_completed_steps"],
                         "reaching_success_rate":values.get("reaching_success_rate",values["success_rate"]),
                         "sustained_success_rate":values.get("sustained_success_rate"),
                         "hold_window_max_error_m":values.get("mean_hold_window_max_error_m"),
                         "evaluation_failures":values["failures"],"gradient_updates":budget["gradient_updates"],
                         "training_seconds":budget["training_seconds"],
                         "training_equilibrium_calls":budget["physics_costs_including_resets_and_witnesses"]["equilibrium_calls"],
                         "evaluation_equilibrium_calls":values["equilibrium_calls"],
                         "steps":values["steps"],"replaced_actions":values["replaced_actions"],
                         "jacobian_fallbacks":values["jacobian_fallbacks"]}
                    for key,value in budget["physics_costs_including_resets_and_witnesses"].items():
                        row["training_"+key]=value
                    for key,value in values.get("physics_costs_including_resets",{}).items():
                        row["evaluation_"+key]=value
                    for key in ("proposed_action_change_rms","executed_action_change_rms","executed_action_second_difference_rms",
                                "translation_increment_change_rms_m","rotation_increment_change_rms_rad","held_action_fraction"):
                        row[key]=values["motion_all_episodes"][key]["mean"]
                    rows.append(row)
    prefix="final" if final else "development"
    if rows:
        with (folder/f"{prefix}_learning_curves.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    contrasts={}
    for base, guided in (("ddpg","jacobian"),):
        if base not in c["arms"] or guided not in c["arms"]:continue
        by_key={(r["arm"],r["training_seed"],r["timesteps"],r["mode"]):r for r in rows}
        per_metric={}
        for metric in ("success_rate","reaching_success_rate","sustained_success_rate","hold_window_max_error_m",
                       "final_error_m","executed_action_change_rms","proposed_action_change_rms",
                       "training_seconds","training_equilibrium_calls"):
            deltas=[]
            for seed in c["seeds"]:
                a=by_key.get((base,seed,c["total_timesteps"],"actor"));b=by_key.get((guided,seed,c["total_timesteps"],"actor"))
                if a and b and a[metric] is not None and b[metric] is not None:
                    deltas.append(b[metric]-a[metric])
            per_metric[metric]=paired_seed_statistics(deltas)
        if not final:
            deltas=[]
            for seed in c["seeds"]:
                auc=[]
                for arm in (base,guided):
                    curve=[by_key.get((arm,seed,step,"actor")) for step in steps]
                    if all(curve):
                        rates=np.array([r["success_rate"] for r in curve])
                        auc.append(float(np.sum(np.diff(steps)*(rates[1:]+rates[:-1])/2)/c["total_timesteps"]))
                if len(auc)==2:deltas.append(auc[1]-auc[0])
            per_metric["normalized_success_auc"]=paired_seed_statistics(deltas)
        # Pair episodes where BOTH policies succeed. A stopped or failing
        # robot must not win a smoothness comparison just by barely moving.
        common=[];counts={}
        for seed in c["seeds"]:
            a=datasets.get((base,seed,c["total_timesteps"],"actor"),[])
            b=datasets.get((guided,seed,c["total_timesteps"],"actor"),[])
            a={int(e["seed"]):e for e in a};b={int(e["seed"]):e for e in b};d=[]
            for key in a.keys() & b.keys():
                if a[key]["success"]==b[key]["success"]=="True" and a[key]["executed_action_change_rms"] and b[key]["executed_action_change_rms"]:
                    d.append(float(b[key]["executed_action_change_rms"])-float(a[key]["executed_action_change_rms"]))
            counts[str(seed)]=len(d)
            if d:common.append(float(np.mean(d)))
        per_metric["common_success_executed_action_change_rms"]={**paired_seed_statistics(common),"episodes_per_seed":counts}
        contrasts[f"{guided}_minus_{base}"]=per_metric
    summary={"complete":not missing,"missing_jobs":missing,"training_seeds":c["seeds"],"planned_arms":c["arms"],
             "plant":"joint_constraints_v1",
             "evaluation_failures":sum(r["evaluation_failures"] for r in rows),"contrasts":contrasts,
             "uncertainty_unit":"paired training seed; percentile bootstrap is descriptive, especially with few seeds",
             "task_profile":c.get("task_profile","legacy"),
             "success_definition":"final_hold_window" if c.get("task_profile","legacy") == "generalized_hold" else "first_hit",
             "scope":"configured reachable-witness distribution; finite command differences, no physical acceleration/jerk or global stability guarantee",
             "final_evaluation":final,"sample_efficiency_improvement_demonstrated":False,
             "note":"Interpret success, error, motion and cost jointly; empty common-success metrics are not zero."}
    write_json(folder/f"{prefix}_report.json",summary)
    print(f"Report: {folder/prefix}_report.json; complete={summary['complete']}, missing={len(missing)}",flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["plan","train","evaluate","report","all","final-evaluate","final-report"],default="plan")
    p.add_argument("--output-dir",type=Path,default=Path("runs/simple_jacobian_study"))
    for name,value in DEFAULTS.items():
        kwargs={"default":None}
        if isinstance(value,list):kwargs.update(nargs="+",type=type(value[0]))
        else:kwargs["type"]=float if value is None else type(value)
        if name=="system":kwargs["choices"]=[f"ctr_{i}" for i in range(4)]
        if name=="exploration_profile":kwargs["choices"]=["paper","gaussian"]
        if name=="task_profile":kwargs["choices"]=TASK_PROFILES
        p.add_argument("--"+name.replace("_","-"),**kwargs)
    a=p.parse_args();folder=a.output_dir.resolve();path=folder/"study.json"
    overrides={k:getattr(a,k) for k in DEFAULTS if getattr(a,k) is not None}
    if path.exists():
        plan=read_json(path)
        if plan.get("plant") != "joint_constraints_v1":
            p.error("This runner uses the joint-only plant. Use a new study directory; old plans belong to the previous implementation.")
        if any(v!=plan["config"].get(k) for k,v in overrides.items()):
            p.error("Study configuration is frozen. Use another output directory for a changed experiment.")
        if a.stage not in ("report","final-report") and plan["source_hashes"]!=source_hashes():
            p.error("Source code changed since this study was planned. Use a new study directory.")
        if a.stage not in ("report","final-report") and plan["runtime_versions"]!=runtime_versions():
            p.error("Dependency versions changed since this study was planned. Use a new study directory.")
    else:
        if a.stage not in ("plan","all","train"):p.error("Create a study plan first")
        if folder.exists() and any(folder.iterdir()):p.error("Study directory is not empty")
        config={**DEFAULTS,**overrides}
        try:validate_config(config)
        except ValueError as exc:p.error(str(exc))
        folder.mkdir(parents=True,exist_ok=True)
        plan={"schema_version":2,"plant":"joint_constraints_v1","config":config,"source_hashes":source_hashes(),"runtime_versions":runtime_versions(),
              "exploration":exploration_settings(config["exploration_profile"],config["noise_std"],config["random_exploration"]),
              "primary_endpoint":"unassisted actor final-window sustained success" if config["task_profile"] == "generalized_hold" else "unassisted actor first-hit success",
              "secondary_endpoints":["first-hit reaching","terminal error","final-window maximum error","command variation","solver work","wall time","interventions"],
              "selection_rule":"final fixed-budget checkpoint; development episodes are not final-test episodes",
              "seed_matching":"same initial network/noise seed and episode-index reset stream, trajectories may diverge"}
        write_json(path,plan)
    print(f"Study {path}: {len(plan['config']['seeds'])} seeds, arms={plan['config']['arms']}, "
          f"{plan['config']['total_timesteps']} transitions/run",flush=True)
    ok=True
    if a.stage in ("train","all"):ok=train(plan,folder)
    if a.stage in ("evaluate","all"):evaluate(plan,folder)
    result=None
    if a.stage in ("report","all"):result=report(plan,folder)
    if a.stage=="final-evaluate":evaluate(plan,folder,final=True);result=report(plan,folder,final=True)
    if a.stage=="final-report":result=report(plan,folder,final=True)
    if not ok or result is not None and (not result["complete"] or result["evaluation_failures"]):
        raise SystemExit(1)


if __name__=="__main__":main()
