"""Matched ordinary and guided DDPG/HER on the original IVP robot model."""
import argparse
import csv
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from ctr_reach_envs.her_replay_buffer import GoalTerminationHerReplayBuffer
from ctr_reach_envs.paper_policy import PaperDDPG, PaperMlpPolicy, PaperStateExtractor
from ctr_reach_envs.ivp.config import load_spec, resolve, make_env
from ctr_reach_envs.ivp.rl import OriginalHerReplayBuffer, OriginalJacobianDDPG
from ctr_reach_envs.ivp.observations import FEATURE_COUNTS, observation_settings
from ctr_reach_envs.ivp.policy import PhysicsStateExtractor


def build_model(env, config, *, force_guided_class=False, device="cpu", verbose=0):
    s, p = config["spec"], config["physics"]
    guided = force_guided_class or max(p["weight"], p["final_weight"]) > 0 or p["objective"] == "rl_only_checked"
    cls = OriginalJacobianDDPG if guided else PaperDDPG
    extra = {}
    if guided:
        extra = dict(physics_lengths=env.trig_obj.tube_lengths[0].tolist(),
            original_n_substeps=env.n_substeps, original_constrain_alpha=env.trig_obj.constrain_alpha,
            physics_weight=p["weight"], physics_final_weight=p["final_weight"],
            physics_anneal_steps=p["anneal_steps"], physics_integration=p["integration"],
            physics_max_aux_ratio=p["max_aux_ratio"], actor_objective=p["objective"],
            physics_gain=p["gain"], physics_cartesian_scale=p["scale_m"], physics_max_tip_step=p["max_tip_step_m"],
            wait_for_completed_batch=True)
    model = cls(PaperMlpPolicy, env, learning_rate=s["actor_lr"],
        buffer_size=s["buffer_size"], learning_starts=0, batch_size=s["batch_size"],
        tau=s["legacy_defaults"]["tau"], gamma=s["gamma"],
        train_freq=(s["legacy_defaults"]["rollout_steps"], "step"), gradient_steps=s["legacy_defaults"]["gradient_steps"],
        action_noise=NormalActionNoise(np.zeros(6), np.asarray(s["normalized_action_noise_std"])),
        random_exploration=s["random_exploration"],
        replay_buffer_class=OriginalHerReplayBuffer if guided else GoalTerminationHerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=s["n_sampled_goal"], goal_selection_strategy=s["goal_selection_strategy"], copy_info_dict=True),
        policy_kwargs=dict(net_arch=s["hidden_layers"], features_extractor_class=(
                           PhysicsStateExtractor if "physics" in env.observation_space.spaces else PaperStateExtractor),
                           activation_fn=torch.nn.ReLU, optimizer_kwargs={"eps": 1e-8}),
        seed=config["seed"], device=device, verbose=verbose, **extra)
    model.original_ivp_config = config
    return model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def motion_metrics(actions, deltas):
    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else 0.
    a, dq = np.asarray(actions).reshape(-1, 6), np.asarray(deltas).reshape(-1, 6)
    da, ddq = np.diff(a, axis=0), np.diff(dq, axis=0)
    return dict(action_change_rms=rms(da), extension_increment_rms_m=rms(dq[:, :3]),
        rotation_increment_rms_rad=rms(dq[:, 3:]), extension_increment_change_rms_m=rms(ddq[:, :3]),
        rotation_increment_change_rms_rad=rms(ddq[:, 3:]))


class AuditCallback(BaseCallback):
    def __init__(self, plant, config, output, checkpoint_freq, progress_every):
        super().__init__()
        self.plant, self.config, self.output = plant, config, output
        self.checkpoint_freq, self.progress_every = checkpoint_freq, progress_every
        self.started = time.perf_counter()
        self.episodes, self.successes, self.numerical_failures = 0, 0, 0
        self.actions, self.deltas, self.episode_reward = [], [], 0.
        self.last_update = -1
        self.maximum_replay_error = 0.
        self.linearization_sum, self.linearization_max, self.linearization_count = 0., 0., 0
        self.streams = []
        for name, fields in (("episodes", ["episode", "timestep", "success", "steps", "reward", "error_m", "tolerance_m",
                "action_change_rms", "extension_increment_rms_m", "rotation_increment_rms_rad",
                "extension_increment_change_rms_m", "rotation_increment_change_rms_rad"]),
                ("updates", ["timestep", "gradient_updates", "physics_metrics_json"])):
            stream = (output / (name + ".csv")).open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            self.streams.append(stream); setattr(self, name + "_writer", writer)

    def record_updates(self):
        if self.model._n_updates != self.last_update:
            self.updates_writer.writerow(dict(timestep=self.num_timesteps, gradient_updates=self.model._n_updates,
                physics_metrics_json=json.dumps(getattr(self.model, "last_physics_metrics", {}))))
            self.last_update = self.model._n_updates

    def _on_step(self):
        info = self.locals["infos"][0]
        proposal = info["proposed_action"]
        discrepancy = float(np.max(abs(self.locals["buffer_actions"][0] - proposal)))
        self.maximum_replay_error = max(discrepancy, self.maximum_replay_error)
        if "jacobian_linearization_error_m" in info:
            linearization_error = info["jacobian_linearization_error_m"]
            self.linearization_sum += linearization_error
            self.linearization_max = max(self.linearization_max, linearization_error)
            self.linearization_count += 1
        if discrepancy > 1e-6:
            raise RuntimeError("Collected proposal/replay mismatch")
        if info["solver_failure"]:
            self.numerical_failures += 1
            raise RuntimeError("Original IVP failed; aborting instead of learning from a fabricated transition")
        self.actions.append(proposal.copy()); self.deltas.append(info["applied_delta_q"].copy())
        self.episode_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episodes += 1; self.successes += int(info["is_success"])
            self.episodes_writer.writerow(dict(episode=self.episodes, timestep=self.num_timesteps,
                success=bool(info["is_success"]), steps=len(self.actions), reward=self.episode_reward,
                error_m=info["error"], tolerance_m=info["position_tolerance"], **motion_metrics(self.actions, self.deltas)))
            self.actions, self.deltas, self.episode_reward = [], [], 0.
        # Preserve the historical paper callback timing, including tolerance in HER infos.
        self.plant.update_goal_tolerance(self.num_timesteps)
        physics = self.config["physics"]
        if (physics["final_weight"] == 0 and self.num_timesteps >= physics["anneal_steps"]
                and not self.plant.observation_uses_jacobian):
            # Future updates no longer use guidance. Do not pay for unused
            # derivatives; older source-state derivatives remain in replay.
            self.plant.compute_jacobian = False
        self.record_updates()
        if self.checkpoint_freq and self.num_timesteps % self.checkpoint_freq == 0:
            folder = self.output / "checkpoints" / f"step_{self.num_timesteps:09d}"
            folder.mkdir(parents=True, exist_ok=False)
            self.model.save(folder / "model")
            write_json(folder / "config.json", self.config)
        if self.num_timesteps % self.progress_every == 0:
            for stream in self.streams: stream.flush()
            print(f"Step {self.num_timesteps}: episodes={self.episodes}, successes={self.successes}, "
                  f"error={1000*info['error']:.3f} mm, tolerance={1000*info['position_tolerance']:.3f} mm, "
                  f"elapsed={time.perf_counter()-self.started:.0f} s", flush=True)
        return True

    def close(self):
        for stream in self.streams: stream.close()


def parse_args(argv=None, *, baseline=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--segment-mode", choices=("legacy", "continuous"), default=None)
    p.add_argument("--tolerance-curriculum", choices=("exponential", "linear", "constant"))
    for flag in ("total-timesteps", "curriculum-steps", "episode-steps", "batch-size", "buffer-size", "hidden-width", "train-freq", "gradient-steps"):
        p.add_argument("--" + flag, type=int)
    for flag in ("initial-tolerance-m", "tolerance-m"):
        p.add_argument("--" + flag, type=float)
    p.add_argument("--physics-observation", choices=tuple(FEATURE_COUNTS),
                   help="Actor/critic information: none (default), jacobian, jacobian_limits, or zeros control")
    p.add_argument("--physics-observation-scale-m", type=float,
                   help="Distance scale before bounded Jacobian encoding (default 0.002 m)")
    p.add_argument("--physics-weight", type=float, default=0. if baseline else .1)
    p.add_argument("--physics-final-weight", type=float,
                   help="Defaults to min(initial weight, 0.01); set 0 explicitly to end guidance")
    p.add_argument("--physics-anneal-steps", type=int)
    p.add_argument("--physics-integration", choices=("rl_priority", "sum"), default="rl_priority")
    p.add_argument("--physics-max-aux-ratio", type=float, default=.1,
                   help="Maximum weighted auxiliary / RL parameter-gradient norm in rl_priority mode")
    p.add_argument("--physics-gain", type=float, default=.5)
    p.add_argument("--physics-scale-m", type=float, default=.002)
    p.add_argument("--physics-max-tip-step-m", type=float, default=.002)
    p.add_argument("--actor-objective", choices=("hybrid", "mechanics_only", "rl_only_checked"), default="hybrid")
    p.add_argument("--checkpoint-freq", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7101)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if baseline and (args.physics_weight != 0 or args.physics_final_weight not in (None, 0.) or args.actor_objective != "hybrid"):
        p.error("The baseline command requires zero guidance and the ordinary RL objective")
    return args


def main(argv=None, *, baseline=False):
    a = parse_args(argv, baseline=baseline)
    spec, environment, provenance = load_spec(a.config)
    observation = observation_settings(spec.get("physics_observation"))
    if a.physics_observation is not None:
        observation["mode"] = a.physics_observation
    if a.physics_observation_scale_m is not None:
        observation["scale_m"] = a.physics_observation_scale_m
    spec["physics_observation"] = observation_settings(observation)
    if a.tolerance_curriculum is not None:
        spec["curriculum_function"] = "decay" if a.tolerance_curriculum == "exponential" else a.tolerance_curriculum
    fields = dict(total_timesteps="total_timesteps", curriculum_steps="curriculum_steps",
        initial_tolerance_m="initial_tolerance_m", tolerance_m="final_tolerance_m", episode_steps="max_steps_per_episode",
        batch_size="batch_size", buffer_size="buffer_size")
    for arg, key in fields.items():
        if getattr(a, arg) is not None: spec[key] = getattr(a, arg)
    if a.hidden_width is not None: spec["hidden_layers"] = [a.hidden_width] * len(spec["hidden_layers"])
    if a.train_freq is not None: spec["legacy_defaults"]["rollout_steps"] = a.train_freq
    if a.gradient_steps is not None: spec["legacy_defaults"]["gradient_steps"] = a.gradient_steps
    if min(a.torch_threads, a.progress_every) < 1 or a.checkpoint_freq < 0:
        raise ValueError("Invalid frequency or thread count")
    physics = dict(weight=a.physics_weight, final_weight=min(a.physics_weight,.01) if a.physics_final_weight is None else a.physics_final_weight,
        anneal_steps=a.physics_anneal_steps or spec["curriculum_steps"], integration=a.physics_integration,
        max_aux_ratio=a.physics_max_aux_ratio, gain=a.physics_gain, scale_m=a.physics_scale_m,
        max_tip_step_m=a.physics_max_tip_step_m, objective=a.actor_objective)
    if any(not np.isfinite(physics[k]) or physics[k] < 0 for k in ("weight", "final_weight")):
        raise ValueError("Physics weights must be finite and nonnegative")
    if any(not np.isfinite(physics[k]) or physics[k] <= 0 for k in ("anneal_steps", "max_aux_ratio", "gain", "scale_m", "max_tip_step_m")):
        raise ValueError("Physics scales, gain, ratio and duration must be positive")
    config = resolve(spec, environment, segment_mode=a.segment_mode, seed=a.seed, physics=physics, provenance=provenance)
    config.update(versions={name: version(name) for name in ("stable-baselines3", "gymnasium", "torch", "numpy", "scipy")},
                  python=platform.python_version(), torch_threads=a.torch_threads, device=a.device)
    if a.dry_run:
        print(json.dumps(config, indent=2)); return
    output = a.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    torch.set_num_threads(a.torch_threads)
    started = time.perf_counter()
    plant, model, audit, failure = None, None, None, None
    complete = False
    try:
        collect_jacobian = max(physics["weight"], physics["final_weight"]) > 0 and physics["objective"] != "rl_only_checked"
        plant = make_env(config, compute_jacobian=collect_jacobian)
        model = build_model(plant, config, device=a.device)
        audit = AuditCallback(plant, config, output, a.checkpoint_freq, a.progress_every)
        model.learn(total_timesteps=spec["total_timesteps"], callback=audit)
        complete = True
    except (Exception, KeyboardInterrupt) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            model.save(output / ("final_model" if complete else "interrupted_model"))
        if audit is not None:
            audit.record_updates(); audit.close()
        summary = dict(complete=complete, failure=failure, timesteps=0 if model is None else model.num_timesteps,
            gradient_updates=0 if model is None else model._n_updates,
            completed_episodes=0 if audit is None else audit.episodes, rollout_successes=0 if audit is None else audit.successes,
            maximum_replay_proposal_error=None if audit is None else audit.maximum_replay_error,
            numerical_failures=0 if audit is None else audit.numerical_failures,
            costs={} if plant is None else plant.costs,
            mean_linearization_error_m=(audit.linearization_sum/audit.linearization_count
                if audit is not None and audit.linearization_count else None),
            max_linearization_error_m=(audit.linearization_max if audit is not None and audit.linearization_count else None),
            invalid_jacobian_reasons={} if plant is None else dict(plant.invalid_reasons),
            physics_updates=0 if model is None else getattr(model, "physics_update_count", 0),
            last_physics_metrics={} if model is None else getattr(model, "last_physics_metrics", {}),
            all_parameters_finite=None if model is None else all(bool(torch.isfinite(x).all()) for x in model.policy.parameters()),
            elapsed_seconds=time.perf_counter()-started, environment_fingerprint=config["environment_fingerprint"],
            segment_mode=config["segment_mode"], equilibrium_shooting_calls=0,
            training_tolerance_m=None if plant is None else plant.get_goal_tolerance())
        summary.update(physics_observation=config["spec"]["physics_observation"],
                       actor_uses_jacobian=False if plant is None else plant.observation_uses_jacobian,
                       physics_feature_count=FEATURE_COUNTS[observation["mode"]])
        write_json(output / "summary.json", summary)
        if plant is not None: plant.close()
    print(json.dumps(summary, indent=2), flush=True)
    if not complete:
        raise SystemExit(1)
    print(f"Saved {output / 'final_model.zip'}", flush=True)
