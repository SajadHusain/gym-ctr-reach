"""Offline diagnostics for original-IVP checkpoints; never resume RL training."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from stable_baselines3 import DDPG
from .config import PROFILE, fingerprint, make_env
from .diagnostic_metrics import motion_metrics, summarize_motion
from .rl import OriginalJacobianLoss


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def actor_actions(actor, observations):
    with torch.no_grad():
        return actor(observations).detach().cpu().numpy()


def collect_states(env, actor, count, seed, rollout_steps):
    """One source state per independently seeded task, on frozen-actor rollouts.

    Terminal states are never stepped again. Numerical failures abort collection
    rather than silently replacing difficult tasks with easier ones.
    """
    states = []
    for index in range(count):
        task_seed = seed+index
        obs, info = env.reset(seed=task_seed)
        depth = int(np.random.default_rng(task_seed).integers(rollout_steps+1))
        steps = 0
        for _ in range(depth):
            if info["is_success"]:
                break
            source_joints, source_goal = env.trig_obj.joints.copy(), env.desired_goal.copy()
            action = actor_actions(actor, actor.obs_to_tensor(obs)[0])[0]
            obs, _, terminated, truncated, info = env.step(action)
            if info["solver_failure"]:
                raise RuntimeError(f"IVP failure collecting task seed {task_seed}")
            steps += 1
            if terminated or truncated:
                # Retain the last actual decision state, not a terminal state.
                obs, info = env.reset(seed=task_seed, options=dict(
                    initial_joints=source_joints, goal=source_goal, system=env.system))
                steps -= 1
                break
        previous = env.compute_jacobian
        try:
            env.compute_jacobian = True
            context = env.source_physics()
        finally:
            env.compute_jacobian = previous
        state = dict(seed=task_seed, rollout_steps=steps, system=env.system,
                     joints=context["joints"].copy(), tip=env.achieved_goal.copy(),
                     goal=env.desired_goal.copy(), scales=context["action_scales"].copy(),
                     jacobian=context["jacobian"].copy(), jacobian_valid=bool(context["jacobian_valid"]),
                     observation={k: v.copy() for k, v in obs.items()},
                     initially_successful=bool(info["is_success"]))
        state["fingerprint"] = hashlib.sha256(np.r_[state["joints"], state["goal"]].astype("<f8").tobytes()).hexdigest()
        states.append(state)
    return states


def physics_sample(states, device="cpu"):
    def tensor(values):
        return torch.as_tensor(np.asarray(values), device=device)
    return SimpleNamespace(
        observations={k: tensor([s["observation"][k] for s in states]) for k in states[0]["observation"]},
        joints=tensor([s["joints"] for s in states]),
        action_scales=tensor([s["scales"] for s in states]),
        jacobian=tensor([s["jacobian"] for s in states]),
        jacobian_valid=tensor([s["jacobian_valid"] and not s["initially_successful"] for s in states]))


def auxiliary_probe(actor, loss_fn, fit_sample, relative_step=1e-4, max_backtracks=8):
    """One normalized SGD step on a COPY, chosen using fit loss only.

    This is a causal sensitivity probe, not a proposed training optimizer.
    Neither held-out outcomes nor the critic participate in step selection.
    """
    candidate = deepcopy(actor)
    candidate.set_training_mode(False)
    parameters = tuple(candidate.parameters())
    original = parameters_to_vector(parameters).detach().clone()
    loss = loss_fn(candidate(fit_sample.observations), fit_sample)
    before = float(loss.detach())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    gradient = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(parameters, grads)]).detach()
    norm = float(gradient.norm())
    if not np.isfinite(before) or not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite auxiliary probe")
    result = dict(fit_loss_before=before, fit_loss_after=before, accepted=False,
                  gradient_norm=norm, requested_relative_step=relative_step,
                  parameter_step_norm=0., backtracks=0,
                  valid_fit_samples=int(fit_sample.jacobian_valid.sum()))
    if norm == 0.:
        return candidate, result
    step_norm = relative_step*max(float(original.norm()), 1.)
    for backtracks in range(max_backtracks+1):
        delta = -(step_norm*2.**(-backtracks)/norm)*gradient
        with torch.no_grad():
            vector_to_parameters((original+delta).clone(), parameters)
            after = float(loss_fn(candidate(fit_sample.observations), fit_sample))
        if np.isfinite(after) and after < before:
            result.update(fit_loss_after=after, accepted=True,
                          parameter_step_norm=float(delta.norm()), backtracks=backtracks)
            return candidate, result
    with torch.no_grad():
        vector_to_parameters(original.clone(), parameters)
    result["backtracks"] = max_backtracks
    return candidate, result


def measure_action(env, state, action, loss_fn):
    """Use the actual environment transition, not the Jacobian as ground truth."""
    action = np.asarray(action, dtype=np.float32)
    row = dict(seed=state["seed"], state_fingerprint=state["fingerprint"],
               jacobian_valid=state["jacobian_valid"], initially_successful=state["initially_successful"],
               action_json=json.dumps(action.tolist()), failure="")
    try:
        env.reset(seed=state["seed"], options=dict(initial_joints=state["joints"],
                  goal=state["goal"], system=state["system"]))
        if (not np.allclose(env.trig_obj.joints, state["joints"], atol=1e-12, rtol=0)
                or not np.allclose(env.achieved_goal, state["tip"], atol=1e-9, rtol=0)):
            raise RuntimeError("Probe could not restore its source state")
        _, _, _, _, info = env.step(action)
        if info["solver_failure"]:
            raise RuntimeError("IVP forward solve failed")
        actual = env.achieved_goal-state["tip"]
        with torch.no_grad():
            dq = loss_fn.projected_delta(torch.as_tensor(action[None]),
                 torch.as_tensor(state["joints"][None]), torch.as_tensor(state["scales"][None])).numpy()[0]
        predicted = state["jacobian"]@dq if state["jacobian_valid"] else None
        row.update(motion_metrics(state["tip"], state["goal"], actual, predicted))
        row["projection_error_inf"] = float(np.max(np.abs(dq-info["applied_delta_q"])))
        row["constraint_active"] = bool(np.max(np.abs(
            dq-env.n_substeps*state["scales"]*action)) > 1e-10)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        row["failure"] = f"{type(exc).__name__}: {exc}"
    return row


def serializable_states(states):
    return [{k: ({a: b.tolist() for a, b in v.items()} if k == "observation"
                  else v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in s.items()} for s in states]


def run_diagnostics(model, config, output_dir, *, states=16, seed=910000,
                    rollout_steps=8, action_fractions=(.1, .5, 1.), relative_step=1e-4):
    if config.get("profile") != PROFILE:
        raise ValueError("Diagnostics require original-ivp-comparison-v1, not mechanics. Use an original-IVP checkpoint.")
    if states < 1 or seed < 0 or rollout_steps < 0 or not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("Invalid diagnostic count, seed or probe step")
    if not action_fractions or any(not np.isfinite(x) or not 0 < x <= 1 for x in action_fractions):
        raise ValueError("Action fractions must lie in (0,1]")
    if fingerprint(getattr(model, "original_ivp_config", {})) != fingerprint(config):
        raise ValueError("Adjacent config.json does not match the checkpoint's embedded config")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    actor = deepcopy(model.actor).to("cpu")
    actor.set_training_mode(False)
    p = config["physics"]
    env = make_env(config, evaluation=True, compute_jacobian=False)
    loss_fn = OriginalJacobianLoss(env.trig_obj.tube_lengths[0], env.n_substeps,
        env.trig_obj.constrain_alpha, p["scale_m"], p["gain"], p["max_tip_step_m"])
    summary = dict(complete=False, profile=PROFILE, environment_fingerprint=config["environment_fingerprint"],
        states_per_split=states, seed=seed, probe_seed=seed+states,
        checkpoint_timesteps=model.num_timesteps, action_fractions=list(action_fractions),
        heldout_selection="Independent task seeds, frozen-actor rollout states; no fit/probe overlap",
        scope="Local one-step diagnostics, not an estimate of long-horizon return or a stability certificate",
        training_unchanged=True, prediction_rows=0, probe_rows=0)
    prediction_rows, probe_rows = [], []
    try:
        fit = collect_states(env, actor, states, seed, rollout_steps)
        probe = collect_states(env, actor, states, seed+states, rollout_steps)
        if {s["fingerprint"] for s in fit} & {s["fingerprint"] for s in probe}:
            raise RuntimeError("Fit/probe state overlap; choose different diagnostic seeds")
        write_json(output/"states.json", dict(fit=serializable_states(fit), probe=serializable_states(probe)))
        fit_sample, probe_sample = physics_sample(fit), physics_sample(probe)
        candidate, update = auxiliary_probe(actor, loss_fn, fit_sample, relative_step)
        before_actions = actor_actions(actor, probe_sample.observations)
        after_actions = actor_actions(candidate, probe_sample.observations)
        with torch.no_grad():
            update.update(heldout_loss_before=float(loss_fn(actor(probe_sample.observations), probe_sample)),
                          heldout_loss_after=float(loss_fn(candidate(probe_sample.observations), probe_sample)),
                          valid_heldout_samples=int(probe_sample.jacobian_valid.sum()))
        summary["auxiliary_probe"] = update
        summary["initially_successful_sources"] = dict(
            fit=sum(s["initially_successful"] for s in fit),
            probe=sum(s["initially_successful"] for s in probe))
        for index, state in enumerate(probe):
            # Already-successful source states are documented but are not decisions.
            if state["initially_successful"]:
                continue
            random_action = np.random.default_rng(state["seed"]).uniform(-1., 1., 6).astype(np.float32)
            for name, proposal in (("actor", before_actions[index]), ("random", random_action)):
                for fraction in action_fractions:
                    row = measure_action(env, state, fraction*proposal, loss_fn)
                    row.update(action_kind=name, action_fraction=fraction)
                    prediction_rows.append(row)
            zero = measure_action(env, state, np.zeros(6, dtype=np.float32), loss_fn)
            zero.update(action_kind="zero", action_fraction=0.)
            prediction_rows.append(zero)
            before = measure_action(env, state, before_actions[index], loss_fn)
            after = measure_action(env, state, after_actions[index], loss_fn)
            row = dict(after, baseline_failure=before["failure"],
                       actual_progress_before_m=before.get("actual_progress_m"),
                       actual_progress_change_m=None,
                       action_change_norm=float(np.linalg.norm(after_actions[index]-before_actions[index])))
            if before["failure"]:
                row["failure"] = row["failure"] or "Baseline: "+before["failure"]
            if not row["failure"]:
                row["actual_progress_change_m"] = after["actual_progress_m"]-before["actual_progress_m"]
            probe_rows.append(row)
            print(f"Diagnosed source {index+1}/{states}", flush=True)
        summary["complete"] = True
    except BaseException as exc:
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for name, rows in (("prediction", prediction_rows), ("auxiliary_probe", probe_rows)):
            with (output/(name+".csv")).open("w", newline="", encoding="utf-8") as stream:
                if rows:
                    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                    writer.writeheader(); writer.writerows(rows)
        groups = sorted({(r["action_kind"], r["action_fraction"]) for r in prediction_rows})
        summary.update(prediction_rows=len(prediction_rows), probe_rows=len(probe_rows),
            prediction_by_action={f"{name}:{fraction:g}": summarize_motion([
                r for r in prediction_rows if r["action_kind"] == name and r["action_fraction"] == fraction])
                for name, fraction in groups},
            heldout_actual_progress=summarize_motion(probe_rows),
            costs=env.costs, invalid_jacobian_reasons=dict(env.invalid_reasons))
        summary["numerical_failures"] = sum(bool(r["failure"]) for r in prediction_rows+probe_rows)
        summary["auxiliary_probe_informative"] = bool(summary.get("auxiliary_probe", {}).get("accepted")
            and summary.get("auxiliary_probe", {}).get("valid_heldout_samples", 0) and probe_rows)
        if summary["numerical_failures"]:
            summary["complete"] = False
        write_json(output/"summary.json", summary)
        env.close()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--action-fractions", type=float, nargs="+", default=[.1, .5, 1.])
    parser.add_argument("--relative-step", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    config = json.loads((args.model.parent/"config.json").read_text(encoding="utf-8"))
    if config.get("profile") != PROFILE:
        parser.error("Use an original-IVP checkpoint; the mechanics profile has a different plant.")
    model = DDPG.load(args.model, device="cpu")
    summary = run_diagnostics(model, config, args.output_dir, states=args.states, seed=args.seed,
        rollout_steps=args.rollout_steps, action_fractions=args.action_fractions, relative_step=args.relative_step)
    summary["checkpoint"] = str(args.model.resolve())
    digest = hashlib.sha256()
    with args.model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    summary["checkpoint_sha256"] = digest.hexdigest()
    write_json(args.output_dir/"summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not summary["complete"]:
        raise SystemExit(1)
