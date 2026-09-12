#!/usr/bin/env python3
"""Train the documented 2024 system-0 experiment on a modern SB3 backend."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from ctr_reach_envs.her_replay_buffer import GoalTerminationHerReplayBuffer
from ctr_reach_envs.paper_config import PAPER_PROFILE, paper_configuration
from ctr_reach_envs.paper_policy import PaperDDPG, PaperMlpPolicy, PaperStateExtractor
from rl_utils import make_env
from ctr_reach_envs.training.curriculum import GoalToleranceCurriculumCallback


def build_model(env, spec, *, seed=0, device="cpu", tensorboard_log=None, verbose=1):
    legacy = spec["legacy_defaults"]
    model = PaperDDPG(
        PaperMlpPolicy, env,
        learning_rate=spec["actor_lr"],
        buffer_size=spec["buffer_size"], learning_starts=0,
        batch_size=spec["batch_size"], tau=legacy["tau"], gamma=spec["gamma"],
        train_freq=(legacy["rollout_steps"], "step"), gradient_steps=legacy["gradient_steps"],
        action_noise=NormalActionNoise(
            np.zeros(6), np.asarray(spec["normalized_action_noise_std"]),
        ),
        random_exploration=spec["random_exploration"],
        replay_buffer_class=GoalTerminationHerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": spec["n_sampled_goal"],
            "goal_selection_strategy": spec["goal_selection_strategy"],
            "copy_info_dict": True,
        },
        policy_kwargs={
            "net_arch": spec["hidden_layers"],
            "features_extractor_class": PaperStateExtractor,
            "activation_fn": torch.nn.ReLU,
            "optimizer_kwargs": {"eps": 1e-8},
        },
        seed=seed, device=device, tensorboard_log=tensorboard_log, verbose=verbose,
    )
    if spec["actor_lr"] != spec["critic_lr"]:
        raise ValueError("This profile requires the same actor and critic learning rate")
    model.paper_profile = PAPER_PROFILE
    model.paper_configuration = spec
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper_2024_seed0"))
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override 3M for a smoke test; the 1.5M curriculum is unchanged")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--periodic-eval-episodes", type=int, default=25)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration and exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spec = paper_configuration()
    if args.total_timesteps is not None:
        if args.total_timesteps <= 0:
            raise ValueError("--total-timesteps must be positive")
        spec["total_timesteps"] = args.total_timesteps
    if min(args.eval_freq, args.periodic_eval_episodes, args.checkpoint_freq, args.torch_threads) <= 0:
        raise ValueError("Frequencies, episode count, and thread count must be positive")
    spec["paper_training_budget_used"] = spec["total_timesteps"] == 3_000_000
    spec["seed"] = args.seed
    spec["device"] = args.device
    spec["torch_threads"] = args.torch_threads
    spec["periodic_eval_episodes"] = args.periodic_eval_episodes
    spec["eval_freq"] = args.eval_freq
    spec["checkpoint_freq"] = args.checkpoint_freq
    spec["versions"] = {name: version(name) for name in
                        ("stable-baselines3", "gymnasium", "torch", "numpy", "scipy")}
    spec["versions"]["python"] = platform.python_version()
    print(json.dumps(spec, indent=2), flush=True)
    if args.dry_run:
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Use a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    torch.set_num_threads(args.torch_threads)
    train_env = make_env(evaluation=False, seed=args.seed, profile=PAPER_PROFILE,
                         monitor_dir=output / "monitor")
    eval_env = None
    try:
        eval_env = make_env(evaluation=True, seed=args.seed + 10_000, profile=PAPER_PROFILE,
                            monitor_dir=output / "monitor")
        model = build_model(train_env, spec, seed=args.seed, device=args.device,
                            tensorboard_log=str(output / "tensorboard"))
        callbacks = CallbackList([
            GoalToleranceCurriculumCallback(),
            EvalCallback(eval_env, best_model_save_path=str(output / "best"),
                         log_path=str(output / "evaluations"), eval_freq=args.eval_freq,
                         n_eval_episodes=args.periodic_eval_episodes, deterministic=True),
            CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(output / "checkpoints"),
                               name_prefix="paper_ddpg_her_ctr"),
        ])
        try:
            model.learn(total_timesteps=spec["total_timesteps"], callback=callbacks)
        except KeyboardInterrupt:
            model.save(output / "interrupted_model")
            print("Interrupted checkpoint saved. This script starts fresh training runs.")
            return
        model.save(output / "final_model")
        print(f"Saved final model to {output / 'final_model.zip'}")
        print(f'python evaluate.py "{output / "final_model.zip"}" --profile {PAPER_PROFILE} '
              f'--episodes 1000 --seed 300000 --output-dir "{output / "evaluation_1000"}"')
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
