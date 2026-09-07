#!/usr/bin/env python3
"""Train a goal-conditioned CTR reaching policy with DDPG and HER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from rl_utils import make_env


class GoalToleranceCurriculumCallback(BaseCallback):
    """Advance the environment's 20 mm -> 1 mm tolerance schedule."""

    def _on_step(self) -> bool:
        self.training_env.env_method("update_goal_tolerance", self.num_timesteps)
        tolerances = self.training_env.env_method("get_goal_tolerance")
        self.logger.record("curriculum/position_tolerance_m", float(np.mean(tolerances)))
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ddpg_her"))
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--periodic-eval-episodes", type=int, default=25)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    positive_arguments = {
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "periodic_eval_episodes": args.periodic_eval_episodes,
        "checkpoint_freq": args.checkpoint_freq,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
    }
    if any(value <= 0 for value in positive_arguments.values()):
        raise ValueError(f"Arguments must be positive: {positive_arguments}")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts cannot be negative")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_env = make_env(
        evaluation=False,
        seed=args.seed,
        monitor_dir=output_dir / "monitor",
    )
    eval_env = make_env(
        evaluation=True,
        seed=args.seed + 10_000,
        monitor_dir=output_dir / "monitor",
    )

    # SB3 adds this noise in normalized action coordinates, which prevents metre/radian
    # scale mismatch from biasing exploration toward rotations.
    n_actions = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=0.1 * np.ones(n_actions, dtype=np.float32),
    )
    model = DDPG(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
            # Reward recomputation uses the tolerance stored with each transition.
            "copy_info_dict": True,
        },
        learning_rate=1e-3,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.95,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 256, 256]},
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    callbacks = CallbackList(
        [
            GoalToleranceCurriculumCallback(),
            EvalCallback(
                eval_env,
                best_model_save_path=str(output_dir / "best"),
                log_path=str(output_dir / "evaluations"),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.periodic_eval_episodes,
                deterministic=True,
            ),
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(output_dir / "checkpoints"),
                name_prefix="ddpg_her_ctr",
            ),
        ]
    )
    run_config = vars(args).copy()
    run_config["output_dir"] = str(output_dir)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        model.save(output_dir / "final_model")
    finally:
        train_env.close()
        eval_env.close()

    print(f"Saved final model to {output_dir / 'final_model.zip'}")
    print(
        "Run the separate evaluator for the requested 1,000 episodes: "
        f"python evaluate.py {output_dir / 'final_model.zip'} --episodes 1000"
    )


if __name__ == "__main__":
    main()
