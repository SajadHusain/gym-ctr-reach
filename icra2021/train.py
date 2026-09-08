"""Train the author's archived DDPG/HER learner, one environment per MPI rank."""
import argparse
import functools
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from .bootstrap import activate
from .env import ArchivedCTREnv, LegacyLearnerInterface

ALGORITHM_SETTINGS = dict(policy_kwargs={'layers': [128, 128, 128]},
                          actor_lr=.0005, critic_lr=.0005, gamma=.95, tau=.001,
                          buffer_size=10000, batch_size=256, normalize_observations=True,
                          normalize_returns=False, random_exploration=.294,
                          nb_rollout_steps=100, nb_train_steps=50, nb_eval_steps=200,
                          n_cpu_tf_sess=1, verbose=0)
ACTION_NOISE_STD = [.00065] * 3 + [.025] * 3


def make_model(env, seed=0, mpi_ready=True):
    activate()
    from stable_baselines import DDPG, HER
    from stable_baselines.common.noise import NormalActionNoise
    from stable_baselines.her.replay_buffer import HindsightExperienceReplayWrapper
    from mpi4py import MPI
    model = HER('MlpPolicy', LegacyLearnerInterface(env), DDPG,
                n_sampled_goal=4, goal_selection_strategy='future',
                action_noise=NormalActionNoise(np.zeros(6), np.array(ACTION_NOISE_STD)),
                seed=seed, **ALGORITHM_SETTINGS)
    if mpi_ready:
        class CollectiveReadyHER(HindsightExperienceReplayWrapper):
            def can_sample(self, n_samples):
                ready = super().can_sample(n_samples)
                # With per-rank episode buffers, local readiness can differ.
                # All ranks must enter MPI Adam's collectives together.
                return bool(MPI.COMM_WORLD.allreduce(int(ready), op=MPI.MIN))
        model.replay_wrapper = functools.partial(
            CollectiveReadyHER, n_sampled_goal=4,
            goal_selection_strategy=model.goal_selection_strategy, wrapped_env=model.env)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representation', choices=['egocentric', 'proprioceptive'], default='egocentric')
    parser.add_argument('--curriculum', choices=['constant', 'linear', 'decay'], default='decay')
    parser.add_argument('--noisy', action='store_true')
    parser.add_argument('--total-timesteps', type=int, default=500000, help='Per MPI worker, as in original learner')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expected-workers', type=int, default=19)
    parser.add_argument('--checkpoint-every', type=int, default=125000)
    parser.add_argument('--progress-every', type=int, default=1000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.total_timesteps < 1 or args.progress_every < 1 or args.checkpoint_every < 1:
        parser.error('Step counts must be positive')
    activate()
    import tensorflow as tf
    import gymnasium
    import scipy
    from mpi4py import MPI
    if not tf.__version__.startswith('1.15.'):
        raise RuntimeError('Use the isolated TensorFlow 1.15 runtime; see icra2021/README.md')
    comm = MPI.COMM_WORLD
    rank, workers = comm.Get_rank(), comm.Get_size()
    if workers != args.expected_workers:
        raise RuntimeError('Expected {} MPI workers, got {}. Use --expected-workers explicitly for a smoke run.'.format(args.expected_workers, workers))
    error = None
    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            error = 'Output directory must be empty: ' + str(args.output_dir)
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    env = ArchivedCTREnv(args.representation, args.curriculum, noisy=args.noisy)
    rank_seed = args.seed + 1000000 * rank
    model = make_model(env, rank_seed)
    if rank == 0:
        config = dict(vars(args), output_dir=str(args.output_dir), workers=workers,
                      global_real_transitions=args.total_timesteps * workers,
                      environment=env.configuration, source_priority='author_archive',
                      algorithm=ALGORITHM_SETTINGS, action_noise_std=ACTION_NOISE_STD,
                      her=dict(strategy='future', sampled_goals=4, insertion='end_of_episode'),
                      python=platform.python_version(), tensorflow=tf.__version__,
                      numpy=np.__version__, scipy=scipy.__version__, gymnasium=gymnasium.__version__,
                      seed_rule='base_seed + 1000000 * MPI_rank',
                      curriculum_clock='original callback locals[step]',
                      mpi_replay_readiness='collective MIN to prevent mismatched optimizer collectives')
        (args.output_dir / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    start = time.monotonic()
    completed = []

    def callback(local, unused):
        step = local['self'].num_timesteps
        env.update_goal_tolerance(local['step'])
        if rank == 0:
            if local['done']:
                completed.append(bool(local['info']['is_success']))
            if step % args.progress_every == 0:
                print('step/worker={} global={} tolerance_mm={:.4f} recent_success={:.3f} elapsed_s={:.1f}'.format(
                    step, step * workers, env.core.goal_tol_obj.get_tol() * 1000,
                    np.mean(completed[-100:]) if completed else 0., time.monotonic() - start), flush=True)
            if step % args.checkpoint_every == 0:
                model.save(str(args.output_dir / ('checkpoint_{}.zip'.format(step))))
        return True

    model.learn(args.total_timesteps, callback=callback, log_interval=10)
    params = model.model.get_parameters()
    assert all(np.all(np.isfinite(v)) for v in params.values()), 'Non-finite learned parameters'
    digest = hashlib.sha256(b''.join(v.tobytes() for v in params.values())).hexdigest()
    digests = comm.allgather(digest)
    assert len(set(digests)) == 1, 'MPI worker parameters diverged'
    if rank == 0:
        model.save(str(args.output_dir / 'final_model.zip'))
        health = dict(finite_parameters=True, mpi_parameter_hashes=digests,
                      critic_optimizer_steps=int(model.model.critic_optimizer.t))
        (args.output_dir / 'training_health.json').write_text(json.dumps(health, indent=2) + '\n')
        print('Saved ' + str(args.output_dir / 'final_model.zip'), flush=True)
    comm.Barrier()
    env.close()


if __name__ == '__main__':
    main()
