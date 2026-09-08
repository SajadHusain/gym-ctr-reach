"""Evaluate an archived or newly trained 19-input policy without unpickling it."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import scipy
import gymnasium
from .env import ArchivedCTREnv
from .policy import NumpyActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--representation', choices=['egocentric', 'proprioceptive'], default='egocentric')
    parser.add_argument('--noisy', action='store_true')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=300000)
    parser.add_argument('--progress-every', type=int, default=50)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.progress_every < 1:
        parser.error('Episode and progress counts must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory')
    env = ArchivedCTREnv(args.representation, evaluation=True, noisy=args.noisy)
    actor = NumpyActor(args.checkpoint, env.action_space)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, start = [], time.monotonic()
    with (args.output_dir / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['episode', 'seed', 'success', 'error_m', 'steps'])
        writer.writeheader()
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            for step in range(1, 151):
                obs, reward, terminated, truncated, info = env.step(actor.predict(obs))
                if terminated or truncated:
                    break
            row = dict(episode=episode, seed=args.seed + episode, success=int(info['is_success']),
                       error_m=float(info['error']), steps=step)
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            if (episode + 1) % args.progress_every == 0 or episode + 1 == args.episodes:
                print('Evaluation {}/{}: success={:.1%}, mean_error={:.3f} mm'.format(
                    episode + 1, args.episodes, np.mean([r['success'] for r in rows]),
                    np.mean([r['error_m'] for r in rows]) * 1000), flush=True)
    errors_mm = np.array([r['error_m'] for r in rows]) * 1000
    summary = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   episodes=len(rows), success_rate=float(np.mean([r['success'] for r in rows])),
                   mean_error_mm=float(errors_mm.mean()), variance_mm2=float(errors_mm.var()),
                   median_error_mm=float(np.median(errors_mm)), p95_error_mm=float(np.percentile(errors_mm, 95)),
                   mean_steps=float(np.mean([r['steps'] for r in rows])), evaluation_tolerance_m=.001,
                   elapsed_seconds=time.monotonic() - start, first_seed=args.seed,
                   representation=args.representation, noisy=args.noisy,
                   python=platform.python_version(), numpy=np.__version__,
                   scipy=scipy.__version__, gymnasium=gymnasium.__version__,
                   source_priority='author_archive', inference_backend='numpy_float32')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == '__main__':
    main()
