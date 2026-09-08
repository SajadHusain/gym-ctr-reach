"""Download pinned author sources, verify them, and build compatibility copies."""
import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / '_sources'
RUNTIME = ROOT / '_runtime'
CHECKPOINT_SHA256 = 'f4f19d922d9cce52876bbc3256c65b6d606e6c2731317f69d1554f87bc05be1b'
CHECKPOINT_URL = ('https://raw.githubusercontent.com/keshaviyengar/ctr_policy_ros/'
                  '2a220e37ab14b98b89bb7ce9fa234b357a94e952/example_model/'
                  'cras_exp_6/learned_policy/500000_saved_model.pkl')


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def manifest():
    return json.loads((ROOT / 'sources.json').read_text())


def fetch(item):
    path = SOURCES / item['path']
    if path.exists() and blob_sha(path.read_bytes()) == item['git_blob_sha1']:
        return
    url = 'https://raw.githubusercontent.com/{}/{}/{}'.format(
        item['repository'], item['ref'], item['source_path'])
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if blob_sha(data) != item['git_blob_sha1']:
        raise ValueError('Source hash mismatch: ' + item['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def patch_source(path, text, scalar_compat=True):
    # Redirect imports only; do not monkey-patch the installed Gymnasium module.
    text = text.replace('import gym\n', 'from icra2021 import gym_compat as gym\n')
    text = text.replace('from gym.spaces import ', 'from gymnasium.spaces import ')
    text = text.replace('from gym import spaces', 'from gymnasium import spaces')
    text = text.replace('from gym.envs.registration import register',
                        'from icra2021.gym_compat import register')
    # Remove eager imports of unrelated algorithms and video/Atari dependencies.
    if path == 'stable_baselines/__init__.py':
        return "from stable_baselines.ddpg import DDPG\nfrom stable_baselines.her import HER\n__version__ = '2.10.0a0'\n"
    if path in ('stable_baselines/a2c/__init__.py', 'stable_baselines/deepq/__init__.py'):
        return '# Only utility modules are used by this DDPG experiment.\n'
    if path == 'stable_baselines/common/__init__.py':
        text = text.replace('from stable_baselines.common.cmd_util import make_vec_env\n', '')
    if path == 'stable_baselines/common/vec_env/__init__.py':
        text = text.replace('from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder\n', '')
    # Explicit scalar extraction is numerically identical and supports new NumPy.
    # The pinned legacy runtime also runs an unmodified-solver comparison test.
    if scalar_compat and path == 'ctm_envs/envs/exact_model.py':
        text = text.replace('np.empty([self.num_tubes, 1])', 'np.empty(self.num_tubes)')
        text = text.replace('dr = np.dot(R, e3)', 'dr = np.dot(R, e3).ravel()')
        text = text.replace('np.arange(a, c)', 'np.arange(a.item(), c.item())')
        text = text.replace('np.arange(b, c)', 'np.arange(b.item(), c.item())')
    return text


def build():
    changes = []
    for item in manifest():
        source = SOURCES / item['path']
        if not source.exists() or blob_sha(source.read_bytes()) != item['git_blob_sha1']:
            raise RuntimeError('Run python -m icra2021.bootstrap first: ' + item['path'])
        rel = item['source_path']
        target = RUNTIME / item['path']
        data = source.read_bytes()
        if rel.endswith('.py'):
            data = patch_source(rel, data.decode('utf-8')).encode('utf-8')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if data != source.read_bytes():
            changes.append(item['path'])
    (RUNTIME / 'compatibility_changes.json').write_text(json.dumps(changes, indent=2) + '\n')


def activate():
    if not (RUNTIME / 'compatibility_changes.json').exists():
        raise RuntimeError('Run python -m icra2021.bootstrap before importing the archive.')
    for name in ('stable-baselines', 'gym-ctm-ros'):
        path = str(RUNTIME / name)
        if path not in sys.path:
            sys.path.insert(0, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    if not args.offline:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, manifest()))
    build()
    if args.checkpoint:
        path = ROOT / '_checkpoints' / 'author_500000.zip'
        if not path.exists():
            if args.offline:
                raise FileNotFoundError(str(path))
            with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != CHECKPOINT_SHA256:
                raise ValueError('Author checkpoint hash mismatch')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('Author checkpoint hash mismatch')
    print('Verified {} author source files; built Gymnasium compatibility copies.'.format(len(manifest())))


if __name__ == '__main__':
    main()
