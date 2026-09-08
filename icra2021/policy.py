"""Read numeric archived policy weights without executing serialized objects."""
import io
import json
import zipfile
from collections import OrderedDict
import numpy as np


def read_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read('data'))
        names = json.loads(archive.read('parameter_list'))
        with np.load(io.BytesIO(archive.read('parameters')), allow_pickle=False) as data:
            parameters = OrderedDict((name, data[name].copy()) for name in names)
    return metadata, parameters


def flatten(obs):
    return np.concatenate([obs[key] for key in ('observation', 'achieved_goal', 'desired_goal')])


class NumpyActor:
    def __init__(self, path, action_space):
        self.metadata, self.weights = read_checkpoint(path)
        if self.metadata.get('policy_kwargs') != {'layers': [128, 128, 128]}:
            raise ValueError('Not an archived 3x128 checkpoint')
        if self.weights['model/pi/fc0/kernel:0'].shape != (19, 128):
            raise ValueError('Expected 19 policy inputs')
        self.low, self.high = action_space.low, action_space.high

    def predict(self, obs):
        x = np.asarray(flatten(obs) if isinstance(obs, dict) else obs, dtype=np.float32)
        w = self.weights
        if self.metadata['normalize_observations']:
            prefix = 'input/obs_rms/'
            count = w[prefix + 'count:0']
            mean = (w[prefix + 'runningsum:0'] / count).astype(np.float32)
            variance = (w[prefix + 'runningsumsq:0'] / count).astype(np.float32) - np.square(mean)
            std = np.sqrt(np.maximum(variance, np.float32(.01)))
            x = np.clip((x - mean) / std, *self.metadata.get('observation_range', [-5, 5]))
        for layer in range(3):
            p = 'model/pi/fc{}/'.format(layer)
            x = np.maximum(x @ w[p + 'kernel:0'] + w[p + 'bias:0'], 0)
        x = np.tanh(x @ w['model/pi/pi/kernel:0'] + w['model/pi/pi/bias:0'])
        return self.low + .5 * (x + 1.) * (self.high - self.low)
