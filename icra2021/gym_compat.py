"""Small explicit API bridge for archived code; the Gym package is not used."""
import types
import gymnasium as gymnasium
import numpy as np

Env = gymnasium.Env
GoalEnv = gymnasium.Env
Wrapper = gymnasium.Wrapper
spaces = types.SimpleNamespace(**{name: getattr(gymnasium.spaces, name) for name in
                                 ('Space', 'Dict', 'Tuple', 'Discrete', 'MultiDiscrete', 'MultiBinary')})
REGISTRY = {}


class LegacyBox(gymnasium.spaces.Box):
    """Gymnasium Box with legacy bounded-uniform sampling and repaired metadata.

    The author's observation bounds include low > high. They are never used
    to clip observations. Only their ordering is repaired here; the public
    environment replaces those bounds with valid bounds on actual values.
    """
    def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
        lo = np.asarray(np.minimum(low, high), dtype=dtype)
        hi = np.asarray(np.maximum(low, high), dtype=dtype)
        super().__init__(lo, hi, shape=shape, dtype=dtype)
        self.seed(seed)

    def seed(self, seed=None):
        # An explicit reproducibility choice: original seeds were not archived.
        self.legacy_random = np.random.RandomState(seed)
        return [seed]

    def sample(self, mask=None, **kwargs):
        if np.all(np.isfinite(self.low)) and np.all(np.isfinite(self.high)):
            return self.legacy_random.uniform(self.low, self.high, self.shape).astype(self.dtype)
        return super().sample(mask=mask)


spaces.Box = LegacyBox


class Logger:
    MIN_LEVEL = 30
    DISABLED = 50

    @classmethod
    def set_level(cls, level):
        cls.MIN_LEVEL = level


logger = Logger


def register(id, entry_point, kwargs=None, **unused):
    REGISTRY[id] = dict(entry_point=entry_point, kwargs=kwargs or {})


def make(*args, **kwargs):
    raise RuntimeError('Use icra2021.env.ArchivedCTREnv; legacy gym.make is not exposed.')
