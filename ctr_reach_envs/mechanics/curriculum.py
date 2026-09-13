"""Predetermined tolerance schedule used by the mechanics training arms."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ToleranceCurriculum:
    """Exponential tolerance schedule sampled at episode reset."""
    initial_m: float = .005
    final_m: float = .001
    decay_steps: int = 25_000

    def __post_init__(self):
        for value in (self.initial_m, self.final_m):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Curriculum tolerances must be positive and finite")
        if self.initial_m < self.final_m:
            raise ValueError("Initial tolerance must be at least the final tolerance")
        if isinstance(self.decay_steps, bool) or not isinstance(self.decay_steps, int) or self.decay_steps < 1:
            raise ValueError("Curriculum decay_steps must be a positive integer")

    def value(self, transitions: int) -> float:
        if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
            raise ValueError("Curriculum progress must be a nonnegative integer")
        if transitions >= self.decay_steps:
            return float(self.final_m)
        return float(self.initial_m * (self.final_m / self.initial_m) **
                     (transitions / self.decay_steps))
