"""SI tube parameters, exact segment events, and complete joint feasibility."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class TubeParameters:
    """Unloaded, isotropic-cross-section rod with zero intrinsic torsion.

    E and G are independent positive constitutive inputs. This class does not
    infer a Poisson ratio or certify that fitted inputs describe a real material.
    ``length`` is total tube length, not straight-section length.
    """

    length: float
    length_curved: float
    diameter_inner: float
    diameter_outer: float
    stiffness: float
    torsional_stiffness: float
    x_curvature: float
    y_curvature: float = 0.0

    def __post_init__(self):
        if not np.all(np.isfinite(list(vars(self).values()))):
            raise ValueError("All tube parameters must be finite")
        if not 0 <= self.length_curved <= self.length or self.length <= 0:
            raise ValueError("Require 0 <= curved length <= total length and positive total length")
        if not 0 <= self.diameter_inner < self.diameter_outer:
            raise ValueError("Require 0 <= inner diameter < outer diameter")
        if min(self.stiffness, self.torsional_stiffness) <= 0:
            raise ValueError("Constitutive stiffnesses must be positive")

    @property
    def EI(self):
        return self.stiffness * np.pi * (self.diameter_outer**4 - self.diameter_inner**4) / 64

    @property
    def GJ(self):
        return self.torsional_stiffness * np.pi * (self.diameter_outer**4 - self.diameter_inner**4) / 32

    @classmethod
    def from_legacy(cls, tube):
        return cls(tube.L, tube.L_c, tube.diameter_inner, tube.diameter_outer,
                   tube.E, tube.G, tube.U_x, tube.U_y)


class JointConstraints:
    """Euclidean projection onto all extension inequalities simultaneously.

    Tube order is inner/longest first. The feasible extension set is convex.
    A projection is found by enumerating its independent active faces (n <= 3),
    so it does not depend on sequential clipping or an optimizer's stopping rule.
    """

    def __init__(self, lengths, minimum_deployed=1e-3, constrain_alpha=False):
        self.lengths = np.asarray(lengths, dtype=float).copy()
        self.n = self.lengths.size
        if (self.lengths.ndim != 1 or not 1 <= self.n <= 3
                or not np.all(np.isfinite(self.lengths))
                or np.any(np.diff(self.lengths) > 0)):
            raise ValueError("Supply 1 to 3 finite tube lengths in non-increasing order")
        if not np.isfinite(minimum_deployed) or not 0 < minimum_deployed < self.lengths.min():
            raise ValueError("minimum_deployed must be positive and shorter than every tube")
        self.minimum_deployed = float(minimum_deployed)
        self.constrain_alpha = bool(constrain_alpha)
        eye = np.eye(self.n)
        ordering = eye[:-1] - eye[1:]
        self.A = np.vstack((eye, -eye, ordering, -ordering))
        self.b = np.r_[np.zeros(self.n), self.lengths - minimum_deployed,
                       np.zeros(self.n - 1), -np.diff(self.lengths)]
        projections, offsets = [eye], [np.zeros(self.n)]
        for count in range(1, self.n + 1):
            for active in combinations(range(len(self.b)), count):
                a = self.A[list(active)]
                if np.linalg.matrix_rank(a) < count:
                    continue
                k = np.linalg.solve(a @ a.T, a).T
                projections.append(eye - k @ a)
                offsets.append(k @ self.b[list(active)])
        self._projections = np.stack(projections)
        self._offsets = np.stack(offsets)

    def _array(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (2 * self.n,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Expected {2 * self.n} finite joint values")
        return q

    def margins(self, q):
        q = self._array(q)
        margins = self.b - self.A @ q[:self.n]
        if self.constrain_alpha:
            margins = np.r_[margins, np.pi - np.abs(q[self.n:])]
        return margins

    def is_feasible(self, q, atol=1e-12):
        return bool(np.all(self.margins(q) >= -atol))

    def project(self, q):
        q = self._array(q).copy()
        beta = q[:self.n]
        if np.any(self.A @ beta > self.b):
            candidates = np.einsum("ijk,k->ij", self._projections, beta) + self._offsets
            feasible = np.all(candidates @ self.A.T <= self.b + 1e-13, axis=1)
            if not np.any(feasible):
                raise RuntimeError("No numerically feasible extension projection")
            candidates = candidates[feasible]
            q[:self.n] = candidates[np.argmin(np.sum((candidates - beta)**2, axis=1))]
        if self.constrain_alpha:
            q[self.n:] = np.clip(q[self.n:], -np.pi, np.pi)
        if not self.is_feasible(q):
            raise RuntimeError("Projection failed its feasibility check")
        return q


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    active: np.ndarray
    curvature: np.ndarray


def segment_tubes(tubes, beta):
    """Split at deployed tube tips and curved-section starts without quantization.

    Only floating-point coincidences (8 eps * longest length) are coalesced.
    The region s < 0 is a straight guide and is handled analytically by the BVP.
    """
    lengths = np.array([t.length for t in tubes])
    beta = np.asarray(beta, dtype=float)
    if beta.shape != lengths.shape or not np.all(np.isfinite(beta)):
        raise ValueError("beta must match the tube count and be finite")
    ends = lengths + beta
    if np.any(ends <= 0) or np.any(beta > 1e-12):
        raise ValueError("Every tube must cross the template at s = 0")
    starts = ends - np.array([t.length_curved for t in tubes])
    curvature = np.array([[t.x_curvature, t.y_curvature] for t in tubes])
    candidates = np.sort(np.r_[0.0, ends, starts[(starts > 0) & (starts < ends.max())]])
    eps = 8 * np.finfo(float).eps * lengths.max()
    points = [float(candidates[0])]
    for point in candidates[1:]:
        if point - points[-1] > eps:
            points.append(float(point))
        else:
            points[-1] = float(max(points[-1], point))
    intervals = []
    for start, end in zip(points[:-1], points[1:]):
        midpoint = (start + end) / 2
        active = midpoint < ends
        intrinsic = curvature * (active & (midpoint >= starts))[:, None]
        intervals.append(Interval(start, end, active, intrinsic))
    return tuple(intervals)
