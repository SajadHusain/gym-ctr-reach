"""DOP853 with a scale-safe version of its unchanged error-norm formula.

SciPy's squared error norms can underflow around an aligned straight solution.
In particular, a subnormal third-order norm times 0.01 can become zero while
the fifth-order norm is already zero. This produces 0/0 in the stock estimate.
The fallback below factors out a common scale before squaring. It does not
change the ODE, method order, tolerances, or boundary/branch acceptance tests.
"""
import numpy as np
from scipy.integrate import DOP853


class ScaleSafeDOP853(DOP853):
    # Protected SciPy extension point: regression tests compare with stock
    # DOP853 in the normal range and exercise subnormal and large errors.
    def _estimate_error_norm(self, K, h, scale):
        fifth = K.T @ self.E5 / scale
        third = K.T @ self.E3 / scale
        peak = max(float(np.max(np.abs(fifth))), float(np.max(np.abs(third))))
        if not np.isfinite(peak):
            raise FloatingPointError("Nonfinite DOP853 error estimate")
        if peak == 0:
            return 0.0
        # Use SciPy unchanged where its squared norms are safely representable.
        if 1e-140 < peak < 1e140:
            return super()._estimate_error_norm(K, h, scale)
        a = np.sum((fifth / peak) ** 2)
        b = np.sum((third / peak) ** 2)
        return float((abs(h) * peak) * (a / np.sqrt(a + 0.01*b)) / np.sqrt(len(scale)))
