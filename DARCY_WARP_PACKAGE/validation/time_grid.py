"""Floating-point comparisons for saved simulation endpoints."""

import numpy as np


def same_step_times(*, saved, expected):
    """Require the same number of strictly ordered, finite time endpoints."""
    actual = np.asarray(saved, dtype=np.float64)
    target = np.asarray(expected, dtype=np.float64)
    return bool(actual.ndim == 1 and actual.shape == target.shape and actual.size
        and np.isfinite(actual).all() and np.isfinite(target).all()
        and np.all(np.diff(actual) > 0.0) and np.all(np.diff(target) > 0.0)
        and np.allclose(actual, target, rtol=1e-12, atol=0.0))
