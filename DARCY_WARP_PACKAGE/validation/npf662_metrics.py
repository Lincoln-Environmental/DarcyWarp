"""Finite physical acceptance and stable FP64 diagnostic reductions."""
import math
import numpy as np


def final_tolerances(*, physical_tolerance=1e-12, relative_tolerance=1e-8,
                     head_tolerance_m=1e-6, boundary_tolerance_m=1e-10):
    values = dict(physical_rms_m3_day=physical_tolerance,
                  relative_cell_imbalance=relative_tolerance,
                  head_equivalent_rms_m=head_tolerance_m,
                  prescribed_head_error_m=boundary_tolerance_m)
    for name, value in values.items():
        if np.ndim(value) != 0 or not np.isfinite(value) or value < 0.:
            raise ValueError(f'{name} tolerance must be a finite nonnegative scalar')
    return {name: float(value) for name, value in values.items()}


def stable_sum(*, values):
    selected = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(selected)):
        return float('nan')
    scale = float(np.max(np.abs(selected))) if selected.size else 0.
    if scale == 0.:
        return 0.
    return scale * math.fsum(selected / scale)


def stable_rms(*, values):
    selected = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(selected)):
        return float('inf')
    scale = float(np.max(np.abs(selected))) if selected.size else 0.
    if scale == 0.:
        return 0.
    return scale * float(np.sqrt(np.mean((selected / scale)**2)))
