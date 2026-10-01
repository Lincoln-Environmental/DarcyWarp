"""Accepted RIV outflow using the authoritative nonlinear device relation."""
from __future__ import annotations

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT, accepted_riv_flux_kernel


def riv_discharge_2d(*, head, stage, bottom, conductance, mask, device="cpu"):
    """Return signed cell-volume rates; positive means aquifer -> boundary.

    This CPU/CUDA extraction launches the same RIV function as the residual.
    The caller must supply an accepted nonlinear head and sanitized mask.
    """
    h = np.asarray(head)
    if h.ndim != 2:
        raise ValueError("RIV head must be two dimensional")
    fields = [np.asarray(field) for field in (h, stage, bottom, conductance, mask)]
    if any(field.shape != h.shape for field in fields):
        raise ValueError("RIV fields must have matching shapes")
    if any(not np.all(np.isfinite(field)) for field in fields):
        raise ValueError("RIV fields must be finite")
    if np.any(fields[3] < 0.0):
        raise ValueError("RIV conductance must be non-negative")
    inputs = [wp.array(field, dtype=WP_FLOAT, device=device) for field in fields[:4]]
    inputs.append(wp.array(fields[4], dtype=wp.int32, device=device))
    output = wp.zeros(h.shape, dtype=WP_FLOAT, device=device)
    wp.launch(accepted_riv_flux_kernel, dim=h.shape, inputs=inputs+[output], device=device)
    return output.numpy()
