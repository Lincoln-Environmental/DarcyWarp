# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate public 3D inputs before any narrowing conversion."""
import numpy as np


def binary_mask(*, value, name, shape=None):
    raw = np.asarray(value)
    if shape is not None and raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if raw.dtype.kind not in "buif" or not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must contain finite, exact binary 0/1 values")
    return raw.astype(np.int32, copy=True)


def floating_field(*, value, name, dtype=np.float64):
    with np.errstate(over="ignore", invalid="ignore"):
        field = np.array(value, dtype=dtype, copy=True)
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite in the destination dtype")
    return field


def positive_integer(*, value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or not np.isfinite(value) or value < minimum or int(value) != value:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)
