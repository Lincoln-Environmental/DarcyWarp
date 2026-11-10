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
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be a finite real numeric field")
    with np.errstate(over="ignore", invalid="ignore"):
        field = np.array(raw, dtype=dtype, copy=True)
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite in the destination dtype")
    return field


def positive_integer(*, value, name, minimum=1):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or not np.isfinite(raw) or raw < minimum or int(raw) != raw:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(raw)
