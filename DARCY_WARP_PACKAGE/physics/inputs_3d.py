# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input checks performed before lossy casts or mutation of a live 3D model."""
from __future__ import annotations

import numpy as np


def positive_grid_size(value: int, *, name: str) -> int:
    """Accept positive integral numeric dimensions, never booleans or fractions."""
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(raw) or raw <= 0 or raw != np.floor(raw):
        raise ValueError(f"{name} must be a positive integer.")
    return int(raw)


def binary_mask(value, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Validate the original mask before int32 conversion can discard values."""
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a numeric binary mask with shape {shape}.")
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must be a finite binary mask containing only 0/1.")
    return np.array(raw, dtype=np.int32, copy=True)


def floating_field(value, *, name: str, shape: tuple[int, ...], dtype=None) -> np.ndarray:
    """Own a finite real field, promoting integers and checking the stored dtype."""
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real numeric field with shape {shape}.")
    target = np.result_type(raw.dtype, np.float32) if dtype is None else np.dtype(dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=target, copy=True)
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values representable in {target}.")
    return result
