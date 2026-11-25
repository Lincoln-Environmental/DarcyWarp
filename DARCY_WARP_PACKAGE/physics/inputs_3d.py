# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate public 3D inputs before any narrowing conversion."""
import numpy as np


def binary_mask(*, value, name, shape=None):
    """Validate a binary mask before integer conversion.

    :param value: Boolean or numeric input containing only exact zero and one.
    :param name: Field name used in validation errors.
    :param shape: Optional required array shape.
    :returns: Owned int32 mask.
    :raises ValueError: If values are non-binary, nonfinite or incorrectly shaped.
    """
    raw = np.asarray(value)
    if shape is not None and raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if raw.dtype.kind not in "buif" or not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must contain finite, exact binary 0/1 values")
    return raw.astype(np.int32, copy=True)


def floating_field(*, value, name, dtype=np.float64):
    """Convert a finite real field without silently overflowing the target type.

    :param value: Real numeric scalar or array.
    :param name: Field name used in validation errors.
    :param dtype: Destination NumPy floating-point type.
    :returns: Owned array of the requested dtype.
    :raises ValueError: If the source or converted field is nonfinite or nonnumeric.
    """
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be a finite real numeric field")
    with np.errstate(over="ignore", invalid="ignore"):
        field = np.array(raw, dtype=dtype, copy=True)
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite in the destination dtype")
    return field


def positive_integer(*, value, name, minimum=1):
    """Validate an exact scalar integer control.

    :param value: Numeric scalar to validate.
    :param name: Control name used in validation errors.
    :param minimum: Smallest allowed integer.
    :returns: Validated Python integer.
    :raises ValueError: If the input is not a finite integer at least minimum.
    """
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or not np.isfinite(raw) or raw < minimum or int(raw) != raw:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(raw)
