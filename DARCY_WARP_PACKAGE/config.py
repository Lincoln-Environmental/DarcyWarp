# SPDX-License-Identifier: AGPL-3.0-only
"""
Shared solver configuration.

This module centralises precision handling so that the 2D/3D solver modules
and kernel modules can share a single ``WP_FLOAT`` / ``NP_FLOAT`` pair without
circular imports.  ``WP_FLOAT`` is evaluated lazily so importing the package
does not require ``warp`` to be installed.
"""

from __future__ import annotations

import os

import numpy as np


_float_env = os.environ.get("DARCY_FLOAT", "float32")

__all__ = ["NP_FLOAT", "WP_FLOAT"]


def __getattr__(name: str):
    if name == "WP_FLOAT":
        import warp as wp

        if _float_env == "float64":
            return wp.float64
        if _float_env == "float32":
            return wp.float32
        raise ValueError("DARCY_FLOAT must be 'float32' or 'float64'")

    if name == "NP_FLOAT":
        if _float_env == "float64":
            return np.float64
        if _float_env == "float32":
            return np.float32
        raise ValueError("DARCY_FLOAT must be 'float32' or 'float64'")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
