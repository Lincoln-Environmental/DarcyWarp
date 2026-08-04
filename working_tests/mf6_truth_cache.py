# SPDX-License-Identifier: AGPL-3.0-only
"""Shared MF6 truth-artifact cache loader for the mixed-precision campaign.

Import-light: does not pin DARCY_FLOAT, so it is safe for fp64/fp32/mixed
profiling processes.  Mirrors the cache format written by
``DARCY_WARP_PACKAGE.model_convergence_and_sanity_tests``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_cached_mf6_truth(truth_path: Path):
    """Return (heads, mf6_seconds) for a matching artifact, else None."""
    if not truth_path.exists():
        return None
    try:
        with np.load(truth_path, allow_pickle=False) as truth:
            if "heads" not in truth.files:
                return None
            heads = np.asarray(truth["heads"], dtype=np.float64)
            if heads.ndim != 2 or not np.all(np.isfinite(heads)):
                return None
            mf6_seconds = None
            if "mf6_seconds" in truth.files:
                candidate = float(truth["mf6_seconds"])
                if np.isfinite(candidate) and candidate >= 0.0:
                    mf6_seconds = candidate
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return heads, mf6_seconds
