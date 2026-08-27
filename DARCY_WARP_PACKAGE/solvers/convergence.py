# SPDX-License-Identifier: AGPL-3.0-only
"""Stable convergence-controller import points for 2D nonlinear backends."""

from __future__ import annotations

import numpy as np


def chebyshev_update_weights(
    order: int,
    lambda_min_fraction: float,
) -> tuple[float, ...]:
    """Build bounded nonlinear-Picard Chebyshev damping weights."""
    m = int(order)
    if m <= 0:
        return tuple()
    lam_hi = 1.0
    lam_lo = max(1.0e-12, min(float(lambda_min_fraction), 0.999999 * lam_hi))
    center = 0.5 * (lam_hi + lam_lo)
    radius = 0.5 * (lam_hi - lam_lo)
    weights: list[float] = []
    for index in range(1, m + 1):
        theta = np.pi * (2.0 * float(index) - 1.0) / (2.0 * float(m))
        denominator = max(center - radius * float(np.cos(theta)), 1.0e-12)
        weights.append(float(1.0 / denominator))
    return tuple(weights)


def chebyshev_relaxation_sequence(
    order: int,
    lambda_min: float,
    lambda_max: float,
) -> tuple[float, ...]:
    """Build Chebyshev weighted-Jacobi factors without changing ordering."""
    m = int(order)
    if m <= 0:
        return tuple()
    lam_hi = max(float(lambda_max), 1.0e-12)
    lam_lo = max(1.0e-12, min(float(lambda_min), 0.999999 * lam_hi))
    center = 0.5 * (lam_hi + lam_lo)
    radius = 0.5 * (lam_hi - lam_lo)
    if radius <= 0.0:
        return tuple(float(1.0 / center) for _ in range(m))
    factors: list[float] = []
    for index in range(1, m + 1):
        theta = np.pi * (2.0 * float(index) - 1.0) / (2.0 * float(m))
        denominator = max(center - radius * float(np.cos(theta)), 1.0e-12)
        factors.append(float(1.0 / denominator))
    return tuple(factors)


_NAMES = {
    "AdaptiveInnerSolveConfig", "AdaptiveInnerSolveState",
    "_adaptive_dt_dh_contraction_estimate", "_adaptive_dt_projected_outer_to_tol",
    "_adaptive_dt_should_early_shrink", "_adaptive_dt_should_extend_budget",
    "_adaptive_practical_acceptance_allowed", "_build_adaptive_inner_solve_config_from_controls",
    "_run_adaptive_inner_kcycle_blocks",
}


def __getattr__(name: str):
    if name not in _NAMES:
        raise AttributeError(name)
    from DARCY_WARP_PACKAGE import warped_darcy

    return getattr(warped_darcy, name)
