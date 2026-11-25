# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stable convergence-controller import points for 2D nonlinear backends."""

from __future__ import annotations

import numpy as np


def chebyshev_update_weights(
    order: int,
    lambda_min_fraction: float,
) -> tuple[float, ...]:
    """Build bounded nonlinear-Picard Chebyshev damping weights.

    :param order: Number of weights; nonpositive order returns an empty tuple.
    :param lambda_min_fraction: Lower spectral bound relative to the unit upper bound.
    :returns: Ordered reciprocal Chebyshev roots, used as damping weights.
    """
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
    """Build the weighted-Jacobi Chebyshev relaxation factors.

    :param order: Number of weights; nonpositive order returns an empty tuple.
    :param lambda_min: Lower spectral bound of the preconditioned operator.
    :param lambda_max: Upper spectral bound of the preconditioned operator.
    :returns: Ordered positive relaxation factors.
    """
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
    """Resolve adaptive controller symbols without eager Warp imports.

    :param name: Requested module attribute.
    :returns: Named adaptive controller object.
    :raises AttributeError: If the requested symbol is not exported.
    """
    if name not in _NAMES:
        raise AttributeError(name)
    from DARCY_WARP_PACKAGE import warped_darcy

    return getattr(warped_darcy, name)


def steady_flow_tolerance(*, initial_rms, relative_tolerance, absolute_tolerance, fine, free):
    """Set a flow-residual gate without accepting inaccurate small-source starts.

    :param initial_rms: Initial flow residual RMS.
    :param relative_tolerance: Fraction of the initial residual.
    :param absolute_tolerance: Requested absolute residual floor.
    :param fine: Fine-grid level with its device inverse diagonal.
    :param free: Boolean free-cell mask for the fine grid.
    :returns: Absolute tolerance after adjusting a below-floor start using
        the largest free-cell inverse diagonal.
    """
    if 0.0 < initial_rms <= absolute_tolerance:
        inverse_diagonal = float(np.max(fine.M_inv_wp.numpy()[free]))
        absolute_tolerance = absolute_tolerance / max(1.0, inverse_diagonal)
    return max(absolute_tolerance, relative_tolerance * initial_rms)
