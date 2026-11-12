# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell-based general-head exchange: outward flow C * (h - boundary head).

Conductance is an integrated volume/time/head coefficient, independent of
cell conductivity and saturation. One equivalent boundary is stored per cell.
"""
import numpy as np

from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask, floating_field


def validate_ghb_fields(*, conductance, head, active, fixed, mask=None):
    """Own finite full-grid inputs; validate before conversion or assignment."""
    shape = np.asarray(active).shape
    if conductance is None and head is None and mask is None:
        return None, None
    if conductance is None or head is None:
        raise ValueError("GHB requires both ghb_conductance and ghb_head")
    c = floating_field(value=conductance, name="ghb_conductance")
    hb = floating_field(value=head, name="ghb_head")
    if c.ndim not in (0, len(shape)) or hb.ndim not in (0, len(shape)):
        raise ValueError("GHB fields must be scalars or full-grid arrays")
    if (c.ndim and c.shape != shape) or (hb.ndim and hb.shape != shape):
        raise ValueError(f"GHB fields must have shape {shape}")
    if np.any(c < 0.0):
        raise ValueError("ghb_conductance must be nonnegative")
    c = np.broadcast_to(array=c, shape=shape).copy()
    hb = np.broadcast_to(array=hb, shape=shape).copy()
    selected = c > 0.0 if mask is None else binary_mask(value=mask, name="ghb_mask", shape=shape) != 0
    if np.any(selected & ((np.asarray(active) == 0) | (np.asarray(fixed) != 0))):
        raise ValueError("GHB cells must be active and cannot also have prescribed heads")
    if mask is not None and np.any((c != 0.0) & ~selected):
        # A scalar conductance with an explicit mask is a convenient uniform C.
        if np.asarray(conductance).ndim != 0:
            raise ValueError("Nonzero ghb_conductance outside ghb_mask would be lost")
    c[~selected] = 0.0
    hb[~selected] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        forcing = c * hb
    if not np.all(np.isfinite(forcing)):
        raise ValueError("GHB conductance times boundary head must be representable")
    return c, hb


def add_ghb_linear_terms(*, rhs, diagonal, conductance, head):
    """Add the GHB diagonal/forcing while retaining separate physical storage."""
    if conductance is None:
        return rhs, diagonal
    with np.errstate(over="ignore", invalid="ignore"):
        result_rhs = np.asarray(rhs) + conductance * head
        result_diagonal = np.asarray(diagonal) + conductance
    if not np.all(np.isfinite(result_rhs)) or not np.all(np.isfinite(result_diagonal)):
        raise ValueError("Assembled GHB forcing or diagonal is not representable")
    return result_rhs, result_diagonal


def ghb_flux(*, head, conductance, boundary_head):
    """Evaluate physical exchange directly, avoiding C*h minus C*h_boundary."""
    if conductance is None:
        return np.zeros_like(a=head, dtype=np.float64)
    result = np.zeros_like(a=conductance, dtype=np.float64)
    np.subtract(head, boundary_head, out=result, where=conductance > 0.0)
    result *= conductance
    return result
