# SPDX-License-Identifier: AGPL-3.0-or-later
"""Map areal groundwater recharge to the finite-volume cell source."""

import numpy as np

from DARCY_WARP_PACKAGE.physics.inputs_3d import floating_field


def recharge_depth_rate(*, R_field, horizontal_shape, dtype=np.float64):
    """Validate a signed surface flux and take ownership of its values.

    :param R_field: Scalar or horizontal array of recharge rates [L/T], or None.
    :param horizontal_shape: Grid shape (ny, nx).
    :param dtype: Floating-point storage type.
    :return: An owned scalar or horizontal array, or None for absent recharge.
    """
    if R_field is None:
        return None
    rate = floating_field(value=R_field, name="R_field", dtype=dtype)
    if rate.ndim != 0 and rate.shape != horizontal_shape:
        raise ValueError(f"R_field must be a scalar or have horizontal shape {horizontal_shape}")
    return rate


def surface_recharge_cells(active, bc_mask):
    """Locate the uppermost active cell in each column that can receive recharge.

    :param active: Validated 3D active-cell mask in top-to-bottom layer order.
    :param bc_mask: Validated fixed-head mask with the same shape.
    :return: C-order flat cell indices in horizontal row-major order. A column
        whose uppermost active cell is fixed receives no recharge.
    """
    nz, ny, nx = active.shape
    layers = np.argmax(active, axis=0)
    rows, columns = np.ogrid[:ny, :nx]
    receives = (active[layers, rows, columns] != 0) & (bc_mask[layers, rows, columns] == 0)
    rows, columns = np.nonzero(receives)
    return np.ravel_multi_index((layers[rows, columns], rows, columns), (nz, ny, nx))


def combine_cell_sources(*, rhs, R_field, active, bc_mask, dx, dy, dtype=np.float64):
    """Add signed areal recharge to any volumetric sources in the receiving cells.

    :param rhs: Volumetric cell rates [L³/T] of shape (nz, ny, nx), or None.
    :param R_field: Scalar or horizontal recharge rates [L/T], or None.
    :param active: Validated 3D active-cell mask, ordered from top to bottom.
    :param bc_mask: Validated 3D fixed-head mask.
    :param dx: East-west cell width [L].
    :param dy: North-south cell width [L].
    :param dtype: Floating-point storage type of the assembled source.
    :return: An owned 3D array of combined volumetric rates [L³/T].
    """
    if rhs is None:
        source = np.zeros(active.shape, dtype=dtype)
    else:
        source = floating_field(value=rhs, name="rhs", dtype=dtype)
        if source.shape != active.shape:
            raise ValueError(f"rhs shape {source.shape} does not match {active.shape}")
    rates = recharge_depth_rate(R_field=R_field, horizontal_shape=active.shape[1:])
    if rates is None or (rates.ndim == 0 and rates == 0.0):
        return source

    cells = surface_recharge_cells(active, bc_mask)
    columns = cells % (active.shape[1] * active.shape[2])
    selected_rates = np.broadcast_to(rates, active.shape[1:]).flat[columns]
    area = dx * dy
    with np.errstate(over="ignore", invalid="ignore"):
        source.flat[cells] += selected_rates * area
    if not np.all(np.isfinite(source.flat[cells])):
        raise ValueError("R_field produces non-finite volumetric cell rates")
    return source
