# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference DRN/RIV physics for structured 3D groundwater cells.

Rates are positive out of the aquifer. Conductance is integrated [L2/T].
The two packages retain separate fluxes even when they occupy the same cell.
"""
from dataclasses import dataclass

import numpy as np

from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask, floating_field
from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values


@dataclass(frozen=True)
class BoundaryTerms3D:
    """Physical cell discharge and its piecewise-linear operator terms.

    :param discharge: Signed aquifer outflow [L3/T].
    :param diagonal: Derivative of outward discharge with respect to head [L2/T].
    :param rhs: Boundary forcing [L3/T] added to the groundwater RHS.
    """

    discharge: np.ndarray
    diagonal: np.ndarray
    rhs: np.ndarray


def _owned_fields(*, package, conductance, levels, active, fixed, mask):
    """Validate and own one integrated boundary of each type per cell.

    :param package: Package name used in validation errors.
    :param conductance: Scalar or full-grid integrated conductance.
    :param levels: Mapping of elevation field names to scalars or full grids.
    :param active: Binary active-cell grid.
    :param fixed: Binary prescribed-head grid.
    :param mask: Optional binary package mask.
    :return: Conductance and a dictionary of owned elevation fields.
    """
    shape = np.asarray(active).shape
    if len(shape) != 3 or any(size == 0 for size in shape):
        raise ValueError("Boundary fields require a non-empty (nz, ny, nx) grid")
    active = binary_mask(value=active, name="active", shape=shape)
    fixed = binary_mask(value=fixed, name="fixed", shape=shape)

    if conductance is None and all(value is None for value in levels.values()) and mask is None:
        return None, {name: None for name in levels}
    if conductance is None or any(value is None for value in levels.values()):
        raise ValueError(f"{package} requires conductance and all elevation fields")

    fields = {}
    for name, value in (("conductance", conductance), *levels.items()):
        field = floating_field(value=value, name=f"{package}_{name}")
        if field.ndim != 0 and field.shape != shape:
            raise ValueError(f"{package}_{name} must be scalar or have shape {shape}")
        fields[name] = np.broadcast_to(field, shape).copy()

    c = fields.pop("conductance")
    if np.any(c < 0.0):
        raise ValueError(f"{package} conductance must be nonnegative")
    selected = c > 0.0 if mask is None else binary_mask(
        value=mask, name=f"{package}_mask", shape=shape
    ) != 0
    if np.any(selected & ((active == 0) | (fixed != 0))):
        raise ValueError(f"{package} cells must be active and cannot have prescribed heads")
    if mask is not None and np.asarray(conductance).ndim != 0 and np.any((c != 0.0) & ~selected):
        raise ValueError(f"Nonzero {package} conductance outside mask would be lost")
    c[~selected] = 0.0
    for name, field in fields.items():
        field[~selected] = 0.0
        if np.any(invalid_head_values(head=field[c > 0.0])):
            raise ValueError(f"Invalid conducting {package} {name}")
    return c, fields


def validate_drn_fields(*, conductance, elevation, active, fixed, mask=None):
    """Validate and copy structured 3D drain inputs.

    :param conductance: Integrated drain conductance [L2/T], scalar or full grid.
    :param elevation: Drain activation elevation [L], scalar or full grid.
    :param active: Binary active-cell grid (nz, ny, nx).
    :param fixed: Binary prescribed-head grid.
    :param mask: Optional binary drain mask; zero conductance disables exchange.
    :return: Owned conductance and elevation arrays, or (None, None).
    """
    c, fields = _owned_fields(package="DRN", conductance=conductance,
                              levels={"elevation": elevation}, active=active,
                              fixed=fixed, mask=mask)
    e = fields["elevation"]
    if c is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            _require_finite(c * e)
    return c, e


def validate_riv_fields(*, conductance, stage, bottom, active, fixed, mask=None):
    """Validate and copy structured 3D river inputs.

    :param conductance: Integrated riverbed conductance [L2/T].
    :param stage: River water surface elevation [L].
    :param bottom: Bottom elevation of the riverbed layer [L].
    :param active: Binary active-cell grid (nz, ny, nx).
    :param fixed: Binary prescribed-head grid.
    :param mask: Optional binary river mask; zero conductance disables exchange.
    :return: Owned conductance, stage and bottom arrays, or three None values.
    """
    c, fields = _owned_fields(package="RIV", conductance=conductance,
                              levels={"stage": stage, "bottom": bottom},
                              active=active, fixed=fixed, mask=mask)
    s, b = fields["stage"], fields["bottom"]
    if c is not None:
        if np.any((c > 0.0) & (s < b)):
            raise ValueError("RIV stage must be at or above riverbed bottom")
        with np.errstate(over="ignore", invalid="ignore"):
            _require_finite(c * s, c * (s - b))
    return c, s, b


def _require_finite(*fields):
    """Reject unrepresentable boundary terms instead of silently clipping.

    :param fields: Arrays of assembled boundary quantities.
    """
    if any(not np.all(np.isfinite(field)) for field in fields):
        raise ValueError("Boundary discharge or operator terms are not representable")


def _empty_terms(*, head, conductance):
    """Allocate terms and identify conducting cells without reading inactive heads.

    :param head: Current groundwater head grid.
    :param conductance: Validated conductance array or None.
    :return: Head, conducting mask and zero-initialised boundary terms.
    """
    h = np.asarray(head, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError("Head must be a three-dimensional grid")
    enabled = np.zeros(h.shape, dtype=bool)
    if conductance is not None:
        if conductance.shape != h.shape:
            raise ValueError("Conductance must match head shape")
        enabled = conductance > 0.0
    if np.any(invalid_head_values(head=h[enabled])):
        raise ValueError("Invalid head in a conducting boundary cell")
    terms = BoundaryTerms3D(np.zeros_like(h), np.zeros_like(h), np.zeros_like(h))
    return h, enabled, terms


def drn_terms(*, head, conductance, elevation):
    """Evaluate DRN outflow and the exact active-branch diagonal/RHS.

    :param head: Current groundwater head [L], full 3D grid.
    :param conductance: Conductance from validate_drn_fields.
    :param elevation: Elevation from validate_drn_fields.
    :return: BoundaryTerms3D; zero derivative at the activation threshold.
    """
    h, enabled, terms = _empty_terms(head=head, conductance=conductance)
    if conductance is None:
        return terms
    flowing = enabled & (h > elevation)
    with np.errstate(over="ignore", invalid="ignore"):
        terms.discharge[flowing] = conductance[flowing] * (h[flowing] - elevation[flowing])
        terms.diagonal[flowing] = conductance[flowing]
        terms.rhs[flowing] = conductance[flowing] * elevation[flowing]
    _require_finite(terms.discharge, terms.diagonal, terms.rhs)
    return terms


def riv_terms(*, head, conductance, stage, bottom):
    """Evaluate RIV outflow, including constant leakage below the bed bottom.

    :param head: Current groundwater head [L], full 3D grid.
    :param conductance: Conductance from validate_riv_fields.
    :param stage: Stage from validate_riv_fields.
    :param bottom: Bottom from validate_riv_fields.
    :return: BoundaryTerms3D; zero derivative at and below the bottom.
    """
    h, enabled, terms = _empty_terms(head=head, conductance=conductance)
    if conductance is None:
        return terms
    coupled = enabled & (h > bottom)
    disconnected = enabled & ~coupled
    with np.errstate(over="ignore", invalid="ignore"):
        terms.discharge[enabled] = conductance[enabled] * (
            np.maximum(h[enabled], bottom[enabled]) - stage[enabled]
        )
        terms.diagonal[coupled] = conductance[coupled]
        terms.rhs[coupled] = conductance[coupled] * stage[coupled]
        terms.rhs[disconnected] = conductance[disconnected] * (
            stage[disconnected] - bottom[disconnected]
        )
    _require_finite(terms.discharge, terms.diagonal, terms.rhs)
    return terms
