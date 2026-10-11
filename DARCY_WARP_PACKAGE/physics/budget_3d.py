# SPDX-License-Identifier: AGPL-3.0-only
"""Boundary flux diagnostics for structured 3D finite-volume fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from DARCY_WARP_PACKAGE.physics.faces_3d import validate_reciprocal_faces


@dataclass(frozen=True)
class BoundaryInterfaceFluxPlan:
    """Prevalidated sparse boundary-interface flux evaluation plan."""

    shape: tuple[int, int, int]
    boundary_indices: np.ndarray
    neighbour_indices: np.ndarray
    conductance: np.ndarray

    def evaluate(self, head: np.ndarray) -> float:
        """Evaluate flow using only cells touching the selected boundary."""

        values = np.asarray(head, dtype=np.float64)
        if values.shape != self.shape:
            raise ValueError(
                f"head shape {values.shape} does not match plan shape {self.shape}."
            )
        flat = values.reshape(-1)
        boundary_head = flat[self.boundary_indices]
        neighbour_head = flat[self.neighbour_indices]
        if not np.all(np.isfinite(boundary_head)) or not np.all(
            np.isfinite(neighbour_head)
        ):
            raise ValueError("Boundary-interface heads must be finite.")
        return float(
            np.sum(
                self.conductance * (boundary_head - neighbour_head),
                dtype=np.float64,
            )
        )


def compute_3d_flow_budget(
    *,
    head: np.ndarray,
    source: np.ndarray,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    active: np.ndarray,
    bc_mask: np.ndarray,
    storage_diag: np.ndarray | None = None,
    head_prev: np.ndarray | None = None,
    return_field: bool = False,
) -> dict[str, float]:
    """Cellwise and domain water budget for the discrete 7-point system.

    Sign conventions (consistent with the solver kernels):
    positive ``source`` is injection/recharge [m^3/day]; the intercell flux
    across a face is ``C * (h_here - h_neighbour)`` (flow from higher head to
    lower head); the transient storage term is ``storage_diag * (h - h_prev)``
    (positive = uptake into storage).  All fluxes are volumetric [m^3/day].

    Returns aggregate statistics:
      - ``cell_residual_max`` / ``cell_residual_rms``: imbalance of the
        discrete balance equation per free cell [m^3/day],
      - ``source_total``: sum of sources over free cells,
      - ``boundary_inflow`` / ``boundary_outflow``: exchange with Dirichlet
        cells (positive = into the model),
      - ``storage_uptake``: rate of change of stored water (0 when steady),
      - ``domain_imbalance``: source + boundary net - storage uptake; equals
        the sum of cell residuals (exact telescoping of internal faces),
      - ``relative_imbalance``: normalised by the total throughput.
    """
    h = np.asarray(head, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    act = np.asarray(active, dtype=np.int32) != 0
    bcm = np.asarray(bc_mask, dtype=np.int32) != 0
    shape = h.shape
    if src.shape != shape or act.shape != shape or bcm.shape != shape:
        raise ValueError("head, source, active, bc_mask shapes must match.")
    tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = validate_reciprocal_faces(
        faces=faces, active=active, canonicalize=True)
    if any(value.shape != shape for value in (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m)):
        raise ValueError("face shapes must match head.")
    free = act & ~bcm
    n_free = int(np.count_nonzero(free))

    s = (
        np.zeros(shape, dtype=np.float64)
        if storage_diag is None
        else np.maximum(np.asarray(storage_diag, dtype=np.float64), 0.0)
    )
    if s.shape != shape:
        raise ValueError("storage_diag shape must match head.")
    if head_prev is None:
        hp = np.zeros(shape, dtype=np.float64)
    else:
        hp = np.asarray(head_prev, dtype=np.float64)
        if hp.shape != shape:
            raise ValueError("head_prev shape must match head.")

    # Net outflow divergence per cell: sum c*(h_self - h_nb) + storage*h.
    # (The Dirichlet exchange below reports the free<->bc face part of it.)
    div = s * (h - hp)
    bc_exchange = np.zeros(shape, dtype=np.float64)  # c * (h_bc - h_free), inflow +
    bc_inflow = np.zeros(shape, dtype=np.float64)
    bc_outflow = np.zeros(shape, dtype=np.float64)
    diag_c = np.zeros(shape, dtype=np.float64)
    for face, sl_self, sl_nb in (
        (tx_p, (slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
        (tx_m, (slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1))),
        (ty_p, (slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
        (ty_m, (slice(None), slice(1, None), slice(None)), (slice(None), slice(None, -1), slice(None))),
        (tz_p, (slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
        (tz_m, (slice(1, None), slice(None), slice(None)), (slice(None, -1), slice(None), slice(None))),
    ):
        c = face[sl_self]
        both_act = act[sl_self] & act[sl_nb]
        c = np.where(both_act, c, 0.0)
        diag_c[sl_self] += c
        div[sl_self] += c * (h[sl_self] - h[sl_nb])
        is_bc_nb = bcm[sl_nb] & act[sl_nb]
        exchange = np.where(is_bc_nb, c * (h[sl_nb] - h[sl_self]), 0.0)
        bc_exchange[sl_self] += exchange
        # Count gross face exchanges before opposing boundaries cancel at
        # one free cell. The net exchange alone understates throughput.
        bc_inflow[sl_self] += np.maximum(exchange, 0.0)
        bc_outflow[sl_self] += np.minimum(exchange, 0.0)

    # Cellwise balance residual: source + storage release - net outflow = 0.
    residual = src - div
    residual = np.where(free, residual, 0.0)

    source_total = float(np.sum(src[free]))
    storage_uptake = float(np.sum((s * (h - hp))[free]))
    bc_net = float(np.sum(bc_exchange[free]))
    bc_in = float(np.sum(bc_inflow[free]))
    bc_out = float(np.sum(bc_outflow[free]))
    domain_imbalance = float(np.sum(residual[free]))
    throughput = max(
        float(np.sum(np.abs(src[free]))) + abs(bc_in) + abs(bc_out)
        + float(np.sum(np.abs((s * (h - hp))[free]))),
        np.finfo(np.float64).tiny,
    )
    if n_free > 0:
        cell_residual_max = float(np.max(np.abs(residual[free])))
        cell_residual_rms = float(np.sqrt(np.mean(residual[free] ** 2)))
    else:
        cell_residual_max = 0.0
        cell_residual_rms = 0.0
    result = {
        "cell_residual_max": cell_residual_max,
        "cell_residual_rms": cell_residual_rms,
        "source_total": source_total,
        "boundary_inflow": bc_in,
        "boundary_outflow": bc_out,
        "boundary_net": bc_net,
        "storage_uptake": storage_uptake,
        "domain_imbalance": domain_imbalance,
        "relative_imbalance": abs(domain_imbalance) / throughput,
        "throughput": throughput,
        "n_free_cells": float(n_free),
    }
    if return_field:
        result["cell_residual_field"] = residual
        result["operator_diagonal"] = diag_c + s
    return result


def prepare_boundary_interface_flux(
    *,
    boundary_mask: np.ndarray,
    active: np.ndarray,
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
) -> BoundaryInterfaceFluxPlan:
    """Validate fixed fields once and gather boundary-touching faces."""

    active_raw = np.asarray(active)
    boundary_raw = np.asarray(boundary_mask)
    if active_raw.ndim != 3:
        raise ValueError(f"3D fields are required; got active shape {active_raw.shape}.")
    shape = tuple(int(value) for value in active_raw.shape)
    if boundary_raw.shape != shape:
        raise ValueError(
            f"boundary_mask shape {boundary_raw.shape} does not match {shape}."
        )
    if not np.all(np.isfinite(active_raw)) or not np.all(np.isfinite(boundary_raw)):
        raise ValueError("active and boundary_mask must contain finite values.")
    conductances = validate_reciprocal_faces(
        faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m), active=active_raw, canonicalize=True)
    if any(value.shape != shape for value in conductances):
        raise ValueError("All face-conductance shapes must match active.")
    if any(not np.all(np.isfinite(value)) for value in conductances):
        raise ValueError("face conductances must contain finite values.")
    if any(np.any(value < 0.0) for value in conductances):
        raise ValueError("face conductances must be non-negative.")

    act = active_raw != 0
    boundary = (boundary_raw != 0) & act
    other = act & ~boundary
    cell_indices = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    boundary_parts: list[np.ndarray] = []
    neighbour_parts: list[np.ndarray] = []
    conductance_parts: list[np.ndarray] = []

    def append_faces(
        face_mask: np.ndarray,
        boundary_slice: tuple[slice, slice, slice],
        neighbour_slice: tuple[slice, slice, slice],
        conductance_field: np.ndarray,
    ) -> None:
        boundary_parts.append(cell_indices[boundary_slice][face_mask])
        neighbour_parts.append(cell_indices[neighbour_slice][face_mask])
        conductance_parts.append(conductance_field[boundary_slice][face_mask])

    append_faces(
        boundary[:, :, :-1] & other[:, :, 1:],
        (slice(None), slice(None), slice(None, -1)),
        (slice(None), slice(None), slice(1, None)),
        conductances[0],
    )
    append_faces(
        boundary[:, :, 1:] & other[:, :, :-1],
        (slice(None), slice(None), slice(1, None)),
        (slice(None), slice(None), slice(None, -1)),
        conductances[1],
    )
    append_faces(
        boundary[:, :-1, :] & other[:, 1:, :],
        (slice(None), slice(None, -1), slice(None)),
        (slice(None), slice(1, None), slice(None)),
        conductances[2],
    )
    append_faces(
        boundary[:, 1:, :] & other[:, :-1, :],
        (slice(None), slice(1, None), slice(None)),
        (slice(None), slice(None, -1), slice(None)),
        conductances[3],
    )
    append_faces(
        boundary[:-1, :, :] & other[1:, :, :],
        (slice(None, -1), slice(None), slice(None)),
        (slice(1, None), slice(None), slice(None)),
        conductances[4],
    )
    append_faces(
        boundary[1:, :, :] & other[:-1, :, :],
        (slice(1, None), slice(None), slice(None)),
        (slice(None, -1), slice(None), slice(None)),
        conductances[5],
    )
    return BoundaryInterfaceFluxPlan(
        shape=shape,
        boundary_indices=np.concatenate(boundary_parts),
        neighbour_indices=np.concatenate(neighbour_parts),
        conductance=np.concatenate(conductance_parts),
    )


def boundary_interface_flux(
    head: np.ndarray,
    boundary_mask: np.ndarray,
    active: np.ndarray,
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
) -> float:
    """Return flow from selected boundary cells into other active cells.

    Each internal face is counted exactly once.  Positive values mean flow
    leaves the selected boundary cells and enters the rest of the active model.
    The function is agnostic to what the mask represents (river, coast, drain,
    or another named fixed-head boundary).
    """

    arrays = {
        "head": head,
        "boundary_mask": boundary_mask,
        "active": active,
        "tx_p": tx_p,
        "tx_m": tx_m,
        "ty_p": ty_p,
        "ty_m": ty_m,
        "tz_p": tz_p,
        "tz_m": tz_m,
    }
    shape = np.asarray(head).shape
    if len(shape) != 3:
        raise ValueError(f"3D fields are required; got head shape {shape}.")
    for name, value in arrays.items():
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} shape {np.asarray(value).shape} does not match {shape}.")

    h = np.asarray(head, dtype=np.float64)
    active_raw = np.asarray(active)
    boundary_raw = np.asarray(boundary_mask)
    if not np.all(np.isfinite(active_raw)) or not np.all(np.isfinite(boundary_raw)):
        raise ValueError("active and boundary_mask must contain finite values.")
    act = active_raw != 0
    bnd = (boundary_raw != 0) & act
    if not np.all(np.isfinite(h)):
        raise ValueError("head must contain finite values.")
    conductances = validate_reciprocal_faces(
        faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m), active=active_raw, canonicalize=True)
    if any(not np.all(np.isfinite(arr)) for arr in conductances):
        raise ValueError("face conductances must contain finite values.")
    if any(np.any(arr < 0.0) for arr in conductances):
        raise ValueError("face conductances must be non-negative.")
    tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = conductances
    other = act & ~bnd
    total = 0.0

    face = bnd[:, :, :-1] & other[:, :, 1:]
    total += float(np.sum(np.asarray(tx_p, dtype=np.float64)[:, :, :-1][face] * (h[:, :, :-1][face] - h[:, :, 1:][face])))
    face = bnd[:, :, 1:] & other[:, :, :-1]
    total += float(np.sum(np.asarray(tx_m, dtype=np.float64)[:, :, 1:][face] * (h[:, :, 1:][face] - h[:, :, :-1][face])))

    face = bnd[:, :-1, :] & other[:, 1:, :]
    total += float(np.sum(np.asarray(ty_p, dtype=np.float64)[:, :-1, :][face] * (h[:, :-1, :][face] - h[:, 1:, :][face])))
    face = bnd[:, 1:, :] & other[:, :-1, :]
    total += float(np.sum(np.asarray(ty_m, dtype=np.float64)[:, 1:, :][face] * (h[:, 1:, :][face] - h[:, :-1, :][face])))

    face = bnd[:-1, :, :] & other[1:, :, :]
    total += float(np.sum(np.asarray(tz_p, dtype=np.float64)[:-1, :, :][face] * (h[:-1, :, :][face] - h[1:, :, :][face])))
    face = bnd[1:, :, :] & other[:-1, :, :]
    total += float(np.sum(np.asarray(tz_m, dtype=np.float64)[1:, :, :][face] * (h[1:, :, :][face] - h[:-1, :, :][face])))
    return total


def named_boundary_interface_fluxes(
    head: np.ndarray,
    named_masks: dict[str, np.ndarray],
    active: np.ndarray,
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
) -> dict[str, float]:
    """Evaluate ``boundary_interface_flux`` for each named mask."""

    return {
        name: boundary_interface_flux(
            head=head,
            boundary_mask=mask,
            active=active,
            tx_p=tx_p,
            tx_m=tx_m,
            ty_p=ty_p,
            ty_m=ty_m,
            tz_p=tz_p,
            tz_m=tz_m,
        )
        for name, mask in named_masks.items()
    }
