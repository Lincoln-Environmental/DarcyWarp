# SPDX-License-Identifier: AGPL-3.0-or-later
"""Boundary flux diagnostics for structured 3D finite-volume fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from DARCY_WARP_PACKAGE.physics.faces_3d import validate_reciprocal_faces
from DARCY_WARP_PACKAGE.validation.npf662_metrics import stable_rms, stable_sum
from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask
from DARCY_WARP_PACKAGE.physics.ghb_3d import validate_ghb_fields, ghb_flux


def prepare_3d_budget_source(*, source, head_prev, extra_storage_diag):
    """Retain declared forcing without subtracting a large assembled RHS.

    An extra preassembled diagonal requires its own corresponding RHS; only
    that explicit extra term is removed to express a previous-head budget.
    Ordinary storage_coeff forcing is used directly, even when Q + S*h_prev
    loses Q to floating-point rounding.
    """
    result = np.asarray(source, dtype=np.float64)
    if head_prev is not None and extra_storage_diag is not None:
        result = result - np.asarray(extra_storage_diag, dtype=np.float64) * head_prev
    return result


def is_3d_budget_conserved(*, budget, absolute_tolerance, relative_tolerance=1e-3):
    """Finite dimensional RMS OR normalized L1, never overflow-based success.

    Both diagnostics and available budget aggregates must remain finite even
    when only one tolerance branch is needed. In particular, an overflowing
    throughput cannot turn a finite residual into an accepted zero ratio.
    """
    for name, value in (("absolute_tolerance", absolute_tolerance),
                        ("final_budget_relative_tol", relative_tolerance)):
        if np.ndim(value) != 0 or not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative scalar.")
    if not budget.get("physical_metrics_valid", True) or not budget.get("finite_physical_budget", True):
        return False
    required = ("cell_residual_rms", "relative_cell_imbalance")
    if any(name not in budget for name in required):
        return False
    nonnegative = required + ("cell_residual_max", "relative_imbalance", "throughput", "head_residual_rms")
    scalars = nonnegative + ("source_total", "boundary_inflow", "boundary_outflow",
                             "boundary_net", "storage_uptake", "domain_imbalance", "imbalance")
    for name in scalars:
        if name not in budget:
            continue
        value = budget[name]
        if value is None or np.ndim(value) != 0 or not np.isfinite(value):
            return False
        if name in nonnegative and value < 0.0:
            return False
    return bool(budget["cell_residual_rms"] <= absolute_tolerance
                or budget["relative_cell_imbalance"] <= relative_tolerance)


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
    ghb_conductance: np.ndarray | None = None,
    ghb_head: np.ndarray | None = None,
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
      - ``boundary_inflow`` / ``boundary_outflow``: combined CHD/GHB exchange
        (legacy signed convention: inflow positive, outflow negative),
      - ``chd_inflow/outflow`` and ``ghb_inflow/outflow``: positive magnitudes
        for the individual boundary packages; ``ghb_net`` is positive inward,
      - ``storage_uptake``: rate of change of stored water (0 when steady),
      - ``domain_imbalance``: source + boundary net - storage uptake; equals
        the sum of cell residuals (exact telescoping of internal faces),
      - ``relative_imbalance``: normalised by the total throughput.
    """
    h = np.asarray(head, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    act = binary_mask(value=active, name='active') != 0
    bcm = binary_mask(value=bc_mask, name='bc_mask') != 0
    shape = h.shape
    if src.shape != shape or act.shape != shape or bcm.shape != shape:
        raise ValueError("head, source, active, bc_mask shapes must match.")
    tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = validate_reciprocal_faces(
        faces=faces, active=active, canonicalize=True, copy_arrays=False)
    if any(value.shape != shape for value in (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m)):
        raise ValueError("face shapes must match head.")
    free = act & ~bcm
    n_free = int(np.count_nonzero(free))
    ghb_c, ghb_h = validate_ghb_fields(conductance=ghb_conductance, head=ghb_head,
                                      active=active, fixed=bc_mask)
    ghb_q = ghb_flux(head=h, conductance=ghb_c, boundary_head=ghb_h)

    s = (
        np.zeros(shape, dtype=np.float64)
        if storage_diag is None
        else np.asarray(storage_diag, dtype=np.float64)
    )
    if s.shape != shape:
        raise ValueError("storage_diag shape must match head.")
    if not np.all(np.isfinite(s)) or np.any(s < 0.):
        raise ValueError("storage_diag must be finite and nonnegative")
    if head_prev is None:
        hp = np.zeros(shape, dtype=np.float64)
    else:
        hp = np.asarray(head_prev, dtype=np.float64)
        if hp.shape != shape:
            raise ValueError("head_prev shape must match head.")

    # Compute each physical face once, then add its equal/opposite cell fluxes.
    # Reciprocity and inactive connections have already been validated above.
    storage_change = s * (h - hp)
    div = storage_change + ghb_q
    diag_c = np.zeros(shape, dtype=np.float64) if ghb_c is None else ghb_c.copy()
    bc_in = 0.0
    bc_out = 0.0
    fluxes_finite = bool(np.all(np.isfinite(ghb_q)))
    for face, axis in ((tx_p, 2), (ty_p, 1), (tz_p, 0)):
        here, there = [slice(None)] * 3, [slice(None)] * 3
        here[axis], there[axis] = slice(None, -1), slice(1, None)
        here, there = tuple(here), tuple(there)
        c = face[here]
        q = np.zeros_like(c, dtype=np.float64)
        np.subtract(h[here], h[there], out=q, where=c > 0.0)
        q *= c
        fluxes_finite = fluxes_finite and bool(np.all(np.isfinite(q)))
        div[here] += q
        div[there] -= q
        diag_c[here] += c
        diag_c[there] += c
        # Gross exchanges are accumulated before opposing boundaries cancel.
        into_here = -q[free[here] & bcm[there]]
        into_there = q[bcm[here] & free[there]]
        exchange = np.concatenate((into_here, into_there))
        bc_in = stable_sum(values=[bc_in, stable_sum(values=np.maximum(exchange, 0.))])
        bc_out = stable_sum(values=[bc_out, stable_sum(values=np.minimum(exchange, 0.))])

    chd_in, chd_out = bc_in, bc_out
    ghb_in = stable_sum(values=np.maximum(-ghb_q[free], 0.0))
    ghb_out = stable_sum(values=np.minimum(-ghb_q[free], 0.0))
    bc_in = stable_sum(values=[chd_in, ghb_in])
    bc_out = stable_sum(values=[chd_out, ghb_out])
    residual = src - div
    residual[~free] = 0.0
    source_free = src[free]
    storage_free = storage_change[free]
    source_total = stable_sum(values=source_free)
    storage_uptake = stable_sum(values=storage_free)
    bc_net = bc_in + bc_out
    domain_imbalance = stable_sum(values=residual[free])
    throughput = max(
        stable_sum(values=[stable_sum(values=np.abs(source_free)), abs(bc_in), abs(bc_out),
                           stable_sum(values=np.abs(storage_free))]),
        np.finfo(np.float64).tiny,
    )
    if n_free > 0:
        cell_residual_max = float(np.max(np.abs(residual[free])))
        cell_residual_rms = stable_rms(values=residual[free])
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
        "chd_inflow": chd_in,
        "chd_outflow": -chd_out,
        "ghb_inflow": ghb_in,
        "ghb_outflow": -ghb_out,
        "ghb_net": ghb_in + ghb_out,
        "storage_uptake": storage_uptake,
        "domain_imbalance": domain_imbalance,
        "relative_imbalance": abs(domain_imbalance) / throughput,
        "relative_cell_imbalance": stable_sum(values=np.abs(residual[free])) / throughput,
        "throughput": throughput,
        "n_free_cells": float(n_free),
    }
    result["physical_metrics_valid"] = bool(fluxes_finite and all(np.all(np.isfinite(value)) for value in
        (h[act], src[act], hp[act], storage_change[act], div[act], residual[free], (diag_c+s)[act])))
    result["physical_metrics_valid"] = bool(result["physical_metrics_valid"] and
        all(np.isfinite(value) for value in result.values()))
    if not result["physical_metrics_valid"]:
        result["relative_cell_imbalance"] = float("inf")
        result["relative_imbalance"] = float("inf")
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
