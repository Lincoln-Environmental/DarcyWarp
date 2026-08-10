# SPDX-License-Identifier: AGPL-3.0-only
"""Boundary flux diagnostics for structured 3D finite-volume fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    conductances = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m)
    )
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
    conductances = tuple(np.asarray(arr, dtype=np.float64) for arr in (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m))
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
