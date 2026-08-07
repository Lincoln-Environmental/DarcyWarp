# SPDX-License-Identifier: AGPL-3.0-only
"""Validated, solver-neutral inputs for structured 3D Darcy models.

The 3D solver already accepts ordinary NumPy fields.  ``Model3DInputs`` is a
small boundary between case construction and solver execution: case studies
can assemble and validate their fields without adding case-specific concepts
to :class:`WarpDarcySolver3D`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


FaceConductance = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _array(name: str, value: np.ndarray, shape: tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, copy=True)


def _binary_mask(name: str, value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} shape {raw.shape} does not match {shape}.")
    if not (np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    if not np.all(np.isfinite(raw.astype(np.float64, copy=False))):
        raise ValueError(f"{name} must contain only finite numeric values.")
    if not np.all(np.isin(raw, (0, 1))):
        raise ValueError(f"{name} must contain only 0/1 values.")
    return np.asarray(raw, dtype=np.int32).copy()


@dataclass(frozen=True)
class Model3DInputs:
    """Validated fields and grid information for one structured 3D model.

    Arrays use the solver's ``(nz, ny, nx)`` order.  Conductivity may be zero
    in inactive cells, but all values must be finite and non-negative.
    """

    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    active: np.ndarray
    bc_mask: np.ndarray
    bc_values: np.ndarray
    rhs: np.ndarray
    initial_head: np.ndarray | None
    dx: float
    dy: float
    dz: float
    named_masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kx = np.asarray(self.kx)
        if kx.ndim != 3:
            raise ValueError(f"3D fields must have three dimensions; got {kx.shape}.")
        shape = tuple(int(value) for value in kx.shape)
        if any(value <= 0 for value in shape):
            raise ValueError(f"3D fields must have positive dimensions; got {shape}.")

        for name, value in (("kx", self.kx), ("ky", self.ky), ("kz", self.kz)):
            conductivity = _array(name, value, shape, np.dtype(np.float64))
            if np.any(conductivity < 0.0):
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, conductivity)

        active = _binary_mask("active", self.active, shape)
        bc_mask = _binary_mask("bc_mask", self.bc_mask, shape)
        if np.any((bc_mask != 0) & (active == 0)):
            raise ValueError("bc_mask cannot select inactive cells.")
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "bc_mask", bc_mask)
        object.__setattr__(self, "bc_values", _array("bc_values", self.bc_values, shape, np.dtype(np.float64)))
        object.__setattr__(self, "rhs", _array("rhs", self.rhs, shape, np.dtype(np.float64)))
        if self.initial_head is not None:
            object.__setattr__(
                self,
                "initial_head",
                _array("initial_head", self.initial_head, shape, np.dtype(np.float64)),
            )

        for name, spacing in (("dx", self.dx), ("dy", self.dy), ("dz", self.dz)):
            spacing_value = float(spacing)
            if not np.isfinite(spacing_value) or spacing_value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0; got {spacing!r}.")
            object.__setattr__(self, name, spacing_value)

        masks: dict[str, np.ndarray] = {}
        for name, mask in dict(self.named_masks).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("named_masks keys must be non-empty strings.")
            mask_array = np.asarray(mask)
            if mask_array.shape != shape:
                raise ValueError(f"named mask {name!r} shape {mask_array.shape} does not match {shape}.")
            if not np.all(np.isfinite(mask_array)):
                raise ValueError(f"named mask {name!r} must contain finite values.")
            masks[name] = np.asarray(mask_array != 0, dtype=bool).copy()
        object.__setattr__(self, "named_masks", masks)

        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serialisable.") from exc
        object.__setattr__(self, "metadata", metadata)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape in ``(nz, ny, nx)`` order."""

        return tuple(int(value) for value in self.kx.shape)

    @property
    def nz(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nx(self) -> int:
        return self.shape[2]

    def build_solver(
        self,
        *,
        device: str = "cuda:0",
        solver: str = "kcycle",
        face_conductance: FaceConductance | None = None,
        **solver_kwargs: Any,
    ):
        """Build a ``WarpDarcySolver3D`` using its public build API.

        ``face_conductance`` is an optional precomputed six-array tuple.  It
        lets callers that need boundary fluxes reuse the exact same faces for
        solving and diagnostics; when omitted, faces are built from K fields by
        ``WarpDarcySolver3D.build_from_K_fields``.
        """

        from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

        instance = WarpDarcySolver3D(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            device=device,
            solver=solver,
            **solver_kwargs,
        )
        if face_conductance is None:
            instance.build_from_K_fields(
                kx_field=self.kx,
                ky_field=self.ky,
                kz_field=self.kz,
                active=self.active,
                bc_mask=self.bc_mask,
                bc_values=self.bc_values,
                rhs=self.rhs,
                initial_head=self.initial_head,
            )
        else:
            if len(face_conductance) != 6:
                raise ValueError("face_conductance must contain six arrays.")
            instance.build_from_face_conductance(
                tx_p=face_conductance[0],
                tx_m=face_conductance[1],
                ty_p=face_conductance[2],
                ty_m=face_conductance[3],
                tz_p=face_conductance[4],
                tz_m=face_conductance[5],
                active=self.active,
                bc_mask=self.bc_mask,
                bc_values=self.bc_values,
                rhs=self.rhs,
                initial_head=self.initial_head,
            )
        return instance
