# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reusable operator/session state for structured 3D solves.

The session owns validated host-side operator data and stable update/version
semantics. Both backends share this case/session contract; the fast backend
keeps GPU buffer and graph identities across operator-value changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from DARCY_WARP_PACKAGE.physics.faces_3d import validate_reciprocal_faces
from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask, floating_field
from DARCY_WARP_PACKAGE.physics.ghb_3d import validate_ghb_fields


class ThreeDOperatorSession:
    """Persistent 3D operator inputs with in-place RHS/face updates."""

    def __init__(
        self,
        *,
        faces: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        active: np.ndarray,
        bc_mask: np.ndarray,
        bc_values: np.ndarray,
        rhs: np.ndarray,
        initial_head: np.ndarray | None,
        dx: float,
        dy: float,
        dz: float,
        z_interfaces: np.ndarray | None = None,
        ghb_conductance=None,
        ghb_head=None,
        ghb_mask=None,
    ) -> None:
        self._closed = False
        self.revision = 0
        self.operator_revision = 0
        self.topology_revision = 0
        self.dx = float(dx)
        self.dy = float(dy)
        self.z_interfaces = None if z_interfaces is None else np.asarray(z_interfaces,dtype=float).copy()
        self.layer_geometry = None
        self.dz = float(dz) if z_interfaces is None else None
        values = tuple(floating_field(value=face, name="face") for face in faces)
        if len(values) != 6:
            raise ValueError("faces must contain exactly six conductance arrays.")
        self.faces = values
        self.active = binary_mask(value=active, name="active")
        self.bc_mask = binary_mask(value=bc_mask, name="bc_mask")
        self.bc_values = floating_field(value=bc_values, name="bc_values")
        self.rhs = floating_field(value=rhs, name="rhs")
        self.initial_head = None if initial_head is None else floating_field(value=initial_head, name="initial_head")
        self._fast_workspace_holder: dict[str, Any] = {}
        self._npf662_workspace_holder: dict[str, Any] = {}
        self._validate()
        self.ghb_conductance, self.ghb_head = validate_ghb_fields(
            conductance=ghb_conductance, head=ghb_head, mask=ghb_mask,
            active=self.active, fixed=self.bc_mask)
        self.faces = validate_reciprocal_faces(faces=self.faces, active=self.active, canonicalize=True)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.active.shape)

    def _validate(self) -> None:
        if self.active.ndim != 3:
            raise ValueError("3D session fields must have shape (nz, ny, nx).")
        shape = self.shape
        if len(self.faces) != 6:
            raise ValueError("faces must contain exactly six conductance arrays.")
        for name, array in (
            ("faces", self.faces),
            ("bc_mask", (self.bc_mask,)),
            ("bc_values", (self.bc_values,)),
            ("rhs", (self.rhs,)),
        ):
            for item in array:
                if item.shape != shape:
                    raise ValueError(f"{name} field shape {item.shape} does not match {shape}.")
        if self.initial_head is not None and self.initial_head.shape != shape:
            raise ValueError("initial_head shape does not match the operator shape.")
        for name, value in (("faces", self.faces), ("bc_values", (self.bc_values,)), ("rhs", (self.rhs,))):
            if any(not np.all(np.isfinite(item)) for item in value):
                raise ValueError(f"{name} must contain finite values.")
        if any(np.any(item < 0.0) for item in self.faces):
            raise ValueError("face conductances must be non-negative.")
        for name, mask in (("active", self.active), ("bc_mask", self.bc_mask)):
            if not np.all(np.isfinite(mask)) or np.any((mask != 0) & (mask != 1)):
                raise ValueError(f"{name} must be a finite binary mask.")
        if np.any((self.bc_mask != 0) & (self.active == 0)):
            raise ValueError("bc_mask may only select active cells.")
        if self.initial_head is not None and not np.all(np.isfinite(self.initial_head)):
            raise ValueError("initial_head must contain finite values.")
        spacing=(self.dx,self.dy,self.dz) if self.z_interfaces is None else (self.dx,self.dy)
        if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in spacing):
            raise ValueError("dx, dy, and dz must be positive finite values.")
        if self.z_interfaces is not None:
            from DARCY_WARP_PACKAGE.physics.npf662_layers import layered_geometry
            self.layer_geometry=layered_geometry(z_interfaces=self.z_interfaces,shape=shape,dx=self.dx,dy=self.dy)

    def update_rhs(self, rhs: np.ndarray) -> None:
        self._ensure_open()
        value = floating_field(value=rhs, name="rhs", dtype=self.rhs.dtype)
        if value.shape != self.shape:
            raise ValueError(f"rhs shape {value.shape} does not match {self.shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError("rhs must contain finite values.")
        self.rhs[...] = value
        self.revision += 1

    def update_initial_head(self, initial_head: np.ndarray | None) -> None:
        self._ensure_open()
        if initial_head is None:
            self.initial_head = None
        else:
            value = floating_field(value=initial_head, name="initial_head")
            if value.shape != self.shape:
                raise ValueError(f"initial_head shape {value.shape} does not match {self.shape}.")
            if not np.all(np.isfinite(value)):
                raise ValueError("initial_head must contain finite values.")
            self.initial_head = value.copy()
        self.revision += 1

    def update_boundary_values(self, bc_values: np.ndarray) -> None:
        """Update Dirichlet values without changing the boundary topology."""

        self._ensure_open()
        value = floating_field(value=bc_values, name="bc_values", dtype=self.bc_values.dtype)
        if value.shape != self.shape:
            raise ValueError(f"bc_values shape {value.shape} does not match {self.shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError("bc_values must contain finite values.")
        self.bc_values[...] = value
        self.operator_revision += 1
        self.revision += 1

    def update_ghb(self, *, ghb_conductance, ghb_head, ghb_mask=None):
        """Replace cell GHB values atomically; preserve device buffer ownership."""
        self._ensure_open()
        conductance, head = validate_ghb_fields(conductance=ghb_conductance,
            head=ghb_head, mask=ghb_mask, active=self.active, fixed=self.bc_mask)
        if conductance is None:
            raise ValueError("Use zero conductance to disable GHB cells")
        previous = np.zeros(self.shape, dtype=bool) if self.ghb_conductance is None else self.ghb_conductance > 0.0
        topology_changed = not np.array_equal(previous, conductance > 0.0)
        if self.ghb_conductance is None:
            self.ghb_conductance, self.ghb_head = conductance, head
        else:
            self.ghb_conductance[...] = conductance
            self.ghb_head[...] = head
        self.operator_revision += 1
        self.topology_revision += int(topology_changed)
        self.revision += 1

    def update_face_conductance(
        self,
        faces: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self._ensure_open()
        values = tuple(floating_field(value=face, name="face", dtype=old.dtype)
                       for face, old in zip(faces, self.faces))
        if len(faces) != 6:
            raise ValueError("faces must contain exactly six arrays")
        if len(values) != 6 or any(value.shape != self.shape for value in values):
            raise ValueError("faces must contain six arrays matching the session shape.")
        if any(not np.all(np.isfinite(value)) for value in values):
            raise ValueError("faces must contain finite values.")
        if any(np.any((np.asarray(raw) > 0.0) & (stored == 0.0)) for raw, stored in zip(faces, values)):
            raise ValueError("positive face conductances must remain representable in the stored dtype.")
        if any(np.any(np.asarray(raw) < 0.0) for raw in faces):
            raise ValueError("face conductances must be non-negative.")
        values = validate_reciprocal_faces(faces=values, active=self.active, canonicalize=True)
        topology_changed = any(not np.array_equal(old > 0.0, value > 0.0)
                               for old, value in zip(self.faces, values))
        for old, value in zip(self.faces, values):
            old[...] = value
        self.operator_revision += 1
        self.topology_revision += int(topology_changed)
        if topology_changed:
            self._fast_workspace_holder.clear()
        self.revision += 1

    def solve(self, *, solver: str, device: str, implementation: str = "classic", **kwargs: Any):
        """Solve using the current operator state and selected 3D backend."""
        self._ensure_open()
        if self.z_interfaces is not None:
            raise ValueError('Layered NPF uses its diagonal-preconditioned GMRES path, not the legacy session backend')
        from DARCY_WARP_PACKAGE.solvers_3d import (
            solve_chebyshev_7point_3d,
            solve_multigrid_kcycle_7point_3d,
        )

        solver_name = str(solver).strip().lower()
        if solver_name not in {"kcycle", "chebyshev"}:
            raise ValueError("solver must be 'kcycle' or 'chebyshev'.")
        backend = solve_multigrid_kcycle_7point_3d if solver_name == "kcycle" else solve_chebyshev_7point_3d
        solve_kwargs = dict(kwargs)
        if self.ghb_conductance is not None:
            solve_kwargs.update(ghb_conductance=self.ghb_conductance, ghb_head=self.ghb_head)
        # ``initial_head`` is part of the persistent state but transient calls
        # may intentionally override it (for example, a warm start from the
        # previous period).  Let the explicit call value win without passing a
        # duplicate keyword to the backend.
        initial_head = solve_kwargs.pop("initial_head", self.initial_head)
        return_info = bool(solve_kwargs.pop("return_info", True))
        implementation_mode = str(implementation).strip().lower()
        if implementation_mode not in {"classic", "fast"}:
            raise ValueError("implementation must be 'classic' or 'fast'.")
        if implementation_mode == "fast":
            if solver_name != "kcycle":
                raise ValueError("implementation='fast' currently supports solver='kcycle' only.")
            if bool(solve_kwargs.pop("unconfined", False)):
                from DARCY_WARP_PACKAGE.solvers.fast_3d import solve_fast_unconfined_kcycle_3d

                result = solve_fast_unconfined_kcycle_3d(
                    tx_p=self.faces[0], tx_m=self.faces[1], ty_p=self.faces[2], ty_m=self.faces[3],
                    tz_p=self.faces[4], tz_m=self.faces[5], rhs=self.rhs, active=self.active,
                    bc_mask=self.bc_mask, bc_values=self.bc_values, initial_head=initial_head,
                    dx=self.dx, dy=self.dy, dz=self.dz, return_info=return_info, device=device,
                    _workspace_holder=self._fast_workspace_holder,
                    **solve_kwargs,
                )
            else:
                from DARCY_WARP_PACKAGE.solvers.fast_3d import solve_fast_confined_kcycle_3d

                result = solve_fast_confined_kcycle_3d(
                    tx_p=self.faces[0], tx_m=self.faces[1], ty_p=self.faces[2], ty_m=self.faces[3],
                    tz_p=self.faces[4], tz_m=self.faces[5], rhs=self.rhs, active=self.active,
                    bc_mask=self.bc_mask, bc_values=self.bc_values, initial_head=initial_head,
                    dx=self.dx, dy=self.dy, dz=self.dz, return_info=return_info, device=device,
                    _workspace_holder=self._fast_workspace_holder,
                    **solve_kwargs,
                )
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                result[1].setdefault("session_revision", int(self.revision))
                result[1].setdefault("operator_revision", int(self.operator_revision))
                result[1].setdefault("topology_revision", int(self.topology_revision))
                result[1].setdefault("session_backend", "fast_3d")
            return result
        result = backend(
            tx_p=self.faces[0],
            tx_m=self.faces[1],
            ty_p=self.faces[2],
            ty_m=self.faces[3],
            tz_p=self.faces[4],
            tz_m=self.faces[5],
            rhs=self.rhs,
            active=self.active,
            bc_mask=self.bc_mask,
            bc_values=self.bc_values,
            initial_head=initial_head,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            device=device,
            return_info=return_info,
            **solve_kwargs,
        )
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            result[1].setdefault("session_revision", int(self.revision))
            result[1].setdefault("session_backend", "classic_3d")
        return result

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ThreeDOperatorSession is closed.")

    def close(self) -> None:
        self._closed = True
        self.faces = tuple()
        self.active = np.empty((0, 0, 0), dtype=np.int32)
        self.bc_mask = np.empty((0, 0, 0), dtype=np.int32)
        self.bc_values = np.empty((0, 0, 0))
        self.rhs = np.empty((0, 0, 0))
        self.initial_head = None
        self.ghb_conductance = self.ghb_head = None
        self.z_interfaces = None
        self.layer_geometry = None
        for record in self._fast_workspace_holder.values():
            if isinstance(record,dict) and record.get('linear_fgmres') is not None:
                record['linear_fgmres'].close()
        self._fast_workspace_holder.clear()
        workspace=self._npf662_workspace_holder.get('workspace')
        if workspace is not None:workspace.close()
        self._npf662_workspace_holder.clear()
