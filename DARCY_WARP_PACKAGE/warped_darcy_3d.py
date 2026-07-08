# SPDX-License-Identifier: AGPL-3.0-only
"""
Standalone 3D Darcy solver wrapper.

This module exposes ``WarpDarcySolver3D``, a thin class around the 7-point
finite-volume solvers in :mod:`DARCY_WARP_PACKAGE.solvers_3d`.  It is kept
separate from the 2D solver so the 2D stencil and API remain untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from DARCY_WARP_PACKAGE.config import NP_FLOAT
from DARCY_WARP_PACKAGE.solvers_3d import (
    solve_chebyshev_7point_3d,
    solve_multigrid_kcycle_7point_3d,
)
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k


class WarpDarcySolver3D:
    """
    GPU 3D steady/confined Darcy solver on a structured (nz, ny, nx) grid.

    The solver uses a cell-centered 7-point finite-volume stencil with
    harmonic-mean face conductances.  It is a thin wrapper around the
    functions in :mod:`DARCY_WARP_PACKAGE.solvers_3d`.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float | None = None,
        dz: float = 1.0,
        device: str = "cuda:0",
        solver: str = "kcycle",
        diag_preconditioner_backend: str = "auto",
    ):
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("nx, ny, nz must be positive")
        if float(dx) <= 0.0:
            raise ValueError("dx must be > 0")
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.dx = float(dx)
        self.dy = float(dy) if dy is not None else self.dx
        self.dz = float(dz)
        self.device = str(device)
        self.solver = str(solver).lower()
        if self.solver not in {"kcycle", "chebyshev"}:
            raise ValueError("solver must be 'kcycle' or 'chebyshev'")
        backend_mode = str(diag_preconditioner_backend).strip().lower()
        if backend_mode not in {"auto", "host", "device"}:
            raise ValueError("diag_preconditioner_backend must be 'auto', 'host', or 'device'.")
        self.diag_preconditioner_backend = backend_mode

        self._tx_p: np.ndarray | None = None
        self._tx_m: np.ndarray | None = None
        self._ty_p: np.ndarray | None = None
        self._ty_m: np.ndarray | None = None
        self._tz_p: np.ndarray | None = None
        self._tz_m: np.ndarray | None = None
        self._rhs: np.ndarray | None = None
        self._active: np.ndarray | None = None
        self._bc_mask: np.ndarray | None = None
        self._bc_values: np.ndarray | None = None
        self._initial_head: np.ndarray | None = None
        self._kx_field: np.ndarray | None = None
        self._ky_field: np.ndarray | None = None
        self._kz_field: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape (nz, ny, nx)."""
        return (self.nz, self.ny, self.nx)

    def build_from_K_fields(
        self,
        kx_field: np.ndarray,
        ky_field: np.ndarray,
        kz_field: np.ndarray,
        active: np.ndarray,
        bc_mask: np.ndarray,
        bc_values: np.ndarray,
        rhs: np.ndarray,
        initial_head: np.ndarray | None = None,
    ) -> "WarpDarcySolver3D":
        """
        Build face conductances from cell-centered hydraulic conductivity fields.

        Parameters
        ----------
        kx_field, ky_field, kz_field
            Cell-centered K values, shape (nz, ny, nx).
        active
            Active-cell mask, shape (nz, ny, nx).
        bc_mask
            Dirichlet-cell mask, shape (nz, ny, nx).
        bc_values
            Prescribed heads on Dirichlet cells, shape (nz, ny, nx).
        rhs
            Right-hand side (recharge / source term), shape (nz, ny, nx).
        initial_head
            Optional initial guess, shape (nz, ny, nx).
        """
        tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = build_7point_face_conductance_from_k(
            kx_field=kx_field,
            ky_field=ky_field,
            kz_field=kz_field,
            active=active,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
        )
        self.build_from_face_conductance(
            tx_p=tx_p,
            tx_m=tx_m,
            ty_p=ty_p,
            ty_m=ty_m,
            tz_p=tz_p,
            tz_m=tz_m,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            rhs=rhs,
            initial_head=initial_head,
        )
        self._kx_field = np.asarray(kx_field, dtype=NP_FLOAT).copy()
        self._ky_field = np.asarray(ky_field, dtype=NP_FLOAT).copy()
        self._kz_field = np.asarray(kz_field, dtype=NP_FLOAT).copy()
        return self

    def build_from_face_conductance(
        self,
        tx_p: np.ndarray,
        tx_m: np.ndarray,
        ty_p: np.ndarray,
        ty_m: np.ndarray,
        tz_p: np.ndarray,
        tz_m: np.ndarray,
        active: np.ndarray,
        bc_mask: np.ndarray,
        bc_values: np.ndarray,
        rhs: np.ndarray,
        initial_head: np.ndarray | None = None,
    ) -> "WarpDarcySolver3D":
        """
        Build solver from pre-computed 7-point face conductances.
        """
        shape = self.shape
        for name, arr in (
            ("tx_p", tx_p),
            ("tx_m", tx_m),
            ("ty_p", ty_p),
            ("ty_m", ty_m),
            ("tz_p", tz_p),
            ("tz_m", tz_m),
            ("active", active),
            ("bc_mask", bc_mask),
            ("bc_values", bc_values),
            ("rhs", rhs),
        ):
            a = np.asarray(arr)
            if a.shape != shape:
                raise ValueError(f"{name} shape {a.shape} does not match {shape}")

        self._tx_p = np.asarray(tx_p, dtype=NP_FLOAT)
        self._tx_m = np.asarray(tx_m, dtype=NP_FLOAT)
        self._ty_p = np.asarray(ty_p, dtype=NP_FLOAT)
        self._ty_m = np.asarray(ty_m, dtype=NP_FLOAT)
        self._tz_p = np.asarray(tz_p, dtype=NP_FLOAT)
        self._tz_m = np.asarray(tz_m, dtype=NP_FLOAT)
        self._active = np.asarray(active, dtype=np.int32)
        self._bc_mask = np.asarray(bc_mask, dtype=np.int32)
        self._bc_values = np.asarray(bc_values, dtype=NP_FLOAT)
        self._rhs = np.asarray(rhs, dtype=NP_FLOAT)
        self._initial_head = (
            np.asarray(initial_head, dtype=NP_FLOAT)
            if initial_head is not None
            else None
        )
        self._kx_field = None
        self._ky_field = None
        self._kz_field = None
        return self

    def solve(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Solve the assembled 3D system.

        Additional keyword arguments are forwarded to the underlying solver
        (e.g. ``max_iter``, ``rel_tol``, ``transient``, ``storage_coeff``,
        ``dt``, ``head_prev``, ``unconfined``, ...).
        """
        if self._rhs is None:
            raise RuntimeError("Solver has not been built. Call build_from_K_fields or build_from_face_conductance first.")

        common = {
            "tx_p": self._tx_p,
            "tx_m": self._tx_m,
            "ty_p": self._ty_p,
            "ty_m": self._ty_m,
            "tz_p": self._tz_p,
            "tz_m": self._tz_m,
            "rhs": self._rhs,
            "active": self._active,
            "bc_mask": self._bc_mask,
            "bc_values": self._bc_values,
            "initial_head": self._initial_head,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "device": self.device,
            "diag_preconditioner_backend": self.diag_preconditioner_backend,
            "return_info": True,
        }
        solve_kwargs = dict(kwargs)
        if bool(solve_kwargs.get("unconfined", False)):
            if "kx_field" not in solve_kwargs and self._kx_field is not None:
                solve_kwargs["kx_field"] = self._kx_field
            if "ky_field" not in solve_kwargs and self._ky_field is not None:
                solve_kwargs["ky_field"] = self._ky_field
            if "kz_field" not in solve_kwargs and self._kz_field is not None:
                solve_kwargs["kz_field"] = self._kz_field
        common.update(solve_kwargs)

        if self.solver == "kcycle":
            return solve_multigrid_kcycle_7point_3d(**common)
        return solve_chebyshev_7point_3d(**common)

    def close(self) -> None:
        """Release stored host arrays."""
        self._tx_p = None
        self._tx_m = None
        self._ty_p = None
        self._ty_m = None
        self._tz_p = None
        self._tz_m = None
        self._rhs = None
        self._active = None
        self._bc_mask = None
        self._bc_values = None
        self._initial_head = None
        self._kx_field = None
        self._ky_field = None
        self._kz_field = None

    def __enter__(self) -> "WarpDarcySolver3D":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
