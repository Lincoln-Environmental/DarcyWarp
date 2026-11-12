# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Standalone 3D Darcy solver wrapper.

This module exposes ``WarpDarcySolver3D``, a thin class around the 7-point
finite-volume solvers in :mod:`DARCY_WARP_PACKAGE.solvers_3d`.  It is kept
separate from the 2D solver so the 2D stencil and API remain untouched.
"""

from __future__ import annotations

from typing import Any
from time import perf_counter

import numpy as np

from DARCY_WARP_PACKAGE.config import NP_FLOAT
from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask, floating_field, positive_integer
from DARCY_WARP_PACKAGE.solvers_3d import (
    solve_chebyshev_7point_3d,
    solve_multigrid_kcycle_7point_3d,
)
from DARCY_WARP_PACKAGE.solvers.session_3d import ThreeDOperatorSession
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k


class WarpDarcySolver3D:
    """
    Structured (nz, ny, nx) Darcy model for Warp CPU and CUDA.

    The solver uses a cell-centered 7-point finite-volume stencil with
    harmonic-mean face conductances.  It is a thin wrapper around the
    functions in :mod:`DARCY_WARP_PACKAGE.solvers_3d`. ``implementation``
    defaults to ``"classic"``; ``"fast"`` is an explicit FP64 face-array
    K-cycle for steady and backward-Euler confined or quasi-3D unconfined flow.
    Cartesian layers are flat, contiguous and uniformly dz thick.
    ``build_from_layers`` selects conventional variable-elevation DIS/NPF
    connections in the explicit pinned formulation, without terrain metrics.
    None of these modes implements unsaturated-flow physics.
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
        implementation: str = "classic",
        formulation: str = "legacy",
    ):
        nx = positive_integer(value=nx, name="nx")
        ny = positive_integer(value=ny, name="ny")
        nz = positive_integer(value=nz, name="nz")
        if not np.isfinite(float(dx)) or float(dx) <= 0.0:
            raise ValueError("dx must be finite and > 0")
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.dx = float(dx)
        self.dy = float(dy) if dy is not None else self.dx
        self.dz = float(dz)
        if not np.isfinite(self.dy) or self.dy <= 0.0 or not np.isfinite(self.dz) or self.dz <= 0.0:
            raise ValueError("dy and dz must be finite and > 0")
        self.device = str(device)
        self.solver = str(solver).lower()
        if self.solver not in {"kcycle", "chebyshev"}:
            raise ValueError("solver must be 'kcycle' or 'chebyshev'")
        backend_mode = str(diag_preconditioner_backend).strip().lower()
        if backend_mode not in {"auto", "host", "device"}:
            raise ValueError("diag_preconditioner_backend must be 'auto', 'host', or 'device'.")
        self.diag_preconditioner_backend = backend_mode
        implementation_mode = str(implementation).strip().lower()
        if implementation_mode not in {"classic", "fast"}:
            raise ValueError("implementation must be 'classic' or 'fast'.")
        self.implementation = implementation_mode
        if implementation_mode == 'fast' and backend_mode == 'host':
            raise ValueError("fast 3D applies its diagonal on the selected Warp device; use diag_preconditioner_backend='auto' or 'device'")
        if formulation == 'experimental_transformed_terrain':
            raise ValueError('Transformed terrain is experimental and available on dev')
        if formulation not in {'legacy', 'mf6_6_6_2_npf'}:
            raise ValueError('Unknown 3D formulation; choose legacy or mf6_6_6_2_npf')
        self.formulation=formulation
        if formulation=='mf6_6_6_2_npf' and np.dtype(NP_FLOAT)!=np.dtype(np.float64):
            raise ValueError('mf6_6_6_2_npf requires DARCY_FLOAT=float64 before imports')
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
        self._session: ThreeDOperatorSession | None = None
        self.geometry = "cartesian"

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
        *, ghb_conductance=None, ghb_head=None, ghb_mask=None,
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
            Volumetric cell injection [m³/day] in metre/day units, shape
            (nz, ny, nx). Multiply recharge [m/day] by dx*dy once.
        initial_head
            Optional initial guess, shape (nz, ny, nx).
        """
        active = binary_mask(value=active, name="active", shape=self.shape)
        fields = self._validated_K_fields(kx_field=kx_field, ky_field=ky_field, kz_field=kz_field)
        if self.formulation == 'mf6_6_6_2_npf':
            from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case
            case = NPF662Case(kx=fields[0], ky=fields[1], kz=fields[2], active=active,
                fixed=np.asarray(bc_mask), values=floating_field(value=bc_values, name="bc_values", dtype=NP_FLOAT),
                source=floating_field(value=rhs, name="rhs", dtype=NP_FLOAT),
                initial=np.zeros(self.shape) if initial_head is None else floating_field(value=initial_head, name="initial_head", dtype=NP_FLOAT),
                dx=self.dx, dy=self.dy, dz=self.dz)
            case.validate()
        tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = build_7point_face_conductance_from_k(
            kx_field=fields[0],
            ky_field=fields[1],
            kz_field=fields[2],
            active=active,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
        )
        self._build_from_face_conductance(
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
            ghb_conductance=ghb_conductance, ghb_head=ghb_head, ghb_mask=ghb_mask,
        )
        self._kx_field, self._ky_field, self._kz_field = fields
        return self

    def build_from_face_conductance(
        self, tx_p, tx_m, ty_p, ty_m, tz_p, tz_m, active, bc_mask,
        bc_values, rhs, initial_head=None,
        *, ghb_conductance=None, ghb_head=None, ghb_mask=None,
    ) -> "WarpDarcySolver3D":
        """Build independent reciprocal faces for legacy confined flow.

        Physical NPF conductivity/geometry must enter through its K or layered
        builder. Reject an incompatible replacement before changing the model.
        """
        if self.formulation == 'mf6_6_6_2_npf':
            raise ValueError('mf6_6_6_2_npf requires physical K/geometry; use build_from_K_fields or build_from_layers, not arbitrary faces')
        return self._build_from_face_conductance(tx_p=tx_p, tx_m=tx_m, ty_p=ty_p, ty_m=ty_m,
            tz_p=tz_p, tz_m=tz_m, active=active, bc_mask=bc_mask, bc_values=bc_values,
            rhs=rhs, initial_head=initial_head,
            ghb_conductance=ghb_conductance, ghb_head=ghb_head, ghb_mask=ghb_mask)

    def _build_from_face_conductance(
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
        *, ghb_conductance=None, ghb_head=None, ghb_mask=None,
    ) -> "WarpDarcySolver3D":
        """
        Build from reciprocal, nonnegative finite internal conductances.

        Each internal face must agree in its two directional representations
        within 32 input-dtype epsilons relative tolerance (no absolute floor).
        Connections touching inactive cells must be zero. Exterior slots are
        unused. Only flat, contiguous layers of scalar thickness dz are
        supported; unconfined solves validate supplied or inferred elevations.
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

        from DARCY_WARP_PACKAGE.physics.faces_3d import validate_reciprocal_faces
        validated_faces = validate_reciprocal_faces(
            faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m), active=active, canonicalize=True)
        input_faces = tuple(floating_field(value=value, name="face", dtype=NP_FLOAT)
                            for value in validated_faces)
        if any(np.any((raw > 0.0) & (stored == 0.0)) for raw, stored in zip(validated_faces, input_faces)):
            raise ValueError("positive face conductances must remain representable in the model dtype.")
        input_active = binary_mask(value=active, name="active", shape=shape)
        input_bc_mask = binary_mask(value=bc_mask, name="bc_mask", shape=shape)
        input_bc_values = floating_field(value=bc_values, name="bc_values", dtype=NP_FLOAT)
        input_rhs = floating_field(value=rhs, name="rhs", dtype=NP_FLOAT)
        input_initial = None if initial_head is None else floating_field(value=initial_head, name="initial_head", dtype=NP_FLOAT)
        session = ThreeDOperatorSession(
            faces=input_faces,
            active=input_active,
            bc_mask=input_bc_mask,
            bc_values=input_bc_values,
            rhs=input_rhs,
            initial_head=input_initial,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            ghb_conductance=ghb_conductance, ghb_head=ghb_head, ghb_mask=ghb_mask,
        )
        previous = self._session
        self._session = session
        self.geometry = "cartesian"
        self._kx_field = self._ky_field = self._kz_field = None
        # The session is the single owner of operator state.  Keep the
        # compatibility attributes as references, so classic, fast and the
        # unconfined fallback cannot observe divergent host arrays.
        self._tx_p, self._tx_m, self._ty_p, self._ty_m, self._tz_p, self._tz_m = self._session.faces
        self._active = self._session.active
        self._bc_mask = self._session.bc_mask
        self._bc_values = self._session.bc_values
        self._rhs = self._session.rhs
        self._initial_head = self._session.initial_head
        if previous is not None:
            previous.close()
        return self

    def build_from_layers(self, *, z_interfaces, kx_field, ky_field, kz_field,
                          active, bc_mask, bc_values, rhs, initial_head=None,
                          ghb_conductance=None, ghb_head=None, ghb_mask=None):
        """Build conventional layered DIS/NPF connections (not terrain metrics).

        Interfaces (nz+1,ny,nx) are authoritative. Constructor dz is unused;
        solve-time top/geometry overrides are rejected. Recalling this builder
        closes the previous session and all geometry-dependent workspaces.
        """
        if self.formulation!='mf6_6_6_2_npf':
            raise ValueError('build_from_layers requires formulation=mf6_6_6_2_npf')
        from DARCY_WARP_PACKAGE.physics.npf662_layers import build_layered_dis_faces
        from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase
        initial=np.zeros(self.shape) if initial_head is None else np.asarray(initial_head,dtype=NP_FLOAT)
        case=NPF662LayeredCase(kx=np.asarray(kx_field,dtype=NP_FLOAT),ky=np.asarray(ky_field,dtype=NP_FLOAT),
            kz=np.asarray(kz_field,dtype=NP_FLOAT),active=np.asarray(active),fixed=np.asarray(bc_mask),
            values=np.asarray(bc_values,dtype=NP_FLOAT),source=np.asarray(rhs,dtype=NP_FLOAT),initial=initial,
            dx=self.dx,dy=self.dy,z_interfaces=np.asarray(z_interfaces,dtype=NP_FLOAT))
        if case.active.shape!=self.shape:raise ValueError(f'Layered cell shape must be {self.shape}')
        case.validate()
        faces,unused=build_layered_dis_faces(z_interfaces=case.z_interfaces,kx=case.kx,ky=case.ky,kz=case.kz,
            active=case.active,dx=self.dx,dy=self.dy)
        session=ThreeDOperatorSession(faces=faces,active=case.active,bc_mask=case.fixed,bc_values=case.values,
            rhs=case.source,initial_head=case.initial,dx=self.dx,dy=self.dy,dz=self.dz,z_interfaces=case.z_interfaces,
            ghb_conductance=ghb_conductance,ghb_head=ghb_head,ghb_mask=ghb_mask)
        # The layered contract already enforces positive active K; finite
        # inactive conductivity markers remain compatible with that schema.
        fields = self._validated_K_fields(kx_field=case.kx, ky_field=case.ky, kz_field=case.kz,
                                           require_nonnegative=False)
        previous = self._session
        self._session=session
        self.geometry='structured_dis_layered'
        self._kx_field,self._ky_field,self._kz_field=fields
        self._tx_p,self._tx_m,self._ty_p,self._ty_m,self._tz_p,self._tz_m=session.faces
        self._active,self._bc_mask,self._bc_values=session.active,session.bc_mask,session.bc_values
        self._rhs,self._initial_head=session.rhs,session.initial_head
        if previous is not None:
            previous.close()
        return self

    def update_ghb(self, *, ghb_conductance, ghb_head, ghb_mask=None):
        """Update integrated cell conductance [L²/T] and external head [L].

        Conductance is independent of aquifer K and saturation. Zero disables
        a boundary. Terrain subface contacts use update_contacts instead.
        """
        if self._session is None:
            raise RuntimeError("Build the model before updating GHB")
        if self.geometry == "terrain":
            raise ValueError("Terrain uses contact exchange; update_contacts is required")
        self._session.update_ghb(ghb_conductance=ghb_conductance,
                                 ghb_head=ghb_head, ghb_mask=ghb_mask)

    def _validated_K_fields(self, *, kx_field, ky_field, kz_field, require_nonnegative=True):
        fields = tuple(floating_field(value=value, name=name, dtype=NP_FLOAT) for name, value in
                       (("kx", kx_field), ("ky", ky_field), ("kz", kz_field)))
        if any(value.shape != self.shape for value in fields):
            raise ValueError("conductivity fields must match the model shape")
        if require_nonnegative and any(np.any(value < 0.) for value in fields):
            raise ValueError("conductivity fields must be nonnegative")
        for value in fields:
            value.setflags(write=False)
        return fields

    def update_K_fields(self, *, kx_field, ky_field, kz_field):
        """Update physical conductivity, preserving geometry and model ownership.

        All six derived faces change together; cached operators are refreshed
        on the next solve. Independent face models must use their face API.
        """
        if self._session is None or self._kx_field is None:
            raise ValueError("update_K_fields requires build_from_K_fields or build_from_layers")
        fields = self._validated_K_fields(kx_field=kx_field, ky_field=ky_field, kz_field=kz_field,
                                           require_nonnegative=self.geometry != 'structured_dis_layered')
        if self.formulation == 'mf6_6_6_2_npf' and any(np.any(value[self._active != 0] <= 0.) for value in fields):
            raise ValueError("NPF requires positive physical conductivity in all active cells")
        if self.geometry == "structured_dis_layered":
            from DARCY_WARP_PACKAGE.physics.npf662_layers import build_layered_dis_faces
            faces, unused = build_layered_dis_faces(z_interfaces=self._session.z_interfaces,
                kx=fields[0], ky=fields[1], kz=fields[2], active=self._active, dx=self.dx, dy=self.dy)
        else:
            faces = build_7point_face_conductance_from_k(kx_field=fields[0], ky_field=fields[1],
                kz_field=fields[2], active=self._active, dx=self.dx, dy=self.dy, dz=self.dz)
        self._session.update_face_conductance(faces=faces)
        self._kx_field, self._ky_field, self._kz_field = fields
        self._tx_p, self._tx_m, self._ty_p, self._ty_m, self._tz_p, self._tz_m = self._session.faces

    def solve_transient_npf(self, *, source_factors, dt, head_prev, ss, sy, **solver_options):
        """Solve an NPF/STO history with intermediate heads retained on the GPU.

        K, geometry, cell types, storage fields and prescribed heads stay fixed
        within a call. Source factors and dt vary per step. Update physical K
        between calls with update_K_fields. A failed step stops the history.
        """
        from DARCY_WARP_PACKAGE.solvers.npf662_history import solve_transient_history
        return solve_transient_history(model=self, source_factors=source_factors, dt=dt,
            head_prev=head_prev, ss=ss, sy=sy, **solver_options)

    def solve(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve and time the public call, including independent acceptance.

        Backend/event/synchronisation subintervals can overlap; this wall time
        is measured directly and is never assembled by adding those fields.
        """
        started = perf_counter()
        result = self._solve(**kwargs)
        if isinstance(result, tuple):
            result[1]['public_total_seconds'] = perf_counter()-started
            result[1]['public_timing_scope'] = 'dispatch, setup, iterations, readback and independent final acceptance'
        return result

    def _solve(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Solve the assembled 3D system.

        Additional keyword arguments are forwarded to the underlying solver
        (e.g. ``max_iter``, ``rel_tol``, ``transient``, ``storage_coeff``,
        ``dt``, ``head_prev``, ``unconfined``, ...).
        """
        if self._rhs is None:
            raise RuntimeError("Solver has not been built. Call build_from_K_fields or build_from_face_conductance first.")

        if self.formulation=='mf6_6_6_2_npf':
            from DARCY_WARP_PACKAGE.solvers.npf662_3d import solve_npf662
            return solve_npf662(model=self,**kwargs)

        if self._session is not None:
            solve_kwargs = dict(kwargs)
            if bool(solve_kwargs.get("unconfined", False)):
                if self._kx_field is None:
                    raise ValueError("independent face conductances support confined flow; use build_from_K_fields for physical unconfined rebuilding")
                # Unconfined solves rebuild the operator from the stored K
                # fields each Picard iteration; supply them by default.
                for name, field in (
                    ("kx_field", self._kx_field),
                    ("ky_field", self._ky_field),
                    ("kz_field", self._kz_field),
                ):
                    if name in solve_kwargs and field is not None and not np.array_equal(solve_kwargs[name], field):
                        raise ValueError("physical conductivity owns this model; use update_K_fields before solving")
                    if name not in solve_kwargs and field is not None:
                        solve_kwargs[name] = field
            solve_kwargs.setdefault("return_info", True)
            result=self._session.solve(
                solver=self.solver,
                device=self.device,
                implementation=self.implementation,
                diag_preconditioner_backend=self.diag_preconditioner_backend,
                **solve_kwargs,
            )
            if isinstance(result,tuple):result[1].setdefault('formulation','legacy_cartesian')
            return result

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

    def update_rhs(self, rhs: np.ndarray) -> None:
        """Update the source field in the persistent 3D operator session."""
        if self._session is None:
            raise RuntimeError("Solver has not been built.")
        if self.formulation == 'mf6_6_6_2_npf':
            value = np.asarray(rhs)
            if value.shape == self.shape and np.any(value[self._active == 0] != 0.):
                raise ValueError('Inactive NPF source would be lost; rhs must be zero in inactive cells')
        self._session.update_rhs(rhs)
        self._rhs = self._session.rhs

    def update_initial_head(self, initial_head: np.ndarray | None) -> None:
        """Update the persistent initial guess without rebuilding faces."""
        if self._session is None:
            raise RuntimeError("Solver has not been built.")
        self._session.update_initial_head(initial_head)
        self._initial_head = self._session.initial_head

    def update_boundary_values(self, bc_values: np.ndarray) -> None:
        """Update prescribed heads while retaining faces and boundary masks."""

        if self._session is None:
            raise RuntimeError("Solver has not been built.")
        self._session.update_boundary_values(bc_values)
        self._bc_values = self._session.bc_values

    def update_face_conductance(
        self,
        tx_p: np.ndarray,
        tx_m: np.ndarray,
        ty_p: np.ndarray,
        ty_m: np.ndarray,
        tz_p: np.ndarray,
        tz_m: np.ndarray,
    ) -> None:
        """Refresh six face arrays in place and retain the session identity."""
        if self.formulation=='mf6_6_6_2_npf':
            raise ValueError('Pinned NPF requires physical K/geometry; rebuild with build_from_K_fields or build_from_layers instead of overriding conductances')
        if self._session is None:
            raise RuntimeError("Solver has not been built.")
        if self._kx_field is not None:
            raise ValueError("physical conductivity owns these faces; use update_K_fields instead")
        self._session.update_face_conductance(faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m))
        self._tx_p, self._tx_m, self._ty_p, self._ty_m, self._tz_p, self._tz_m = self._session.faces

    def close(self) -> None:
        """Release stored host arrays."""
        if self._session is not None:
            self._session.close()
            self._session = None
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
