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
from DARCY_WARP_PACKAGE.physics.recharge_3d import combine_cell_sources, recharge_depth_rate
from DARCY_WARP_PACKAGE.physics.wel import (
    combine_volumetric_sources,
    signed_rate_budget,
    validate_well_entries,
    well_cell_volume_rates,
)
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
        if formulation == "mf6_6_6_2_npf" and self.solver != "kcycle":
            raise ValueError("mf6_6_6_2_npf requires solver='kcycle'.")
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
        self._volumetric_source: np.ndarray | None = None
        self._recharge_rate: np.ndarray | None = None
        self._wel_entries = None
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
        rhs: np.ndarray | None = None,
        initial_head: np.ndarray | None = None,
        *, R_field: float | np.ndarray | None = None,
        wel_entries=None,
        ghb_conductance=None, ghb_head=None, ghb_mask=None,
        drn_conductance=None, drn_elevation=None, drn_mask=None,
        riv_conductance=None, riv_stage=None, riv_bottom=None, riv_mask=None,
        drn_entries=None, riv_entries=None,
    ) -> "WarpDarcySolver3D":
        """
        Build face conductances from cell-centered hydraulic conductivity fields.

        :param kx_field: Cell-centred east-west conductivity [L/T].
        :param ky_field: Cell-centred north-south conductivity [L/T].
        :param kz_field: Cell-centred vertical conductivity [L/T].
        :param active: Active-cell mask of shape (nz, ny, nx), top layer first.
        :param bc_mask: Fixed-head mask of the same shape.
        :param bc_values: Prescribed heads [L] at fixed-head cells.
        :param rhs: Optional 3D volumetric rates [L³/T], e.g. signed well rates.
        :param initial_head: Optional initial heads [L].
        :param R_field: Signed areal recharge [L/T], scalar or (ny, nx). Positive
            adds water. Recharge acts on the highest active cell of each column
            unless it has a prescribed head. Cell area is applied internally.
        :param wel_entries: Optional explicit sparse wells; ``(cell, rate)`` pairs
            with volumetric rates [L³/T] at ``(layer, row, col)`` cells, a
            cell-to-rate mapping, or an existing :class:`WellEntries` value.
            Negative rates abstract water. Wells on inactive cells are rejected.
        :param ghb_conductance: Optional GHB conductance [L²/T] per cell.
        :param ghb_head: Optional GHB external head [L] per cell.
        :param ghb_mask: Optional mask selecting GHB cells.
        :param drn_conductance: Optional DRN conductance [L²/T], scalar or full grid.
        :param drn_elevation: Drain activation elevation [L], scalar or full grid.
        :param drn_mask: Optional binary mask selecting DRN cells.
        :param riv_conductance: Optional RIV conductance [L²/T], scalar or full grid.
        :param riv_stage: River stage [L], scalar or full grid.
        :param riv_bottom: Riverbed bottom elevation [L], scalar or full grid.
        :param riv_mask: Optional binary mask selecting RIV cells.
        :param drn_entries: Optional sparse drains replacing the dense DRN pair;
            ``(cell, conductance, elevation)`` records in ``(layer, row, col)``
            cells, a cell-to-values mapping, or an existing
            :class:`BoundaryEntryList`. Multiple drains may share one cell.
        :param riv_entries: Optional sparse river entries replacing the dense
            RIV triple; ``(cell, conductance, stage, bottom)`` records with the
            same accepted forms.
        :return: The built solver.
        """
        active = binary_mask(value=active, name="active", shape=self.shape)
        fields = self._validated_K_fields(kx_field=kx_field, ky_field=ky_field, kz_field=kz_field)
        volumetric, rate, entries, source = self._prepare_sources(
            rhs=rhs, R_field=R_field, wel_entries=wel_entries, active=active, bc_mask=bc_mask,
        )
        if self.formulation == 'mf6_6_6_2_npf':
            from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case
            case = NPF662Case(kx=fields[0], ky=fields[1], kz=fields[2], active=active,
                fixed=np.asarray(bc_mask), values=floating_field(value=bc_values, name="bc_values", dtype=NP_FLOAT),
                source=source,
                initial=np.zeros(self.shape) if initial_head is None else floating_field(value=initial_head, name="initial_head", dtype=NP_FLOAT),
                dx=self.dx, dy=self.dy, dz=self.dz,
                drn_conductance=drn_conductance, drn_elevation=drn_elevation,
                drn_mask=drn_mask, riv_conductance=riv_conductance, riv_stage=riv_stage,
                riv_bottom=riv_bottom, riv_mask=riv_mask,
                drn_entries=drn_entries, riv_entries=riv_entries)
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
            rhs=source,
            initial_head=initial_head,
            ghb_conductance=ghb_conductance, ghb_head=ghb_head, ghb_mask=ghb_mask,
            drn_conductance=drn_conductance, drn_elevation=drn_elevation, drn_mask=drn_mask,
            riv_conductance=riv_conductance, riv_stage=riv_stage, riv_bottom=riv_bottom, riv_mask=riv_mask,
            drn_entries=drn_entries, riv_entries=riv_entries,
        )
        self._kx_field, self._ky_field, self._kz_field = fields
        self._store_source_components(volumetric, rate, entries)
        return self

    def build_from_face_conductance(
        self, tx_p, tx_m, ty_p, ty_m, tz_p, tz_m, active, bc_mask,
        bc_values, rhs=None, initial_head=None,
        *, R_field=None, wel_entries=None, ghb_conductance=None, ghb_head=None, ghb_mask=None,
        drn_conductance=None, drn_elevation=None, drn_mask=None,
        riv_conductance=None, riv_stage=None, riv_bottom=None, riv_mask=None,
        drn_entries=None, riv_entries=None,
    ) -> "WarpDarcySolver3D":
        """Build independent reciprocal faces for legacy confined flow.

        Physical NPF conductivity/geometry must enter through its K or layered
        builder. Reject an incompatible replacement before changing the model.

        :param rhs: Optional 3D volumetric rates [L³/T], e.g. signed well rates.
        :param R_field: Signed surface recharge [L/T], scalar or (ny, nx), with
            the same receiving cells and sign as in :meth:`build_from_K_fields`.
        :param wel_entries: Optional sparse wells as in :meth:`build_from_K_fields`.
        """
        if self.formulation == 'mf6_6_6_2_npf':
            raise ValueError('mf6_6_6_2_npf requires physical K/geometry; use build_from_K_fields or build_from_layers, not arbitrary faces')
        volumetric, rate, entries, source = self._prepare_sources(
            rhs=rhs, R_field=R_field, wel_entries=wel_entries, active=active, bc_mask=bc_mask,
        )
        self._build_from_face_conductance(tx_p=tx_p, tx_m=tx_m, ty_p=ty_p, ty_m=ty_m,
            tz_p=tz_p, tz_m=tz_m, active=active, bc_mask=bc_mask, bc_values=bc_values,
            rhs=source, initial_head=initial_head,
            ghb_conductance=ghb_conductance, ghb_head=ghb_head, ghb_mask=ghb_mask,
            drn_conductance=drn_conductance, drn_elevation=drn_elevation, drn_mask=drn_mask,
            riv_conductance=riv_conductance, riv_stage=riv_stage, riv_bottom=riv_bottom, riv_mask=riv_mask,
            drn_entries=drn_entries, riv_entries=riv_entries)
        self._store_source_components(volumetric, rate, entries)
        return self

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
        drn_conductance=None, drn_elevation=None, drn_mask=None,
        riv_conductance=None, riv_stage=None, riv_bottom=None, riv_mask=None,
        drn_entries=None, riv_entries=None,
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

        input_active = binary_mask(value=active, name="active", shape=shape)
        input_bc_mask = binary_mask(value=bc_mask, name="bc_mask", shape=shape)
        input_bc_values = floating_field(value=bc_values, name="bc_values", dtype=NP_FLOAT)
        input_initial = None if initial_head is None else floating_field(value=initial_head, name="initial_head", dtype=NP_FLOAT)
        session = ThreeDOperatorSession(
            faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m),
            active=input_active,
            bc_mask=input_bc_mask,
            bc_values=input_bc_values,
            rhs=rhs,
            initial_head=input_initial,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            ghb_conductance=ghb_conductance, ghb_head=ghb_head, ghb_mask=ghb_mask,
            drn_conductance=drn_conductance, drn_elevation=drn_elevation, drn_mask=drn_mask,
            riv_conductance=riv_conductance, riv_stage=riv_stage, riv_bottom=riv_bottom, riv_mask=riv_mask,
            drn_entries=drn_entries, riv_entries=riv_entries,
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
                          active, bc_mask, bc_values, rhs=None, initial_head=None,
                          R_field=None, wel_entries=None, ghb_conductance=None, ghb_head=None, ghb_mask=None,
                          drn_conductance=None, drn_elevation=None, drn_mask=None,
                          riv_conductance=None, riv_stage=None, riv_bottom=None, riv_mask=None,
                          drn_entries=None, riv_entries=None):
        """Build conventional layered DIS/NPF connections (not terrain metrics).

        Interfaces (nz+1,ny,nx) are authoritative. Constructor dz is unused;
        solve-time top/geometry overrides are rejected. Recalling this builder
        closes the previous session and all geometry-dependent workspaces.

        :param rhs: Optional 3D volumetric rates [L³/T], e.g. signed well rates.
        :param R_field: Signed surface recharge [L/T], scalar or (ny, nx). Uses
            the horizontal cell area and the same receiving cells as the
            Cartesian :meth:`build_from_K_fields` interface.
        :param wel_entries: Optional sparse wells as in :meth:`build_from_K_fields`.
        """
        if self.formulation!='mf6_6_6_2_npf':
            raise ValueError('build_from_layers requires formulation=mf6_6_6_2_npf')
        from DARCY_WARP_PACKAGE.physics.npf662_layers import build_layered_dis_faces
        from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase
        volumetric, rate, entries, source = self._prepare_sources(
            rhs=rhs, R_field=R_field, wel_entries=wel_entries, active=active, bc_mask=bc_mask,
        )
        initial=np.zeros(self.shape) if initial_head is None else np.asarray(initial_head,dtype=NP_FLOAT)
        case=NPF662LayeredCase(kx=np.asarray(kx_field,dtype=NP_FLOAT),ky=np.asarray(ky_field,dtype=NP_FLOAT),
            kz=np.asarray(kz_field,dtype=NP_FLOAT),active=np.asarray(active),fixed=np.asarray(bc_mask),
            values=np.asarray(bc_values,dtype=NP_FLOAT),source=source,initial=initial,
            dx=self.dx,dy=self.dy,z_interfaces=np.asarray(z_interfaces,dtype=NP_FLOAT),
            drn_conductance=drn_conductance,drn_elevation=drn_elevation,
            drn_mask=drn_mask,riv_conductance=riv_conductance,riv_stage=riv_stage,
            riv_bottom=riv_bottom,riv_mask=riv_mask,
            drn_entries=drn_entries,riv_entries=riv_entries)
        if case.active.shape!=self.shape:raise ValueError(f'Layered cell shape must be {self.shape}')
        case.validate()
        faces,unused=build_layered_dis_faces(z_interfaces=case.z_interfaces,kx=case.kx,ky=case.ky,kz=case.kz,
            active=case.active,dx=self.dx,dy=self.dy)
        session=ThreeDOperatorSession(faces=faces,active=case.active,bc_mask=case.fixed,bc_values=case.values,
            rhs=case.source,initial_head=case.initial,dx=self.dx,dy=self.dy,dz=self.dz,z_interfaces=case.z_interfaces,
            ghb_conductance=ghb_conductance,ghb_head=ghb_head,ghb_mask=ghb_mask,
            drn_conductance=drn_conductance,drn_elevation=drn_elevation,drn_mask=drn_mask,
            riv_conductance=riv_conductance,riv_stage=riv_stage,riv_bottom=riv_bottom,riv_mask=riv_mask,
            drn_entries=drn_entries,riv_entries=riv_entries)
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
        self._store_source_components(volumetric, rate, entries)
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

    def update_drn(self, *, drn_conductance, drn_elevation, drn_mask=None):
        """Update the model-owned native 3D DRN package.

        :param drn_conductance: Integrated conductance [L2/T], scalar or full grid; None disables DRN.
        :param drn_elevation: Drain activation elevation [L], scalar or full grid.
        :param drn_mask: Optional binary mask using the model's (nz, ny, nx) layout.
        :return: None.
        :raises RuntimeError: If the model has not been built.
        :raises ValueError: If a field is invalid for the current grid or boundary conditions.
        """
        if self._session is None:
            raise RuntimeError("Build the model before updating DRN")
        if self.geometry == "terrain":
            raise ValueError("Terrain sessions do not support cell DRN; use a supported Cartesian NPF model")
        self._session.update_drn(drn_conductance=drn_conductance,
            drn_elevation=drn_elevation, drn_mask=drn_mask)

    def update_riv(self, *, riv_conductance, riv_stage, riv_bottom, riv_mask=None):
        """Update the model-owned native 3D RIV package.

        :param riv_conductance: Integrated conductance [L2/T], scalar or full grid; None disables RIV.
        :param riv_stage: River surface elevation [L], scalar or full grid.
        :param riv_bottom: Riverbed bottom elevation [L], scalar or full grid.
        :param riv_mask: Optional binary mask using the model's (nz, ny, nx) layout.
        :return: None.
        :raises RuntimeError: If the model has not been built.
        :raises ValueError: If a field is invalid for the current grid or boundary conditions.
        """
        if self._session is None:
            raise RuntimeError("Build the model before updating RIV")
        if self.geometry == "terrain":
            raise ValueError("Terrain sessions do not support cell RIV; use a supported Cartesian NPF model")
        self._session.update_riv(riv_conductance=riv_conductance, riv_stage=riv_stage,
            riv_bottom=riv_bottom, riv_mask=riv_mask)

    def update_drn_entries(self, entries) -> None:
        """Replace the native 3D DRN package with sparse entries.

        Multiple drains may occupy one cell; each keeps its own conductance,
        elevation and accepted exchange rate. Supplying entries replaces any
        dense DRN field pair.

        :param entries: Sparse ``(cell, conductance, elevation)`` records at
            ``(layer, row, col)`` cells, a cell-to-values mapping, or None to
            remove every drain.
        :raises RuntimeError: If the model has not been built.
        :raises ValueError: If an entry is invalid; the model is unchanged then.
        """
        if self._session is None:
            raise RuntimeError("Build the model before updating DRN entries")
        if self.geometry == "terrain":
            raise ValueError("Terrain sessions do not support cell DRN; use a supported Cartesian NPF model")
        self._session.update_drn_entries(entries)

    def update_riv_entries(self, entries) -> None:
        """Replace the native 3D RIV package with sparse entries.

        Multiple reaches may intersect one cell without averaging stages or
        bed bottoms; each entry's physical flux is evaluated independently and
        the cell contributions sum. Supplying entries replaces any dense RIV
        field triple.

        :param entries: Sparse ``(cell, conductance, stage, bottom)`` records,
            a cell-to-values mapping, or None to remove every river entry.
        :raises RuntimeError: If the model has not been built.
        :raises ValueError: If an entry is invalid; the model is unchanged then.
        """
        if self._session is None:
            raise RuntimeError("Build the model before updating RIV entries")
        if self.geometry == "terrain":
            raise ValueError("Terrain sessions do not support cell RIV; use a supported Cartesian NPF model")
        self._session.update_riv_entries(entries)

    def _verify_accepted_state(self, *, head, info, package):
        """Verify accepted-state ownership before releasing package rates.

        An opaque token locates the session-owned historical equation. Supplied
        heads, cell rates and entry rates are compared against owned outputs;
        later changes to current model fields cannot rewrite that equation.

        :param head: Groundwater head returned by the same solve as ``info``.
        :param info: Converged solve metadata carrying ``accepted_state``.
        :param package: ``"drn"`` or ``"riv"`` for error messages.
        :return: The accepted per-cell rate array as an owned copy.
        :raises RuntimeError: If the solve state is not accepted or unowned.
        :raises ValueError: If the head or rate array is malformed or unrelated.
        """
        from DARCY_WARP_PACKAGE.solvers.accepted_npf_state import verified_package_rates
        if self._session is None:
            raise RuntimeError("The model has not been built")
        return verified_package_rates(session=self._session, head=head, info=info, package=package)

    def accepted_drn_discharge(self, *, head, info) -> np.ndarray:
        """Return nonnegative per-cell DRN outflow from a verified accepted solve.

        :param head: Groundwater head returned by the same solve as ``info``.
        :param info: Converged solve metadata containing accepted per-cell DRN rates.
        :return: Owned (nz, ny, nx) discharge array, positive out of the aquifer.
        :raises RuntimeError: If DRN is disabled or the solve state is not accepted.
        :raises ValueError: If the head or rate array is malformed or unrelated.
        """
        rate = self._verify_accepted_state(head=head, info=info, package="drn")
        if np.any(rate < 0.0):
            raise ValueError("drn_discharge_rate_array must be nonnegative")
        return rate

    def accepted_riv_discharge(self, *, head, info) -> np.ndarray:
        """Return signed per-cell RIV exchange from a verified accepted solve.

        :param head: Groundwater head returned by the same solve as ``info``.
        :param info: Converged solve metadata containing accepted per-cell RIV rates.
        :return: Owned (nz, ny, nx) rates, positive out of the aquifer.
        :raises RuntimeError: If RIV is disabled or the solve state is not accepted.
        :raises ValueError: If the head or rate array is malformed or unrelated.
        """
        return self._verify_accepted_state(head=head, info=info, package="riv")

    def _validated_K_fields(self, *, kx_field, ky_field, kz_field, require_nonnegative=True):
        fields = tuple(floating_field(value=value, name=name, dtype=np.float64) for name, value in
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

    def solve_transient_npf(self, *, dt, head_prev, ss, sy, source_factors=None,
                            recharge_rates=None, wel_rates=None, **solver_options):
        """Solve an NPF/STO history with intermediate heads retained on the GPU.

        K, geometry, cell types, storage and prescribed heads stay fixed within
        a call. A failed step stops the history.

        :param dt: Time step durations in model time units.
        :param head_prev: Heads [L] at the start of the history.
        :param ss: Specific storage [1/L].
        :param sy: Specific yield.
        :param source_factors: Optional multipliers on the built cell sources per
            step. Explicit recharge_rates or wel_rates components are excluded
            from the scaling and replaced independently.
        :param recharge_rates: Optional areal recharge [L/T] replacing R_field
            for this call. Supply a scalar, (ny, nx) map, (nstep,) vector, or
            (nstep, ny, nx) stack.
        :param wel_rates: Optional sparse well schedule replacing the built
            entries. Supply None or one well-entry value per step in
            :meth:`build_from_K_fields` entry form.
        :param solver_options: Additional options for the transient NPF solver.
        :return: History results and diagnostics from the transient solver.
        """
        from DARCY_WARP_PACKAGE.solvers.npf662_history import solve_transient_history
        return solve_transient_history(
            model=self, source_factors=source_factors, recharge_rates=recharge_rates,
            wel_rates=wel_rates, dt=dt, head_prev=head_prev, ss=ss, sy=sy, **solver_options,
        )

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

        if {"ghb_head", "ghb_conductance", "ghb_mask"} & kwargs.keys():
            raise ValueError("GHB is model-owned. Call update_ghb() before solving.")

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

    def _prepare_sources(self, *, rhs, R_field, wel_entries, active, bc_mask):
        """Own the inputs separately so updating one preserves the others."""
        active = np.asarray(active)
        fixed = np.asarray(bc_mask)
        if active.shape != self.shape or fixed.shape != self.shape:
            raise ValueError(f"active and bc_mask must have shape {self.shape}")
        volumetric = None if rhs is None else floating_field(value=rhs, name="rhs", dtype=NP_FLOAT)
        rate = recharge_depth_rate(R_field=R_field, horizontal_shape=self.shape[1:])
        entries = validate_well_entries(entries=wel_entries, shape=self.shape, active=active)
        if rate is None and entries is None and volumetric is not None:
            source = volumetric
        else:
            source = self._assemble_source(rhs=volumetric, R_field=rate, entries=entries,
                                           active=active, fixed=fixed, dtype=NP_FLOAT)
        return volumetric, rate, entries, source

    def _assemble_source(self, *, rhs, R_field, entries, active, fixed, dtype):
        """Combine raw volumetric rates, explicit wells and areal recharge.

        Well rates are volumetric cell rates [L³/T] added directly at their
        cells; recharge is converted once with the horizontal cell area.
        """
        wells = well_cell_volume_rates(entries=entries, shape=self.shape, dtype=dtype)
        volumetric = combine_volumetric_sources(rhs=rhs, wells=wells, dtype=dtype)
        return combine_cell_sources(
            rhs=volumetric, R_field=R_field, active=active, bc_mask=fixed,
            dx=self.dx, dy=self.dy, dtype=dtype,
        )

    def _store_source_components(self, volumetric, rate, entries):
        """Share the session array when it contains only volumetric sources."""
        if volumetric is not None and rate is None and entries is None:
            volumetric = self._rhs
        self._volumetric_source = volumetric
        self._recharge_rate = rate
        self._wel_entries = entries

    def _source_with_recharge(self, R_field, dtype=None):
        """Assemble a replacement recharge rate with the current other sources."""
        if dtype is None:
            dtype = NP_FLOAT if R_field is not None else self._session.rhs.dtype
        return self._assemble_source(rhs=self._volumetric_source, R_field=R_field,
            entries=self._wel_entries, active=self._active, fixed=self._bc_mask, dtype=dtype)

    def _source_without_wells(self, dtype=None):
        """Assemble the built source with every explicit well entry removed.

        The remaining volumetric rates and built areal recharge stay in place.
        Transient histories use this as the scaling base when a well schedule
        replaces the built entries per step.
        """
        if dtype is None:
            dtype = self._session.rhs.dtype
        return self._assemble_source(rhs=self._volumetric_source, R_field=self._recharge_rate,
            entries=None, active=self._active, fixed=self._bc_mask, dtype=dtype)

    def wel_budget(self) -> dict:
        """Gross physical well budget of the owned sparse entries.

        :return: Mapping with ``inflow``, ``outflow`` and ``net`` rates [L³/T].
            Opposing entries sharing a cell keep separate gross contributions.
        :raises RuntimeError: If the model has not been built.
        """
        if self._session is None:
            raise RuntimeError("Build the model before reading the well budget.")
        rates = None if self._wel_entries is None else self._wel_entries.rates
        return signed_rate_budget(rates=rates)

    def physical_source_budgets(self) -> dict:
        """Separate gross budgets of every owned physical source component.

        :return: Mapping with ``wel``, ``recharge`` and ``volumetric`` entries.
            Each carries gross ``inflow``, ``outflow`` and ``net`` rates; the
            recharge rates include the horizontal cell area conversion once.
            Components retain their own gross values even when they cancel in
            the assembled numerical source.
        :raises RuntimeError: If the model has not been built.
        """
        if self._session is None:
            raise RuntimeError("Build the model before reading source budgets.")
        wel = self.wel_budget()
        rate = self._recharge_rate
        area = self.dx * self.dy
        if rate is None:
            recharge = dict(inflow=0., outflow=0., net=0.)
        else:
            from DARCY_WARP_PACKAGE.physics.recharge_3d import surface_recharge_cells
            flat_rates = np.broadcast_to(np.asarray(rate, dtype=np.float64),
                self.shape[1:]).ravel() * area
            # Only the actual receiving cells carry recharge; prescribed-head
            # columns receive none and must not enter the physical budget.
            receivers = surface_recharge_cells(self._active, self._bc_mask)
            recharge = signed_rate_budget(rates=flat_rates[receivers % flat_rates.size])
        source = self._volumetric_source
        if source is None:
            volumetric = dict(inflow=0., outflow=0., net=0.)
        else:
            volumetric = signed_rate_budget(rates=np.asarray(source, dtype=np.float64).ravel())
        return dict(wel=wel, recharge=recharge, volumetric=volumetric)

    def update_wel(self, entries) -> None:
        """Replace explicit well entries while retaining recharge and other sources.

        :param entries: Sparse well entries in :meth:`build_from_K_fields` form,
            or None to remove every well.
        :raises RuntimeError: If the model has not been built.
        :raises ValueError: If an entry is invalid; the previous model state is
            left unchanged in that case.
        """
        if self._session is None:
            raise RuntimeError("Build the model before updating wells.")
        validated = validate_well_entries(entries=entries, shape=self.shape, active=self._active)
        source = self._assemble_source(rhs=self._volumetric_source, R_field=self._recharge_rate,
            entries=validated, active=self._active, fixed=self._bc_mask,
            dtype=NP_FLOAT if self._recharge_rate is not None else self._session.rhs.dtype)
        volumetric = self._volumetric_source
        if volumetric is self._rhs and validated is not None:
            # The session upload overwrites its owned RHS in place. Separate
            # the raw physical source only when a second component is added.
            volumetric = volumetric.copy()
        self._session.update_rhs(source)
        self._store_source_components(volumetric, self._recharge_rate, validated)

    def update_R_in_place(self, R_field) -> None:
        """Replace surface recharge while retaining wells and other cell rates.

        :param R_field: Signed recharge [L/T], scalar or (ny, nx), or None to clear.
        """
        if self._session is None:
            raise RuntimeError("Build the model before updating recharge.")
        rate = recharge_depth_rate(R_field=R_field, horizontal_shape=self.shape[1:])
        volumetric = self._volumetric_source
        if volumetric is self._rhs and (rate is not None or self._wel_entries is not None):
            # The first recharge update makes the combined source differ from
            # the raw rates. Preserve the latter before overwriting the session.
            volumetric = volumetric.copy()
        self._session.update_rhs(self._source_with_recharge(rate))
        self._store_source_components(volumetric, rate, self._wel_entries)

    def update_rhs(self, rhs: np.ndarray) -> None:
        """Replace volumetric cell sources while retaining recharge and wells.

        :param rhs: Signed 3D cell source rates [L³/T], e.g. injection or pumping.
        """
        if self._session is None:
            raise RuntimeError("Solver has not been built.")
        if self._recharge_rate is None and self._wel_entries is None:
            self._session.update_rhs(rhs)
            self._volumetric_source = self._rhs
            return
        volumetric = floating_field(value=rhs, name="rhs", dtype=self._session.rhs.dtype)
        source = self._assemble_source(rhs=volumetric, R_field=self._recharge_rate,
            entries=self._wel_entries, active=self._active, fixed=self._bc_mask, dtype=NP_FLOAT)
        self._session.update_rhs(source)
        self._volumetric_source = volumetric

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
        self._volumetric_source = None
        self._recharge_rate = None
        self._wel_entries = None
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
