# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only borrowing context for the 2D nonlinear operator.

``NonlinearOperatorContext2D`` exposes borrowed, non-owning access to every
piece of physical data needed to evaluate the unconfined DarcyWarp equation
directly from hydraulic head.  It is the nonlinear analogue of
``solvers.context.SolverContext``: it composes small frozen dataclasses that
*reference* the caller's NumPy arrays without copying or claiming ownership.

Nothing in this module allocates device memory, mutates the caller's arrays, or
depends on a solver backend.  ``NonlinearOperator2D`` (see ``operator.py``) is
the component that owns persistent device scratch and consumes this context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kernels import NP_FLOAT


@dataclass(frozen=True, slots=True)
class NonlinearGrid:
    """Grid metadata and floating-point / device configuration."""

    nx: int
    ny: int
    dx: float
    device: str
    wp_dtype: Any
    np_dtype: Any

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def area(self) -> float:
        """Plan area of one cell (square grid), ``dx * dx``."""
        d = float(self.dx)
        return d * d


@dataclass(frozen=True, slots=True)
class NonlinearFlowFields:
    """Borrowed aquifer geometry used to build transmissivity from head.

    ``min_sat`` is the positive saturated-thickness floor that retains
    ellipticity of the *flow* operator.  It is deliberately separate from
    physical storage saturation, which is zero-to-full-thickness.
    """

    K: Any                # (ny, nx) hydraulic conductivity [L/T]
    zbot: Any             # (ny, nx) aquifer bottom [L]
    ztop: Any | None      # (ny, nx) aquifer top [L], or None
    min_sat: float        # min saturated thickness (flow ellipticity floor) [L]


@dataclass(frozen=True, slots=True)
class NonlinearBoundaryFields:
    """Borrowed active / Dirichlet / GHB operator data.

    ``ghb_factor`` is the conductance factor produced by
    ``physics.operator_data.compute_ghb_factor_from_raw_fields`` (the same value
    the production operator consumes).  ``ghb_external_head`` is the prescribed
    external stage for each GHB cell.
    """

    active: Any                # (ny, nx) int, !=0 active
    dirichlet_mask: Any        # (ny, nx) int, !=0 fixed-head
    dirichlet_values: Any      # (ny, nx) prescribed head [L]
    ghb_mask: Any              # (ny, nx) int, !=0 GHB
    ghb_external_head: Any     # (ny, nx) external stage [L]
    ghb_factor: Any            # (ny, nx) conductance factor


@dataclass(frozen=True, slots=True)
class NonlinearSourceField:
    """Borrowed signed net source term ``R_field``.

    ``R_field`` follows the production sign convention: positive values are
    recharge (inflow), negative values are aggregated source withdrawal
    (outflow).  Units are ``[L/T]`` and are multiplied by cell area inside the
    operator, matching ``build_rhs_fd_like``.
    """

    R_field: Any  # (ny, nx) signed net source [L/T]


@dataclass(frozen=True, slots=True)
class NonlinearStorageFields:
    """Borrowed transient convertible-storage inputs.

    When ``transient`` is False the operator evaluates the steady residual and
    ``sy``/``ss``/``head_prev``/``dt`` are unused.  When transient, the
    authoritative residual uses the *exact* convertible storage potentials of
    ``physics.storage_2d`` (specific-yield clipped-thickness change plus
    specific-storage potential change, divided by ``dt`` and scaled by area).
    """

    transient: bool
    sy: float                  # specific yield [L/L], >= 0
    ss: float                  # specific storage [1/L], >= 0
    head_prev: Any | None      # (ny, nx) previous-period head [L]
    dt: float | None           # timestep [T], > 0 when transient


@dataclass(frozen=True, slots=True)
class NonlinearOperatorContext2D:
    """Borrowed, non-owning physical state for the 2D nonlinear operator.

    All arrays are references to caller-owned data.  Building this context never
    transfers ownership: the caller is free to keep, mutate, or release the
    underlying arrays independently of the operator that consumes them.
    """

    grid: NonlinearGrid
    flow: NonlinearFlowFields
    boundaries: NonlinearBoundaryFields
    sources: NonlinearSourceField
    storage: NonlinearStorageFields

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    @property
    def ny(self) -> int:
        return self.grid.ny

    @property
    def nx(self) -> int:
        return self.grid.nx

    @property
    def free_mask(self) -> np.ndarray:
        """Boolean mask of free cells (active and non-Dirichlet)."""
        active = np.asarray(self.boundaries.active, dtype=np.int32) != 0
        dirichlet = np.asarray(self.boundaries.dirichlet_mask, dtype=np.int32) != 0
        return active & (~dirichlet)

    @property
    def n_free(self) -> int:
        return int(np.count_nonzero(self.free_mask))


def _as2d(arr: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} is required (got None).")
    a = np.asarray(arr, dtype=NP_FLOAT)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite.")
    return a


def _as_int2d(arr: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int32)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}.")
    return a


def from_arrays(
    *,
    nx: int,
    ny: int,
    dx: float,
    K: Any,
    zbot: Any,
    ztop: Any | None,
    active: Any,
    dirichlet_mask: Any,
    dirichlet_values: Any,
    R_field: Any,
    ghb_mask: Any | None = None,
    ghb_external_head: Any | None = None,
    ghb_factor: Any | None = None,
    sy: float = 0.0,
    ss: float = 0.0,
    head_prev: Any | None = None,
    dt: float | None = None,
    transient: bool = False,
    min_sat: float = 0.1,
    device: str = "cpu",
    wp_dtype: Any | None = None,
    np_dtype: Any | None = None,
) -> NonlinearOperatorContext2D:
    """Build a context directly from borrowed NumPy arrays.

    All array arguments are referenced (not copied unless a cast is required);
    scalar inputs are validated.  ``ghb_*`` default to zero fields when omitted
    so the same call shape serves GHB and non-GHB problems.
    """
    from .kernels import WP_FLOAT

    if wp_dtype is None:
        wp_dtype = WP_FLOAT
    if np_dtype is None:
        np_dtype = NP_FLOAT

    if int(nx) <= 0 or int(ny) <= 0:
        raise ValueError("nx and ny must be positive.")
    if float(dx) <= 0.0 or not np.isfinite(float(dx)):
        raise ValueError("dx must be positive and finite.")
    shape = (int(ny), int(nx))

    min_sat_f = float(min_sat)
    if min_sat_f <= 0.0 or not np.isfinite(min_sat_f):
        raise ValueError("min_sat must be positive and finite.")

    K_arr = _as2d(K, name="K", shape=shape)
    zbot_arr = _as2d(zbot, name="zbot", shape=shape)
    if np.any(K_arr < 0.0):
        raise ValueError("K must be non-negative.")
    ztop_arr = None
    if ztop is not None:
        ztop_arr = _as2d(ztop, name="ztop", shape=shape)

    active_arr = _as_int2d(active, name="active", shape=shape)
    dirichlet_mask_arr = _as_int2d(dirichlet_mask, name="dirichlet_mask", shape=shape)
    dirichlet_values_arr = _as2d(dirichlet_values, name="dirichlet_values", shape=shape)
    R_arr = _as2d(R_field, name="R_field", shape=shape)

    zeros = np.zeros(shape, dtype=NP_FLOAT)
    ghb_mask_arr = _as_int2d(ghb_mask if ghb_mask is not None else zeros, name="ghb_mask", shape=shape)
    ghb_head_arr = _as2d(ghb_external_head if ghb_external_head is not None else zeros, name="ghb_external_head", shape=shape)
    ghb_factor_arr = _as2d(ghb_factor if ghb_factor is not None else zeros, name="ghb_factor", shape=shape)

    sy_f = float(sy)
    ss_f = float(ss)
    if sy_f < 0.0 or not np.isfinite(sy_f):
        raise ValueError("sy must be non-negative and finite.")
    if ss_f < 0.0 or not np.isfinite(ss_f):
        raise ValueError("ss must be non-negative and finite.")

    head_prev_arr = None
    dt_f: float | None = None
    if bool(transient):
        dt_f = float(dt) if dt is not None else float("nan")
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            raise ValueError("transient=True requires dt > 0.")
        head_prev_arr = _as2d(head_prev, name="head_prev", shape=shape)
    else:
        sy_f = 0.0
        ss_f = 0.0

    return NonlinearOperatorContext2D(
        grid=NonlinearGrid(
            nx=int(nx),
            ny=int(ny),
            dx=float(dx),
            device=str(device),
            wp_dtype=wp_dtype,
            np_dtype=np_dtype,
        ),
        flow=NonlinearFlowFields(K=K_arr, zbot=zbot_arr, ztop=ztop_arr, min_sat=min_sat_f),
        boundaries=NonlinearBoundaryFields(
            active=active_arr,
            dirichlet_mask=dirichlet_mask_arr,
            dirichlet_values=dirichlet_values_arr,
            ghb_mask=ghb_mask_arr,
            ghb_external_head=ghb_head_arr,
            ghb_factor=ghb_factor_arr,
        ),
        sources=NonlinearSourceField(R_field=R_arr),
        storage=NonlinearStorageFields(
            transient=bool(transient),
            sy=sy_f,
            ss=ss_f,
            head_prev=head_prev_arr,
            dt=dt_f,
        ),
    )


def from_unconfined_solve_inputs(
    model: Any,
    *,
    K_field: Any,
    zbot_field: Any,
    ztop_field: Any | None = None,
    sy: float | None = None,
    ss: float | None = None,
    dt: float | None = None,
    head_prev: Any | None = None,
    min_sat: float | None = None,
    transient: bool = False,
    device: str | None = None,
) -> NonlinearOperatorContext2D:
    """Build a context from a ``WarpDarcySolver`` plus unconfined solve inputs.

    The model contributes grid metadata and the assembled operator fields it
    already owns (active / Dirichlet / GHB / recharge).  The unconfined physical
    inputs that the production solve threads as kwargs (``K_field``,
    ``zbot_field``, ``ztop_field``, ``sy``, ``ss``, ``dt``, ``head_prev``,
    ``min_sat``) are borrowed directly from the caller.  No model array is
    copied into the operator and no ownership is transferred.
    """
    ny = int(getattr(model, "ny"))
    nx = int(getattr(model, "nx"))
    dx = float(getattr(model, "dx"))
    dev = str(getattr(model, "device_str")) if device is None else str(device)

    active = np.asarray(getattr(model, "active_host"), dtype=np.int32)
    dirichlet_mask = np.asarray(getattr(model, "bc_mask_host"), dtype=np.int32)
    dirichlet_values = np.asarray(getattr(model, "bc_values_host"), dtype=NP_FLOAT)
    R_field = np.asarray(getattr(model, "R_field_host"), dtype=NP_FLOAT)

    ghb_mask = np.asarray(getattr(model, "gh_mask_host"), dtype=np.int32)
    ghb_external_head = np.asarray(getattr(model, "gh_head_host"), dtype=NP_FLOAT)
    ghb_factor = np.asarray(getattr(model, "ghb_factor_host"), dtype=NP_FLOAT)

    min_sat_eff = 0.1 if min_sat is None else float(min_sat)

    sy_eff = 0.0 if sy is None else float(sy)
    ss_eff = 0.0 if ss is None else float(ss)

    return from_arrays(
        nx=nx,
        ny=ny,
        dx=dx,
        K=K_field,
        zbot=zbot_field,
        ztop=ztop_field,
        active=active,
        dirichlet_mask=dirichlet_mask,
        dirichlet_values=dirichlet_values,
        R_field=R_field,
        ghb_mask=ghb_mask,
        ghb_external_head=ghb_external_head,
        ghb_factor=ghb_factor,
        sy=sy_eff,
        ss=ss_eff,
        head_prev=head_prev,
        dt=dt,
        transient=bool(transient),
        min_sat=min_sat_eff,
        device=dev,
    )
