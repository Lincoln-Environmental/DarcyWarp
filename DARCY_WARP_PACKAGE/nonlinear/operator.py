# SPDX-License-Identifier: AGPL-3.0-only
"""Authoritative 2D unconfined nonlinear operator.

``NonlinearOperator2D`` evaluates the true nonlinear groundwater equation
directly from hydraulic head, on device, using the kernels in
:mod:`DARCY_WARP_PACKAGE.nonlinear.kernels`.  It is backend-neutral: it contains
no Picard iteration, damping, relaxation, acceleration, acceptance, fallback, or
convergence logic, and it does not call the production Picard backend.

Ownership model (mirrors the model/``SolverContext`` split):

* the *physical data* is borrowed read-only via :class:`NonlinearOperatorContext2D`;
  nothing is copied away from the caller and no model array is claimed;
* the operator *owns* its persistent device scratch and its own device mirrors
  of the borrowed static fields.  These mirrors are uploaded once at
  construction and never re-allocated per evaluation, so repeated residual
  evaluations do not grow device memory.

Per evaluation the only host/device traffic is one head upload (H2D) plus the
scalar reductions needed for norms (D2H).  There are no host-side per-cell loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from . import kernels as _k
from .context import NonlinearOperatorContext2D


@dataclass(frozen=True, slots=True)
class StorageTerms2D:
    """Host view of the exact convertible storage terms (diagnostic)."""

    total: np.ndarray   # Sy + Ss storage flux per cell [L^3/T]
    sy: np.ndarray      # specific-yield term [L^3/T]
    ss: np.ndarray      # specific-storage term [L^3/T]
    sat_physical: np.ndarray  # physical storage saturation (zero floor) [L]


@dataclass(frozen=True, slots=True)
class FrozenPicardOperator2D:
    """Coefficients of the production Picard linearisation frozen at ``head``.

    These are exactly the arrays the trusted ``unconfined_picard_kcycle``
    backend assembles at this head (transmissivity with the ``min_sat`` floor and
    the MF6 secant specific-yield / specific-storage coefficients), produced here
    *without* invoking the Picard algorithm.  They let a future Newton/FAS solver
    build a Picard-skeletal operator for comparison or hybridisation.

    The Ss coefficient uses the ``min_sat``-floored saturation of the production
    linearisation; the authoritative residual (:meth:`NonlinearOperator2D.residual`)
    uses the zero-floor exact Ss potential instead.  The two agree as ``dh -> 0``.
    """

    transmissivity: np.ndarray
    sy_coeff: np.ndarray
    ss_coeff: np.ndarray
    storage_diag: np.ndarray  # (sy_coeff + ss_coeff) * area / dt
    area: float
    dt: float | None
    min_sat: float


@dataclass(frozen=True, slots=True)
class ResidualNorms:
    """Free-cell residual norms (inactive/Dirichlet rows excluded)."""

    rms: float
    max_abs: float
    l2: float
    n_free: int


class NonlinearOperator2D:
    """Device evaluator of the 2D unconfined nonlinear residual."""

    def __init__(self, context: NonlinearOperatorContext2D):
        self.ctx = context
        self._ensure_wp_init()

        grid = context.grid
        device = grid.device
        WP = grid.wp_dtype
        ny, nx = grid.ny, grid.nx
        shape = (ny, nx)
        self._device = device
        self._WP = WP
        self._ny = ny
        self._nx = nx
        self._dim = shape

        flow = context.flow
        bnd = context.boundaries
        sto = context.storage

        self._has_ztop = 1 if flow.ztop is not None else 0
        self._has_storage = 1 if sto.transient else 0
        self._min_sat = float(flow.min_sat)
        self._area = float(grid.area)
        self._dt = float(sto.dt) if (sto.transient and sto.dt is not None) else float("nan")
        self._inv_dt = (1.0 / self._dt) if self._has_storage else 0.0
        self._sy = float(sto.sy)
        self._ss = float(sto.ss)
        self._n_free = int(context.n_free)

        # ---- owned device mirrors of borrowed static fields (uploaded once) ----
        def _f64_field(name: str, arr: Any) -> Any:
            a = np.ascontiguousarray(np.asarray(arr, dtype=grid.np_dtype))
            if a.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {a.shape}.")
            return wp.array(a, dtype=WP, device=device)

        def _i32_field(name: str, arr: Any) -> Any:
            a = np.ascontiguousarray(np.asarray(arr, dtype=np.int32))
            if a.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {a.shape}.")
            return wp.array(a, dtype=wp.int32, device=device)

        self._K_wp = _f64_field("K", flow.K)
        self._zbot_wp = _f64_field("zbot", flow.zbot)
        self._ztop_wp = _f64_field("ztop", flow.ztop if flow.ztop is not None else np.zeros(shape, dtype=grid.np_dtype))
        self._active_wp = _i32_field("active", bnd.active)
        self._dirichlet_mask_wp = _i32_field("dirichlet_mask", bnd.dirichlet_mask)
        self._dirichlet_values_wp = _f64_field("dirichlet_values", bnd.dirichlet_values)
        self._gh_mask_wp = _i32_field("ghb_mask", bnd.ghb_mask)
        self._gh_head_wp = _f64_field("ghb_external_head", bnd.ghb_external_head)
        self._ghb_factor_wp = _f64_field("ghb_factor", bnd.ghb_factor)
        self._R_field_wp = _f64_field("R_field", context.sources.R_field)

        if sto.transient:
            self._head_prev_wp = _f64_field("head_prev", sto.head_prev)
        else:
            self._head_prev_wp = _f64_field("head_prev", np.zeros(shape, dtype=grid.np_dtype))

        free_mask_int = (context.free_mask.astype(np.int32))
        self._free_mask_wp = _i32_field("free_mask", free_mask_int)

        # ---- owned persistent scratch (never re-allocated per evaluation) ----
        self._head_wp = wp.zeros(shape, dtype=WP, device=device)
        self._F_wp = wp.zeros(shape, dtype=WP, device=device)
        self._sat_flow_wp = wp.zeros(shape, dtype=WP, device=device)
        self._T_wp = wp.zeros(shape, dtype=WP, device=device)
        self._store_total_wp = wp.zeros(shape, dtype=WP, device=device)
        self._store_sy_wp = wp.zeros(shape, dtype=WP, device=device)
        self._store_ss_wp = wp.zeros(shape, dtype=WP, device=device)
        self._sat_phys_wp = wp.zeros(shape, dtype=WP, device=device)
        self._pic_T_wp = wp.zeros(shape, dtype=WP, device=device)
        self._pic_sy_wp = wp.zeros(shape, dtype=WP, device=device)
        self._pic_ss_wp = wp.zeros(shape, dtype=WP, device=device)
        self._pic_diag_wp = wp.zeros(shape, dtype=WP, device=device)
        self._vector_wp = wp.zeros(shape, dtype=WP, device=device)
        self._Jv_wp = wp.zeros(shape, dtype=WP, device=device)

        self._rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
        self._Fmax_buf = wp.zeros(1, dtype=wp.float64, device=device)

        # CPU staging buffer for the per-evaluation head upload.
        self._head_stage = wp.zeros(shape, dtype=WP, device="cpu")
        self._vector_stage = wp.zeros(shape, dtype=WP, device="cpu")

        self._closed = False

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _ensure_wp_init() -> None:
        # Warp initialisation is idempotent; safe to call repeatedly.
        try:
            wp.init()
        except Exception:
            # Already initialised in some builds raises; that is harmless here.
            pass

    @property
    def device(self) -> str:
        return self._device

    @property
    def n_free(self) -> int:
        return self._n_free

    def _stage_head(self, head: Any) -> None:
        a = np.asarray(head, dtype=self.ctx.grid.np_dtype)
        if a.shape != self._dim:
            raise ValueError(f"head must have shape {self._dim}, got {a.shape}.")
        if not np.all(np.isfinite(a)):
            raise ValueError("head must be finite.")
        self._head_stage.numpy()[...] = a
        wp.copy(self._head_wp, self._head_stage)

    def _stage_vector(self, vector: Any) -> None:
        a = np.asarray(vector, dtype=self.ctx.grid.np_dtype)
        if a.shape != self._dim:
            raise ValueError(f"vector must have shape {self._dim}, got {a.shape}.")
        if not np.all(np.isfinite(a)):
            raise ValueError("vector must be finite.")
        self._vector_stage.numpy()[...] = a
        wp.copy(self._vector_wp, self._vector_stage)

    def _zero_reduction_bufs(self) -> None:
        self._rTr_buf.fill_(wp.float64(0.0))
        self._Fmax_buf.fill_(wp.float64(0.0))

    # ----------------------------------------------------------- required API

    def refresh_coefficients(self, head: Any) -> None:
        """Rebuild the head-dependent flow coefficients ``T(h)``.

        Populates the operator's persistent transmissivity / flow-saturation
        device buffers.  No host/device transfer is returned; use
        :meth:`saturated_thickness` or :meth:`frozen_picard_operator` to read
        coefficients back to host when required.
        """
        self._stage_head(head)
        wp.launch(
            kernel=_k.nl_flow_transmissivity_kernel,
            dim=self._dim,
            inputs=[
                self._head_wp, self._K_wp, self._zbot_wp, self._has_ztop, self._ztop_wp,
                float(self._min_sat), self._T_wp, self._sat_flow_wp,
                self._nx, self._ny,
            ],
            device=self._device,
        )

    def residual(self, head: Any, out: Any | None = None) -> Any:
        """Evaluate ``F(head) = A(head) head - b(head)`` into ``out``.

        ``out`` defaults to the operator's persistent residual scratch and is
        returned (a device array).  Inactive and Dirichlet rows are written as
        zero so they do not contaminate free-cell norms.  The reduction buffers
        for :meth:`residual_norms` are refreshed here as a side effect.
        """
        self._stage_head(head)
        return self.residual_device(self._head_wp, out=out, reduce=True)

    def residual_device(self, head: Any, out: Any | None = None, *, reduce: bool = True) -> Any:
        """Evaluate the Stage-1 residual from a device-resident head array.

        This is the allocation-free hot-path API used by nonlinear backends.
        ``reduce=False`` suppresses norm reductions when only the vector is
        required (for example inside a validation finite difference).
        """
        F_wp = self._F_wp if out is None else out
        if tuple(head.shape) != self._dim:
            raise ValueError(f"device head must have shape {self._dim}.")
        self._zero_reduction_bufs()
        wp.launch(
            kernel=_k.nl_residual_kernel,
            dim=self._dim,
            inputs=[
                head, self._K_wp, self._zbot_wp, self._has_ztop, self._ztop_wp,
                float(self._min_sat),
                self._active_wp, self._dirichlet_mask_wp, self._gh_mask_wp,
                self._gh_head_wp, self._ghb_factor_wp, self._R_field_wp, self._head_prev_wp,
                float(self._sy), float(self._ss), float(self._area), float(self._inv_dt),
                self._has_storage,
                F_wp, self._rTr_buf, self._Fmax_buf,
                self._nx, self._ny,
            ],
            device=self._device,
        )
        return F_wp

    def jacobian_vector(self, head: Any, vector: Any, out: Any | None = None) -> Any:
        """Apply the analytic deterministic generalized Jacobian to ``vector``."""
        self._stage_head(head)
        self._stage_vector(vector)
        return self.jacobian_vector_device(self._head_wp, self._vector_wp, out=out)

    def jacobian_vector_device(self, head: Any, vector: Any, out: Any | None = None) -> Any:
        """Allocation-free device action ``out = J(head) @ vector``.

        Warp automatic differentiation is intentionally not used.  The kernel
        implements the analytic face-conductance and exact-storage derivatives
        of the authoritative Stage-1 residual.
        """
        Jv_wp = self._Jv_wp if out is None else out
        wp.launch(
            kernel=_k.nl_jacobian_vector_kernel,
            dim=self._dim,
            inputs=[
                head, vector, self._K_wp, self._zbot_wp, self._has_ztop, self._ztop_wp,
                float(self._min_sat), self._active_wp, self._dirichlet_mask_wp,
                self._gh_mask_wp, self._gh_head_wp, self._ghb_factor_wp,
                float(self._sy), float(self._ss), float(self._area), float(self._inv_dt),
                self._has_storage, Jv_wp, self._nx, self._ny,
            ],
            device=self._device,
        )
        return Jv_wp

    def freeze_picard_device(self, head: Any) -> tuple[Any, Any]:
        """Refresh and return device ``(T, storage_diag)`` Picard coefficients."""
        wp.launch(
            kernel=_k.nl_picard_freeze_kernel,
            dim=self._dim,
            inputs=[
                head, self._head_prev_wp, self._K_wp, self._zbot_wp,
                self._has_ztop, self._ztop_wp, float(self._min_sat),
                float(self._sy), float(self._ss), float(self._area), float(self._inv_dt),
                self._active_wp, self._free_mask_wp,
                self._pic_T_wp, self._pic_sy_wp, self._pic_ss_wp, self._pic_diag_wp,
                self._nx, self._ny,
            ],
            device=self._device,
        )
        return self._pic_T_wp, self._pic_diag_wp

    def set_head_device(self, head: Any) -> None:
        """Copy a device head into the operator-owned accepted-head buffer."""
        wp.copy(self._head_wp, head)

    def set_head(self, head: Any) -> None:
        """Stage a finite host head as the current accepted device iterate."""
        self._stage_head(head)

    @property
    def head_device(self) -> Any:
        return self._head_wp

    @property
    def residual_device_array(self) -> Any:
        return self._F_wp

    @property
    def active_device(self) -> Any:
        return self._active_wp

    @property
    def dirichlet_mask_device(self) -> Any:
        return self._dirichlet_mask_wp

    @property
    def dirichlet_values_device(self) -> Any:
        return self._dirichlet_values_wp

    @property
    def ghb_mask_device(self) -> Any:
        return self._gh_mask_wp

    @property
    def ghb_factor_device(self) -> Any:
        return self._ghb_factor_wp

    @property
    def frozen_transmissivity_device(self) -> Any:
        return self._pic_T_wp

    @property
    def frozen_storage_diagonal_device(self) -> Any:
        return self._pic_diag_wp

    def current_reduced_norms(self) -> ResidualNorms:
        """Read norms produced by the most recent reduced residual launch."""
        rTr = float(self._rTr_buf.numpy()[0])
        Fmax = float(self._Fmax_buf.numpy()[0])
        rms = float(np.sqrt(max(rTr, 0.0) / float(self._n_free))) if self._n_free else 0.0
        return ResidualNorms(rms=rms, max_abs=Fmax, l2=float(np.sqrt(max(rTr, 0.0))), n_free=self._n_free)

    def residual_norms(self, head: Any) -> ResidualNorms:
        """Free-cell RMS / max-abs / L2 norms of the nonlinear residual.

        Two scalar device->host reads are required (``rTr`` and ``Fmax``); these
        are the only host transfers beyond the head upload.
        """
        self.residual(head)
        rTr = float(self._rTr_buf.numpy()[0])
        Fmax = float(self._Fmax_buf.numpy()[0])
        n_free = self._n_free
        rms = float(np.sqrt(max(rTr, 0.0) / float(n_free))) if n_free > 0 else 0.0
        return ResidualNorms(rms=rms, max_abs=Fmax, l2=float(np.sqrt(max(rTr, 0.0))), n_free=n_free)

    def saturated_thickness(self, head: Any, out: Any | None = None) -> Any:
        """Flow saturated thickness ``clip(head - bottom, min_sat, max(top-bottom, min_sat))``.

        This is the *flow* saturation with the positive ``min_sat`` ellipticity
        floor -- intentionally distinct from the zero-floor physical storage
        saturation exposed by :meth:`exact_storage_terms`.
        """
        sat_wp = self._sat_flow_wp if out is None else out
        self._stage_head(head)
        wp.launch(
            kernel=_k.nl_flow_transmissivity_kernel,
            dim=self._dim,
            inputs=[
                self._head_wp, self._K_wp, self._zbot_wp, self._has_ztop, self._ztop_wp,
                float(self._min_sat), self._T_wp, sat_wp,
                self._nx, self._ny,
            ],
            device=self._device,
        )
        return sat_wp

    def exact_storage_terms(self, head: Any, out: Any | None = None) -> StorageTerms2D:
        """Exact Sy / Ss / total convertible storage flux per cell ``[L^3/T]``.

        Uses the repository's exact physical storage potentials
        (specific-yield clipped-thickness change plus specific-storage potential
        change, divided by ``dt`` and scaled by cell area).  Physical storage
        saturation uses the zero-to-full-thickness clipping of
        :mod:`physics.storage_2d`; the ``min_sat`` flow floor is not introduced.
        """
        self._stage_head(head)
        wp.launch(
            kernel=_k.nl_exact_storage_kernel,
            dim=self._dim,
            inputs=[
                self._head_wp, self._head_prev_wp, self._zbot_wp, self._has_ztop, self._ztop_wp,
                float(self._sy), float(self._ss), float(self._area), float(self._inv_dt),
                self._has_storage, self._free_mask_wp,
                self._store_total_wp, self._store_sy_wp, self._store_ss_wp, self._sat_phys_wp,
                self._nx, self._ny,
            ],
            device=self._device,
        )
        return StorageTerms2D(
            total=np.asarray(self._store_total_wp.numpy(), dtype=np.float64).copy(),
            sy=np.asarray(self._store_sy_wp.numpy(), dtype=np.float64).copy(),
            ss=np.asarray(self._store_ss_wp.numpy(), dtype=np.float64).copy(),
            sat_physical=np.asarray(self._sat_phys_wp.numpy(), dtype=np.float64).copy(),
        )

    def frozen_picard_operator(self, head: Any) -> FrozenPicardOperator2D:
        """Return the production Picard linearisation coefficients at ``head``.

        Provides ``T(h)`` (``min_sat``-floored), the MF6 secant Sy coefficient,
        the secant Ss coefficient (``min_sat``-floored saturation, matching the
        production linearisation), and the resulting storage diagonal -- without
        running the Picard algorithm.  Read back to host because this is a
        compatibility/diagnostic bridge, not a hot solver primitive.
        """
        self._stage_head(head)
        wp.launch(
            kernel=_k.nl_picard_freeze_kernel,
            dim=self._dim,
            inputs=[
                self._head_wp, self._head_prev_wp, self._K_wp, self._zbot_wp,
                self._has_ztop, self._ztop_wp, float(self._min_sat),
                float(self._sy), float(self._ss), float(self._area), float(self._inv_dt),
                self._active_wp, self._free_mask_wp,
                self._pic_T_wp, self._pic_sy_wp, self._pic_ss_wp, self._pic_diag_wp,
                self._nx, self._ny,
            ],
            device=self._device,
        )
        return FrozenPicardOperator2D(
            transmissivity=np.asarray(self._pic_T_wp.numpy(), dtype=np.float64).copy(),
            sy_coeff=np.asarray(self._pic_sy_wp.numpy(), dtype=np.float64).copy(),
            ss_coeff=np.asarray(self._pic_ss_wp.numpy(), dtype=np.float64).copy(),
            storage_diag=np.asarray(self._pic_diag_wp.numpy(), dtype=np.float64).copy(),
            area=float(self._area),
            dt=(float(self._dt) if self._has_storage else None),
            min_sat=float(self._min_sat),
        )

    # ------------------------------------------------------------- lifecycle

    def scratch_arrays(self) -> tuple[Any, ...]:
        """Return the owned persistent scratch device arrays.

        Exposed so tests can verify that repeated evaluations reuse the same
        allocations (no device-memory growth) rather than reallocating.
        """
        return (
            self._head_wp, self._F_wp, self._sat_flow_wp, self._T_wp,
            self._store_total_wp, self._store_sy_wp, self._store_ss_wp, self._sat_phys_wp,
            self._pic_T_wp, self._pic_sy_wp, self._pic_ss_wp, self._pic_diag_wp,
            self._vector_wp, self._Jv_wp, self._rTr_buf, self._Fmax_buf,
            self._head_stage, self._vector_stage,
        )

    def close(self) -> None:
        """Drop owned device references (mirrors the model resource-owner pattern)."""
        if self._closed:
            return
        for name in (
            "_K_wp", "_zbot_wp", "_ztop_wp", "_active_wp", "_dirichlet_mask_wp",
            "_dirichlet_values_wp",
            "_gh_mask_wp", "_gh_head_wp", "_ghb_factor_wp", "_R_field_wp",
            "_head_prev_wp", "_free_mask_wp", "_head_wp", "_F_wp", "_sat_flow_wp",
            "_T_wp", "_store_total_wp", "_store_sy_wp", "_store_ss_wp", "_sat_phys_wp",
            "_pic_T_wp", "_pic_sy_wp", "_pic_ss_wp", "_pic_diag_wp", "_rTr_buf",
            "_Fmax_buf", "_head_stage", "_vector_wp", "_Jv_wp", "_vector_stage",
        ):
            setattr(self, name, None)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
