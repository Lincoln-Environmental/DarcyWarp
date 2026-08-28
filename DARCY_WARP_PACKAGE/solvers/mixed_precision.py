# SPDX-License-Identifier: AGPL-3.0-only
"""EXPERIMENTAL mixed-precision defect-correction solver (steady, confined, 2D).

Status: **experimental, opt-in, non-production.**  This module is not part of
the solver registry, has no production alias, and is never selected by
default.  Numerically validated on the tested steady confined benchmark cases
(heterogeneous T, GHB, and isotropic no-GHB), but it provides no performance
advantage in the current K-cycle implementation — see
``MIXED_PRECISION_PLAN.md`` §3–4.  Retained as a reference for possible future
optimisation once FP32 K-cycles become materially cheaper than FP64 cycles.

Algorithm
---------
1. Maintain the authoritative absolute-head field ``h64`` in FP64, starting
   from the caller-supplied FP64 initial head (benchmarks use the original
   DEM).
2. Evaluate the true fine-grid defect ``r64 = b64 - A(h64)`` in FP64.
3. Transfer the defect into the FP32 correction hierarchy
   (``r32 = float32(r64)``; already zero on inactive/Dirichlet cells).
4. Approximately solve the homogeneous correction equation
   ``M32 * delta32 ~= r32`` with one configurable block of FP32 K-cycles on
   the existing multigrid hierarchy (zero Dirichlet correction; GHB diagonal
   retained; no external GHB-head source).
5. Accumulate the correction into the FP64 head (``h64 += delta32``) and
   re-pin Dirichlet cells to their exact FP64 boundary values.
6. Repeat from step 2 until the FP64 convergence criterion is met: the
   recomputed FP64 residual RMS reaches ``max(abs_tol_min, rel_tol * r_rms0)``
   and the head-change safeguards (``dh_max``/``dh_rms``) pass.  Convergence
   is decided exclusively from the recomputed FP64 residual — no recursively
   updated residual is ever trusted.

Precision ownership
-------------------
FP64: master head, fine RHS, Dirichlet/GHB head values, true-residual
evaluation, correction accumulation, convergence reductions, returned head.
FP32: the entire multigrid correction hierarchy (operators, smoothers, work
arrays, coarse levels), i.e. everything the existing K-cycle already owns when
the model is built under ``DARCY_FLOAT=float32``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

from .multigrid_kcycle import solve_kcycle_device_buffers

#: Capability metadata: this solver is experimental, opt-in, non-production.
EXPERIMENTAL = True

_EXPERIMENTAL_WARNING = (
    "MixedPrecisionDefectCorrectionSession is an EXPERIMENTAL, opt-in, "
    "non-production solver (steady confined 2D only). It is not part of the "
    "solver registry and must not be used as a production default; see "
    "MIXED_PRECISION_PLAN.md."
)


# ---------------------------------------------------------------------------
# Kernels (explicit dtypes; compiled independently of WP_FLOAT)
# ---------------------------------------------------------------------------


@wp.kernel
def _mp_residual_f64_kernel(
    x: wp.array(dtype=wp.float64, ndim=2),
    b: wp.array(dtype=wp.float64, ndim=2),
    T_field: wp.array(ndim=2),  # coefficient array (FP32 model storage)
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(ndim=2),  # coefficient array (FP32 model storage)
    r: wp.array(dtype=wp.float64, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    """True residual r64 = b64 - A(h64); zero on inactive/Dirichlet cells.

    Mirrors ``compute_residual_no_storage_kernel`` from warped_darcy.py but
    with FP64 head/RHS/residual state.  Coefficients (T, ghb_factor) are
    loaded from the model's dtype-generic arrays and promoted to FP64,
    exactly matching the row-wise FP64 arithmetic of the production kernels.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = wp.float64(0.0)
        return

    tiny = wp.float64(1.0e-12)

    T_c = wp.float64(T_field[j, i])
    hC = wp.float64(x[j, i])

    hE = wp.float64(0.0)
    hW = wp.float64(0.0)
    hN = wp.float64(0.0)
    hS = wp.float64(0.0)

    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        hE = wp.float64(x[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        hW = wp.float64(x[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        hN = wp.float64(x[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        hS = wp.float64(x[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = T_e + T_w + T_n + T_s + C_gh

    Ax64 = wp.float64(0.0)
    if sum_T < tiny:
        Ax64 = hC
    else:
        Ax64 = sum_T * hC
        if T_e > wp.float64(0.0):
            Ax64 = Ax64 - T_e * hE
        if T_w > wp.float64(0.0):
            Ax64 = Ax64 - T_w * hW
        if T_n > wp.float64(0.0):
            Ax64 = Ax64 - T_n * hN
        if T_s > wp.float64(0.0):
            Ax64 = Ax64 - T_s * hS

    rf64 = wp.float64(b[j, i]) - Ax64
    r[j, i] = rf64
    wp.atomic_add(rTr_buf, 0, rf64 * rf64)


@wp.kernel
def _mp_cast_r64_to_r32_kernel(
    r64: wp.array(dtype=wp.float64, ndim=2),
    r32: wp.array(dtype=wp.float32, ndim=2),
    nx: int,
    ny: int,
):
    """Transfer the FP64 defect into the FP32 correction hierarchy."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    r32[j, i] = wp.float32(r64[j, i])


@wp.kernel
def _mp_accumulate_kernel(
    h64: wp.array(dtype=wp.float64, ndim=2),
    delta32: wp.array(dtype=wp.float32, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values64: wp.array(dtype=wp.float64, ndim=2),
    dh_max_buf: wp.array(dtype=wp.float64, ndim=1),
    dh_sq_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    """h64 += delta32 on free cells; Dirichlet re-pinned exactly; dh stats."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        h64[j, i] = wp.float64(0.0)
        return

    if bc_mask[j, i] != 0:
        h64[j, i] = bc_values64[j, i]
        return

    dh = wp.float64(delta32[j, i])
    h64[j, i] = h64[j, i] + dh
    wp.atomic_add(dh_sq_buf, 0, dh * dh)
    wp.atomic_max(dh_max_buf, 0, wp.abs(dh))


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _build_rhs_f64_host(
    *,
    R_f64: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values_f64: np.ndarray,
    dx: float,
    gh_mask: np.ndarray | None,
    gh_head_f64: np.ndarray | None,
    T_coeff_host: np.ndarray,
    ghb_factor_host: np.ndarray | None,
) -> np.ndarray:
    """Assemble the fine-grid RHS in FP64 (host mirror of build_rhs_kernel).

    Coefficients (T, ghb_factor) are taken from the model's (FP32) host arrays
    and promoted to FP64 so that ``C_gh`` matches the device operator
    bit-for-bit; recharge, GHB *head values* and Dirichlet values are FP64.
    """
    R = np.asarray(R_f64, dtype=np.float64)
    b = R * (float(dx) ** 2)

    if gh_mask is not None and gh_head_f64 is not None and ghb_factor_host is not None:
        gh_mask_arr = np.asarray(gh_mask, dtype=np.int32)
        ghb = np.asarray(ghb_factor_host, dtype=np.float64)
        T_c = np.asarray(T_coeff_host, dtype=np.float64)
        C_gh = T_c * ghb
        mask_gh = (gh_mask_arr != 0) & (np.asarray(active) != 0) & np.isfinite(ghb) & (ghb > 0.0)
        if np.any(mask_gh):
            b[mask_gh] = b[mask_gh] + C_gh[mask_gh] * np.asarray(gh_head_f64, dtype=np.float64)[mask_gh]

    b = np.asarray(b, dtype=np.float64)
    b[np.asarray(active) == 0] = 0.0
    bc_idx = np.asarray(bc_mask) != 0
    b[bc_idx] = np.asarray(bc_values_f64, dtype=np.float64)[bc_idx]
    return b


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_MIXED_DEFAULTS = {
    "max_outer": 40,
    "inner_kcycles": 5,
    "nu_pre": 2,
    "nu_post": 2,
    "nu_coarse": 2,
    "omega": 0.7,
    "smoother": "chebyshev",
    "cheby_lambda_min": 0.1,
    "cheby_lambda_max": 2.0,
    "rel_tol": 5.0e-7,
    "abs_tol_min": 5.0e-7,
    "dh_rms_tol": 1.0e-4,
    "dh_max_tol": None,  # default: 5 * dh_rms_tol (mirrors K-cycle backend)
}


class MixedPrecisionDefectCorrectionSession:
    """Reusable buffers + hierarchy for repeated mixed-precision solves.

    EXPERIMENTAL, opt-in, non-production (see module docstring).  The model
    must be built under ``DARCY_FLOAT=float32`` so the entire K-cycle
    hierarchy is FP32; all FP64 master state lives in this session.  Creating
    the session performs no solve; each :meth:`solve` call is one complete
    solver invocation starting from the supplied FP64 initial head.
    """

    def __init__(
        self,
        model: Any,
        *,
        bc_values_f64: np.ndarray,
        gh_head_f64: np.ndarray | None,
        R_f64: np.ndarray,
        max_levels: int = 6,
        min_coarse_cells: int = 500,
        emit_experimental_warning: bool = True,
    ):
        from DARCY_WARP_PACKAGE import warped_darcy as kernel_module

        if kernel_module.WP_FLOAT is not wp.float32:
            raise RuntimeError(
                "MixedPrecisionDefectCorrectionSession requires the model to be "
                "built under DARCY_FLOAT=float32 (the multigrid hierarchy must "
                "be FP32); got WP_FLOAT="
                + str(kernel_module.WP_FLOAT)
            )

        if emit_experimental_warning:
            warnings.warn(_EXPERIMENTAL_WARNING, stacklevel=2)

        self.model = model
        self.device = str(model.device_str)
        self.nx = int(model.nx)
        self.ny = int(model.ny)
        shape = (self.ny, self.nx)

        if model._operator_dirty or model.mg_levels is None:
            model.build_hierarchy(
                max_levels=int(max_levels),
                min_coarse_n=4,
                min_coarse_cells=int(min_coarse_cells),
            )
        if model.mg_levels is None or len(model.mg_levels) < 1:
            raise RuntimeError("No multigrid levels available for mixed-precision solve.")

        # FP64 master state
        self.h64 = wp.zeros(shape, dtype=wp.float64, device=self.device)
        self.r64 = wp.zeros(shape, dtype=wp.float64, device=self.device)
        self.bc_values64 = wp.array(
            np.asarray(bc_values_f64, dtype=np.float64), dtype=wp.float64, device=self.device
        )

        # FP64 fine RHS (constant across the solve)
        b64_host = _build_rhs_f64_host(
            R_f64=R_f64,
            active=model.active_host,
            bc_mask=model.bc_mask_host,
            bc_values_f64=bc_values_f64,
            dx=float(model.dx),
            gh_mask=model.gh_mask_host if model.use_ghb else None,
            gh_head_f64=gh_head_f64 if model.use_ghb else None,
            T_coeff_host=np.asarray(model.T_field_host, dtype=np.float64),
            ghb_factor_host=model.ghb_factor_host,
        )
        self.b64 = wp.array(b64_host, dtype=wp.float64, device=self.device)

        # Host-side FP64 state needed to rebuild the RHS for ensemble
        # parameter updates (see update_rhs_f64).
        self._R_f64_host = np.asarray(R_f64, dtype=np.float64)
        self._bc_values_f64_host = np.asarray(bc_values_f64, dtype=np.float64)
        self._gh_head_f64_host = (
            None if gh_head_f64 is None else np.asarray(gh_head_f64, dtype=np.float64)
        )

        # Coefficient arrays used by the FP64 residual kernel (model storage)
        self.T_coeff_wp = model.T_wp
        self.ghb_coeff_wp = model.ghb_factor_wp

        # FP32 correction buffers
        self.r32 = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.delta32 = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.zero_bc32 = wp.zeros(shape, dtype=wp.float32, device=self.device)

        # FP64 scalar reduction buffers
        self.rTr_buf = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.dh_max_buf = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.dh_sq_buf = wp.zeros(1, dtype=wp.float64, device=self.device)

        self.n_free = int(
            np.count_nonzero((model.active_host != 0) & (model.bc_mask_host == 0))
        )
        if self.n_free <= 0:
            raise RuntimeError("No free (active, non-Dirichlet) cells.")

    # -- ensemble parameter updates -------------------------------------------

    def update_rhs_f64(self, R_f64: np.ndarray | None = None) -> None:
        """Refresh the FP64 fine RHS in place for ensemble parameter updates.

        Rebuilds ``b64`` from the model's *current* host coefficient state
        (transmissivity and GHB conductance factor) and the boundary heads
        captured at session construction.  Pass ``R_f64`` to also swap the
        recharge field; omit it to keep the current recharge (e.g. after an
        in-place T update).  The device array is updated in place, so eager
        kernels pick up the new values on the next :meth:`solve`.
        """
        if R_f64 is not None:
            self._R_f64_host = np.asarray(R_f64, dtype=np.float64)
        model = self.model
        b64_host = _build_rhs_f64_host(
            R_f64=self._R_f64_host,
            active=model.active_host,
            bc_mask=model.bc_mask_host,
            bc_values_f64=self._bc_values_f64_host,
            dx=float(model.dx),
            gh_mask=model.gh_mask_host if model.use_ghb else None,
            gh_head_f64=self._gh_head_f64_host if model.use_ghb else None,
            T_coeff_host=np.asarray(model.T_field_host, dtype=np.float64),
            ghb_factor_host=model.ghb_factor_host,
        )
        wp.copy(self.b64, wp.array(b64_host, dtype=wp.float64, device=self.device))

    # -- internal steps ------------------------------------------------------

    def _true_residual(self) -> float:
        """Recompute r64 from h64; return the FP64 sum of squared residuals."""
        self.rTr_buf.fill_(0.0)
        wp.launch(
            kernel=_mp_residual_f64_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self.h64,
                self.b64,
                self.T_coeff_wp,
                self.model.active_wp,
                self.model.bc_mask_wp,
                self.model.gh_mask_wp,
                self.ghb_coeff_wp,
                self.r64,
                self.rTr_buf,
                self.nx,
                self.ny,
            ],
            device=self.device,
        )
        return float(self.rTr_buf.numpy()[0])

    def _inner_correction_block(self, solve_controls: dict) -> None:
        """One configurable block of FP32 K-cycles on the correction equation."""
        model = self.model

        wp.launch(
            kernel=_mp_cast_r64_to_r32_kernel,
            dim=(self.ny, self.nx),
            inputs=[self.r64, self.r32, self.nx, self.ny],
            device=self.device,
        )
        self.delta32.fill_(wp.float32(0.0))

        lvl0 = model.mg_levels[0]
        front = (
            lvl0.x_wp,
            lvl0.b_wp,
            lvl0.T_wp,
            lvl0.storage_diag_wp,
            lvl0.active_wp,
            lvl0.bc_mask_wp,
            lvl0.bc_values_wp,
        )
        try:
            solve_kcycle_device_buffers(
                model=model,
                x_wp=self.delta32,
                rhs_wp=self.r32,
                T_wp=model.T_wp,
                storage_diag_wp=None,
                active_wp=model.active_wp,
                bc_mask_wp=model.bc_mask_wp,
                bc_values_wp=self.zero_bc32,
                levels=model.mg_levels,
                solve_controls=solve_controls,
                return_scalar_info=False,
                fixed_work_no_scalar_reads=True,
            )
        finally:
            (
                lvl0.x_wp,
                lvl0.b_wp,
                lvl0.T_wp,
                lvl0.storage_diag_wp,
                lvl0.active_wp,
                lvl0.bc_mask_wp,
                lvl0.bc_values_wp,
            ) = front

    def _accumulate(self) -> tuple[float, float]:
        """h64 += delta32 (FP64); returns (dh_max, dh_rms)."""
        self.dh_max_buf.fill_(0.0)
        self.dh_sq_buf.fill_(0.0)
        wp.launch(
            kernel=_mp_accumulate_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                self.h64,
                self.delta32,
                self.model.active_wp,
                self.model.bc_mask_wp,
                self.bc_values64,
                self.dh_max_buf,
                self.dh_sq_buf,
                self.nx,
                self.ny,
            ],
            device=self.device,
        )
        dh_max = float(self.dh_max_buf.numpy()[0])
        dh_rms = float(np.sqrt(max(float(self.dh_sq_buf.numpy()[0]), 0.0) / float(self.n_free)))
        return dh_max, dh_rms

    # -- public solve ---------------------------------------------------------

    def solve(self, initial_head_f64: np.ndarray, **controls: Any):
        """One complete mixed-precision solver invocation from ``initial_head_f64``.

        Every defect-correction iteration and every FP32 K-cycle is part of
        this single call.  Convergence uses only the recomputed FP64 true
        residual and the head-change safeguards.
        """
        cfg = dict(_MIXED_DEFAULTS)
        cfg.update(controls)

        dh_rms_tol = float(cfg["dh_rms_tol"])
        dh_max_tol = cfg["dh_max_tol"]
        dh_max_tol = (5.0 * dh_rms_tol) if dh_max_tol is None else float(dh_max_tol)

        inner_controls = {
            "max_cycles": int(cfg["inner_kcycles"]),
            "nu_pre": int(cfg["nu_pre"]),
            "nu_post": int(cfg["nu_post"]),
            "nu_coarse": int(cfg["nu_coarse"]),
            "omega": float(cfg["omega"]),
            "smoother": str(cfg["smoother"]),
            "cheby_lambda_min": float(cfg["cheby_lambda_min"]),
            "cheby_lambda_max": float(cfg["cheby_lambda_max"]),
            "check_every_no": int(cfg["inner_kcycles"]) + 1,  # unused in fixed-work mode
            "dh_rms_tol": None,
            "dh_max_tol": None,
        }

        # Initialise FP64 master head from the caller's (FP64) starting field.
        h0 = np.asarray(initial_head_f64, dtype=np.float64).copy()
        model = self.model
        h0[np.asarray(model.active_host) == 0] = 0.0
        bc_idx = np.asarray(model.bc_mask_host) != 0
        h0[bc_idx] = self.bc_values64.numpy()[bc_idx]
        wp.copy(self.h64, wp.array(h0, dtype=wp.float64, device=self.device))

        rTr = self._true_residual()
        r_rms0 = float(np.sqrt(max(rTr, 0.0) / float(self.n_free)))
        tol_abs = float(max(float(cfg["abs_tol_min"]), float(cfg["rel_tol"]) * r_rms0))
        thr_rTr = tol_abs * tol_abs * float(self.n_free)

        history = [
            {
                "outer": 0,
                "r_rms64": r_rms0,
                "dh_max": None,
                "dh_rms": None,
            }
        ]

        converged = False
        outer_used = 0
        if rTr <= thr_rTr:
            converged = True
        else:
            for k in range(1, int(cfg["max_outer"]) + 1):
                outer_used = k
                self._inner_correction_block(inner_controls)
                dh_max, dh_rms = self._accumulate()
                rTr = self._true_residual()
                r_rms = float(np.sqrt(max(rTr, 0.0) / float(self.n_free)))
                history.append(
                    {
                        "outer": k,
                        "r_rms64": r_rms,
                        "dh_max": dh_max,
                        "dh_rms": dh_rms,
                    }
                )
                res_ok = rTr <= thr_rTr
                dh_ok = (dh_max <= dh_max_tol) and (dh_rms <= dh_rms_tol)
                if res_ok and dh_ok:
                    converged = True
                    break

        head_out = self.h64.numpy().copy()
        info = {
            "precision_mode": "mixed_defect_correction",
            "experimental": True,
            "converged": bool(converged),
            "outer_iterations": int(outer_used),
            "inner_kcycles_per_outer": int(cfg["inner_kcycles"]),
            "total_kcycles": int(outer_used) * int(cfg["inner_kcycles"]),
            "r_rms0_64": r_rms0,
            "r_rms_end_64": history[-1]["r_rms64"],
            "dh_max_last": history[-1]["dh_max"],
            "dh_rms_last": history[-1]["dh_rms"],
            "tol_abs": tol_abs,
            "history": history,
        }
        return head_out, info


__all__ = [
    "EXPERIMENTAL",
    "MixedPrecisionDefectCorrectionSession",
]
