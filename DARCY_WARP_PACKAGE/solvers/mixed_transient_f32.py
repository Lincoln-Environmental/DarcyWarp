# SPDX-License-Identifier: AGPL-3.0-only
"""Production FP32 inner correction K-cycle for the 2D transient unconfined
device fast path (originating in Phase C of ``UNCONFINED_FAST_PLAN.md``,
Option 1: FP32 inner linear solve per Picard outer iteration).

Selected by the production replay through the
``transient_mixed_precision_enabled`` control. It requires the face operator
and may also be selected with ``DARCY_TRANSIENT_MIXED=1``.

Structure: the strict Picard gate already evaluates the TRUE FP64 dual
residual every outer iteration, so the inner linear solve may be
approximate as long as the outer loop keeps contracting — the same reason
the confined mixed-precision structure works
(``MIXED_PRECISION_CAMPAIGN.md``).  Per outer iteration the driver (in
correction form, identical to ``use_incremental_picard``) materialises the
FP64 nonlinear residual ``r = b - A*h^k``; this session casts it to FP32,
runs fixed-work FP32 K-cycles for the correction ``A32*delta32 = r32``
(Jacobi-block coarsest, block-reduced FP64 Krylov partials — the
``solvers/mixed_fast_kernels.py`` pattern), and casts the accumulated
correction back to FP64.  FP32 representation error then scales with the
CORRECTION magnitude (``~6e-8 * |delta|``), not with the head magnitude —
a plain FP32 solve for the head itself would floor the FP64 head residual
at ``~6e-8 * |h|`` (~5e-6 m at h~80 m) and starve the strict inner
targets.

The FP32 face operator is rebuilt every outer from the (FP64) level
transmissivity/storage values with the identical harmonic formula
(``face_build_storage_f32_kernel``), so the FP32 operator is the FP32
rounding of the FP64 linearisation.  All session buffers are allocated
once per solve call and refreshed in place (pointer-stable for CUDA-graph
capture; the per-outer FP32 face build rides inside the driver's captured
refresh graph, and the FP32 K-cycle itself is captured once and replayed
per fixed-work block cycle, mirroring the FP64 face path).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .face_kernels_f64 import (
    _BLOCK,
    combine_partials_kernel,
)
from .mixed_fast_kernels import (
    _mf3_applyA_dot_partials_f32,
    _mf3_dot_partials_f32,
    _mf3_jacobi_f32,
    _mf3_residual_f32,
)
from .mixed_precision import _mp_cast_r64_to_r32_kernel

# ---------------------------------------------------------------------------
# Kernels (explicit FP32 arithmetic; untyped array params so FP64 sources
# can be read directly)
# ---------------------------------------------------------------------------


@wp.kernel
def face_build_storage_f32_kernel(
    T_field: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(ndim=2),
    storage_diag: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    nx: int,
    ny: int,
):
    """Storage-aware face conductances + diagonal in FP32 (identical
    harmonic formula and addition order as ``face_build_storage_f64_kernel``;
    reads may be FP64 arrays, every arithmetic step is FP32)."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        Te[j, i] = wp.float32(0.0)
        Tw[j, i] = wp.float32(0.0)
        Tn[j, i] = wp.float32(0.0)
        Ts[j, i] = wp.float32(0.0)
        diag[j, i] = wp.float32(1.0)
        return

    tiny = wp.float32(1.0e-12)
    T_c = wp.float32(T_field[j, i])

    t_e = wp.float32(0.0)
    t_w = wp.float32(0.0)
    t_n = wp.float32(0.0)
    t_s = wp.float32(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float32(T_field[j, i + 1])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_e = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)
    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float32(T_field[j, i - 1])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_w = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)
    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float32(T_field[j - 1, i])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_n = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)
    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float32(T_field[j + 1, i])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_s = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float32(0.0)
    if gh_mask[j, i] != 0:
        ghbf = wp.float32(ghb_factor[j, i])
        if ghbf > wp.float32(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = t_e + t_w + t_n + t_s + C_gh + wp.float32(storage_diag[j, i])
    if sum_T < tiny:
        Te[j, i] = wp.float32(0.0)
        Tw[j, i] = wp.float32(0.0)
        Tn[j, i] = wp.float32(0.0)
        Ts[j, i] = wp.float32(0.0)
        diag[j, i] = wp.float32(1.0)
    else:
        Te[j, i] = t_e
        Tw[j, i] = t_w
        Tn[j, i] = t_n
        Ts[j, i] = t_s
        diag[j, i] = sum_T


@wp.kernel
def cast_r32_to_r64_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    dst: wp.array(dtype=wp.float64, ndim=2),
    nx: int,
    ny: int,
):
    """Transfer the FP32 correction back to the FP64 master head."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    dst[j, i] = wp.float64(src[j, i])


@wp.kernel
def restrict_blockavg_f32_kernel(
    r_f: wp.array(ndim=2),
    active_f: wp.array(dtype=wp.int32, ndim=2),
    bc_mask_f: wp.array(dtype=wp.int32, ndim=2),
    b_c: wp.array(ndim=2),
    nx_f: int,
    ny_f: int,
    nx_c: int,
    ny_c: int,
):
    """Block-average restriction, FP32 (same semantics as
    ``restrict_blockavg_kernel``: mean over free cells, else 0)."""
    jc, ic = wp.tid()
    if jc >= ny_c or ic >= nx_c:
        return
    j0 = 2 * jc
    i0 = 2 * ic
    s = wp.float32(0.0)
    n = wp.float32(0.0)
    if j0 < ny_f and i0 < nx_f:
        if active_f[j0, i0] != 0 and bc_mask_f[j0, i0] == 0:
            s = s + wp.float32(r_f[j0, i0])
            n = n + wp.float32(1.0)
    if j0 < ny_f and (i0 + 1) < nx_f:
        if active_f[j0, i0 + 1] != 0 and bc_mask_f[j0, i0 + 1] == 0:
            s = s + wp.float32(r_f[j0, i0 + 1])
            n = n + wp.float32(1.0)
    if (j0 + 1) < ny_f and i0 < nx_f:
        if active_f[j0 + 1, i0] != 0 and bc_mask_f[j0 + 1, i0] == 0:
            s = s + wp.float32(r_f[j0 + 1, i0])
            n = n + wp.float32(1.0)
    if (j0 + 1) < ny_f and (i0 + 1) < nx_f:
        if active_f[j0 + 1, i0 + 1] != 0 and bc_mask_f[j0 + 1, i0 + 1] == 0:
            s = s + wp.float32(r_f[j0 + 1, i0 + 1])
            n = n + wp.float32(1.0)
    if n > wp.float32(0.0):
        b_c[jc, ic] = s / n
    else:
        b_c[jc, ic] = wp.float32(0.0)


@wp.kernel
def prolong_bilinear_f32_kernel(
    x_c: wp.array(ndim=2),
    e_f: wp.array(ndim=2),
    nx_f: int,
    ny_f: int,
    nx_c: int,
    ny_c: int,
):
    """Bilinear prolongation with clamped edges, FP32 (same indexing as
    ``prolong_bilinear_any_kernel``)."""
    j, i = wp.tid()
    if j >= ny_f or i >= nx_f:
        return
    jc = j // 2
    ic = i // 2
    fy = wp.float32(0.0)
    fx = wp.float32(0.0)
    if (j & 1) == 1:
        fy = wp.float32(0.5)
    if (i & 1) == 1:
        fx = wp.float32(0.5)
    ic1 = ic + 1
    jc1 = jc + 1
    if ic1 >= nx_c:
        ic1 = nx_c - 1
    if jc1 >= ny_c:
        jc1 = ny_c - 1
    v00 = wp.float32(x_c[jc, ic])
    v10 = wp.float32(x_c[jc, ic1])
    v01 = wp.float32(x_c[jc1, ic])
    v11 = wp.float32(x_c[jc1, ic1])
    one = wp.float32(1.0)
    e_f[j, i] = (
        (one - fx) * (one - fy) * v00
        + fx * (one - fy) * v10
        + (one - fx) * fy * v01
        + fx * fy * v11
    )


@wp.kernel
def add_correction_f32_kernel(
    x_f: wp.array(ndim=2),
    e_f: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(ndim=2),
    nx: int,
    ny: int,
):
    """Fine-grid correction add, FP32 (same semantics as
    ``add_correction_kernel``)."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0:
        x_f[j, i] = wp.float32(0.0)
        return
    if bc_mask[j, i] != 0:
        x_f[j, i] = wp.float32(bc_values[j, i])
        return
    x_f[j, i] = wp.float32(x_f[j, i]) + wp.float32(e_f[j, i])


@wp.kernel
def copy_field_f32_kernel(
    src: wp.array(ndim=2),
    dst: wp.array(ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    dst[j, i] = src[j, i]


@wp.kernel
def axpy_active_scalar_2dmask_f32_kernel(
    y: wp.array(ndim=2),
    x: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    alpha_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    """``y += alpha * x`` on free cells, FP32 fields with the FP64 Krylov
    scalar (same semantics as ``axpy_active_scalar_2dmask_kernel``)."""
    iy, ix = wp.tid()
    if iy >= ny or ix >= nx:
        return
    if active[iy, ix] == 0:
        return
    if bc_mask[iy, ix] != 0:
        return
    a = wp.float32(alpha_buf[0])
    y[iy, ix] = wp.float32(y[iy, ix]) + a * wp.float32(x[iy, ix])


# ---------------------------------------------------------------------------
# Session: FP32 face operator + fixed-work correction K-cycle
# ---------------------------------------------------------------------------


class _F32Level:
    """FP32 face arrays + FP32 work buffers for one hierarchy level."""

    __slots__ = ("nx", "ny", "Te", "Tw", "Tn", "Ts", "diag",
                 "x", "b", "r", "Ax", "e", "z",
                 "partials", "rho_buf", "pAp_buf", "alpha_buf", "n_partials")

    def __init__(self, level, device: str):
        self.nx = int(level.nx)
        self.ny = int(level.ny)
        shape = (self.ny, self.nx)
        f32 = wp.float32
        for name in ("Te", "Tw", "Tn", "Ts", "diag", "x", "b", "r", "Ax", "e", "z"):
            setattr(self, name, wp.zeros(shape, dtype=f32, device=device))
        n_cells = self.nx * self.ny
        self.n_partials = (n_cells + _BLOCK - 1) // _BLOCK
        self.partials = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.rho_buf = wp.zeros(1, dtype=wp.float64, device=device)
        self.pAp_buf = wp.zeros(1, dtype=wp.float64, device=device)
        self.alpha_buf = wp.zeros(1, dtype=wp.float64, device=device)


class MixedTransientInnerSession:
    """FP32 inner correction K-cycle workspace for one transient solve call.

    All buffers are allocated once (pointer-stable for CUDA-graph capture)
    and refreshed in place: FP32 faces via :meth:`refresh_faces` every
    Picard outer iteration (the operator changes with T(h)/storage), the
    FP32 residual/correction via :meth:`begin_outer` /
    :meth:`sync_correction_in`, and the fixed-work K-cycle blocks via
    :meth:`solve_block` (CUDA-graph captured once, replayed per cycle).
    """

    def __init__(self, model: Any, levels) -> None:
        from DARCY_WARP_PACKAGE import warped_darcy as kernel_module

        self.model = model
        self.levels = levels
        device = str(model.device_str)
        self.device = device
        self._chebyshev_relaxation_sequence = kernel_module._chebyshev_relaxation_sequence
        self._compute_safe_alpha_kernel = kernel_module.compute_safe_alpha_kernel
        self.f32_levels = [_F32Level(level, device) for level in levels]
        # The correction equation is pinned to ZERO on Dirichlet cells
        # (delta = 0 where the head is fixed).  Coarse levels already carry
        # homogeneous bc_values from the hierarchy build; level 0 carries
        # the physical values, so it gets an explicit zero array.
        lvl0 = levels[0]
        self.zero_bc0 = wp.zeros((int(lvl0.ny), int(lvl0.nx)), dtype=wp.float64, device=device)

    def _bc_values(self, level_id: int):
        return self.zero_bc0 if level_id == 0 else self.levels[level_id].bc_values_wp

    # -- per-outer operator refresh ------------------------------------------

    def refresh_faces(self) -> None:
        """Rebuild FP32 faces from the current (FP64) level T/storage values.

        Launch-stable and pointer-stable — safe inside the driver's captured
        per-outer refresh graph."""
        for fl, level in zip(self.f32_levels, self.levels):
            wp.launch(
                kernel=face_build_storage_f32_kernel,
                dim=(fl.ny, fl.nx),
                inputs=[
                    level.T_wp, level.active_wp, level.gh_mask_wp, level.ghb_factor_wp,
                    level.storage_diag_wp,
                    fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag, fl.nx, fl.ny,
                ],
                device=self.device,
            )

    # -- per-outer correction state ------------------------------------------

    def begin_outer(self, residual_wp) -> None:
        """Cast the FP64 nonlinear residual to FP32 and reset the FP32
        correction (called once per Picard outer iteration)."""
        fl0 = self.f32_levels[0]
        wp.launch(
            kernel=_mp_cast_r64_to_r32_kernel,
            dim=(fl0.ny, fl0.nx),
            inputs=[residual_wp, fl0.b, fl0.nx, fl0.ny],
            device=self.device,
        )
        fl0.x.fill_(wp.float32(0.0))

    def sync_correction_in(self, delta_wp) -> None:
        """Re-cast the FP64 correction into FP32 (used after an adaptive
        rollback restores the FP64 correction from its snapshot)."""
        fl0 = self.f32_levels[0]
        wp.launch(
            kernel=_mp_cast_r64_to_r32_kernel,
            dim=(fl0.ny, fl0.nx),
            inputs=[delta_wp, fl0.x, fl0.nx, fl0.ny],
            device=self.device,
        )

    def cast_correction_out(self, delta_wp) -> None:
        """Cast the accumulated FP32 correction back to FP64 (per block)."""
        fl0 = self.f32_levels[0]
        wp.launch(
            kernel=cast_r32_to_r64_kernel,
            dim=(fl0.ny, fl0.nx),
            inputs=[fl0.x, delta_wp, fl0.nx, fl0.ny],
            device=self.device,
        )

    # -- fixed-work FP32 K-cycle ----------------------------------------------

    def _launch_one_kcycle(self, solve_controls: dict) -> None:
        levels = self.levels
        f32_levels = self.f32_levels
        device = self.device

        nu_pre = int(solve_controls.get("nu_pre", 2))
        nu_post = int(solve_controls.get("nu_post", 2))
        nu_coarse = int(solve_controls.get("nu_coarse", 30))
        omega = float(solve_controls.get("omega", 0.8))
        smoother_mode = str(solve_controls.get("smoother", "chebyshev")).strip().lower()
        cheby_lambda_min = float(solve_controls.get("cheby_lambda_min", 0.05))
        cheby_lambda_max = float(solve_controls.get("cheby_lambda_max", 1.95))
        if smoother_mode == "chebyshev":
            pre_omegas = self._chebyshev_relaxation_sequence(nu_pre, cheby_lambda_min, cheby_lambda_max)
            post_omegas = self._chebyshev_relaxation_sequence(nu_post, cheby_lambda_min, cheby_lambda_max)
        else:
            pre_omegas = tuple(omega for _ in range(nu_pre))
            post_omegas = tuple(omega for _ in range(nu_post))
        if len(pre_omegas) == 0:
            pre_omegas = (float(omega),)
        if len(post_omegas) == 0:
            post_omegas = (float(omega),)

        def smooth(fl, level, omegas, level_id) -> None:
            dimL = (fl.ny, fl.nx)
            x_in = fl.x
            x_out = fl.Ax
            bc_vals = self._bc_values(level_id)
            for omega_step in omegas:
                wp.launch(
                    kernel=_mf3_jacobi_f32,
                    dim=dimL,
                    inputs=[fl.b, x_in, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                            level.active_wp, level.bc_mask_wp, bc_vals,
                            float(omega_step), fl.nx, fl.ny, x_out],
                    device=device,
                )
                x_in, x_out = x_out, x_in
            if x_in is not fl.x:
                wp.launch(kernel=copy_field_f32_kernel, dim=dimL,
                          inputs=[x_in, fl.x, fl.nx, fl.ny], device=device)

        def residual(fl, level, x_arr, b_arr, r_arr) -> None:
            wp.launch(
                kernel=_mf3_residual_f32,
                dim=(fl.ny, fl.nx),
                inputs=[x_arr, b_arr, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                        level.active_wp, level.bc_mask_wp, r_arr, fl.nx, fl.ny],
                device=device,
            )

        def kcycle(level_id: int) -> None:
            fl = f32_levels[level_id]
            level = levels[level_id]
            nxL, nyL = fl.nx, fl.ny
            dimL = (nyL, nxL)

            smooth(fl, level, pre_omegas, level_id)
            residual(fl, level, fl.x, fl.b, fl.r)

            if level_id == len(f32_levels) - 1:
                # Coarsest: fixed Jacobi block (approximate inner solve; no
                # classic-PCG trajectory parity required in FP32 mode).
                smooth(fl, level, tuple(float(omega) for _ in range(nu_coarse)), level_id)
                return

            fc = f32_levels[level_id + 1]
            coarse = levels[level_id + 1]
            nxC, nyC = fc.nx, fc.ny
            dimC = (nyC, nxC)

            wp.launch(kernel=restrict_blockavg_f32_kernel, dim=dimC,
                      inputs=[fl.r, level.active_wp, level.bc_mask_wp, fc.b,
                              nxL, nyL, nxC, nyC], device=device)
            fc.x.fill_(wp.float32(0.0))
            kcycle(level_id + 1)

            coarse_is_coarsest = (level_id + 1) == len(f32_levels) - 1
            if coarse_is_coarsest:
                wp.launch(kernel=copy_field_f32_kernel, dim=dimC,
                          inputs=[fc.x, fc.e, nxC, nyC], device=device)
                z1_wp = fc.e
            else:
                wp.launch(kernel=copy_field_f32_kernel, dim=dimC,
                          inputs=[fc.x, fc.z, nxC, nyC], device=device)
                z1_wp = fc.z

            residual(fc, coarse, z1_wp, fc.b, fc.r)
            wp.launch(kernel=copy_field_f32_kernel, dim=dimC,
                      inputs=[fc.r, fc.b, nxC, nyC], device=device)

            fc.x.fill_(wp.float32(0.0))
            kcycle(level_id + 1)

            n_c = nxC * nyC
            wp.launch(kernel=_mf3_dot_partials_f32, dim=n_c,
                      inputs=[fc.b, fc.x, coarse.active_wp, coarse.bc_mask_wp,
                              fc.partials, nxC, nyC, _BLOCK], device=device)
            wp.launch(kernel=combine_partials_kernel, dim=1,
                      inputs=[fc.partials, fc.rho_buf, fc.n_partials], device=device)
            wp.launch(kernel=_mf3_applyA_dot_partials_f32, dim=n_c,
                      inputs=[fc.x, fc.Te, fc.Tw, fc.Tn, fc.Ts, fc.diag,
                              coarse.active_wp, coarse.bc_mask_wp, fc.partials,
                              nxC, nyC, _BLOCK], device=device)
            wp.launch(kernel=combine_partials_kernel, dim=1,
                      inputs=[fc.partials, fc.pAp_buf, fc.n_partials], device=device)
            wp.launch(kernel=self._compute_safe_alpha_kernel, dim=1,
                      inputs=[fc.rho_buf, fc.pAp_buf, fc.alpha_buf], device=device)
            wp.launch(kernel=axpy_active_scalar_2dmask_f32_kernel, dim=dimC,
                      inputs=[z1_wp, fc.x, coarse.active_wp, coarse.bc_mask_wp,
                              fc.alpha_buf, nxC, nyC], device=device)

            wp.launch(kernel=prolong_bilinear_f32_kernel, dim=dimL,
                      inputs=[z1_wp, fl.e, nxL, nyL, nxC, nyC], device=device)
            wp.launch(kernel=add_correction_f32_kernel, dim=dimL,
                      inputs=[fl.x, fl.e, level.active_wp, level.bc_mask_wp,
                              self._bc_values(level_id), nxL, nyL], device=device)

            smooth(fl, level, post_omegas, level_id)

        kcycle(0)

    def solve_block(
        self,
        block_cycles: int,
        solve_controls: dict,
        graph_cache: dict | None = None,
    ) -> None:
        """Run ``block_cycles`` fixed-work FP32 K-cycles on the current
        correction state (delta32 continues in place across blocks).

        With a graph cache (and a CUDA device), exactly one K-cycle is
        captured per structural key and replayed ``block_cycles`` times —
        bit-identical to eager cycles; any capture failure falls back to
        eager launches (same contract as the FP64 face path)."""
        n = int(block_cycles)
        if n <= 0:
            return
        if graph_cache is None or not str(self.device).startswith("cuda"):
            for _ in range(n):
                self._launch_one_kcycle(solve_controls)
            return
        key = (
            "mixed_transient_f32_kcycle_v1",
            id(self.f32_levels),
            tuple((fl.ny, fl.nx) for fl in self.f32_levels),
            int(solve_controls.get("nu_pre", 2)),
            int(solve_controls.get("nu_post", 2)),
            int(solve_controls.get("nu_coarse", 30)),
            str(solve_controls.get("smoother", "chebyshev")).strip().lower(),
            float(solve_controls.get("omega", 0.8)),
            float(solve_controls.get("cheby_lambda_min", 0.05)),
            float(solve_controls.get("cheby_lambda_max", 1.95)),
        )
        entry = graph_cache.get(key)
        if entry is None:
            graph = None
            executed_eagerly = False
            try:
                with wp.ScopedCapture() as cap:
                    self._launch_one_kcycle(solve_controls)
                graph = cap.graph
            except Exception:
                graph = None
            else:
                executed_eagerly = graph is None
            if graph is not None:
                graph_cache[key] = graph
                for _ in range(n):
                    wp.capture_launch(graph)
            else:
                graph_cache[key] = False
                remaining = n - (1 if executed_eagerly else 0)
                for _ in range(max(remaining, 0)):
                    self._launch_one_kcycle(solve_controls)
        elif entry is False:
            for _ in range(n):
                self._launch_one_kcycle(solve_controls)
        else:
            for _ in range(n):
                wp.capture_launch(entry)


__all__ = [
    "MixedTransientInnerSession",
    "face_build_storage_f32_kernel",
    "cast_r32_to_r64_kernel",
]
