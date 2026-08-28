# SPDX-License-Identifier: AGPL-3.0-only
"""Production mixed-precision fast K-cycle correction driver.

This is the production mixed-precision path used by the confined steady
benchmark runner. It remains an explicit precision choice because the model
must be created under ``DARCY_FLOAT=float32``; it is therefore not represented
as a separate numerical backend in the solver registry.

Structure is deliberately identical to the production fixed-work K-cycle
(two descents + per-level 2-term Krylov; Phase 2 showed that structure is
load-bearing on this approximate hierarchy).  What changes is kernel-level
cost:

* face-conductance arrays built once per level (no per-call FP64 divisions);
* true-FP32 (or true-FP64, for the like-for-like baseline) stencil
  arithmetic via ``mixed_fast_kernels``;
* no per-thread-atomic residual norms (unused in fixed-work cycles);
* two-stage (per-block partial + combine) reductions for the two scalars
  the Krylov combination consumes on device;
* coarsest level handled with a fixed Jacobi block instead of PCG (the
  production PCG coarse solve is ~10 % of cycle GPU time on a 125-cell
  grid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from .mixed_precision import MixedPrecisionDefectCorrectionSession
from . import mixed_fast_kernels as mf3

_BLOCK = 256
EXPERIMENTAL = False


@dataclass(frozen=True)
class MixedFastConfig:
    """Validated production settings for the fast mixed-precision solver.

    Defaults are the campaign-validated configuration
    (``MIXED_PRECISION_CAMPAIGN.md``): 5 fast K-cycles per outer refinement,
    Chebyshev smoothing 2/2, 10 Jacobi sweeps at the coarsest level,
    FP64-outer tolerances matching the production gates.
    """

    inner_kcycles: int = 5
    max_outer: int = 40
    nu_pre: int = 2
    nu_post: int = 2
    nu_coarse: int = 10
    omega: float = 0.7
    smoother: str = "chebyshev"
    cheby_lambda_min: float = 0.1
    cheby_lambda_max: float = 2.0
    rel_tol: float = 5.0e-7
    abs_tol_min: float = 5.0e-7
    dh_rms_tol: float = 1.0e-4
    dh_max_tol: float | None = None
    max_levels: int = 6
    min_coarse_cells: int = 500


def get_mixed_fast_session(
    model: Any,
    *,
    bc_values_f64: np.ndarray,
    gh_head_f64: np.ndarray | None,
    R_f64: np.ndarray,
    config: MixedFastConfig = MixedFastConfig(),
) -> "MixedPrecisionFastSession":
    """Explicit opt-in session factory (cached on the model per config).

    The model must be built under ``DARCY_FLOAT=float32``.  Reusing the
    session keeps FP64 master buffers, face arrays, and the captured
    correction graph alive across solves (ensemble/benchmark use).
    """
    cache = getattr(model, "_mixed_fast_sessions", None)
    if cache is None:
        cache = {}
        model._mixed_fast_sessions = cache
    key = (id(bc_values_f64), id(gh_head_f64), id(R_f64))
    entry = cache.get(key)
    if entry is None:
        session = MixedPrecisionFastSession(
            model,
            bc_values_f64=bc_values_f64,
            gh_head_f64=gh_head_f64,
            R_f64=R_f64,
            max_levels=int(config.max_levels),
            min_coarse_cells=int(config.min_coarse_cells),
        )
        # Hold strong references to the keying arrays so their ids cannot be
        # recycled into a false cache hit.
        cache[key] = (bc_values_f64, gh_head_f64, R_f64, session)
        return session
    return entry[3]


def solve_mixed_fast(
    model: Any,
    initial_head_f64: np.ndarray,
    *,
    bc_values_f64: np.ndarray,
    gh_head_f64: np.ndarray | None,
    R_f64: np.ndarray,
    config: MixedFastConfig = MixedFastConfig(),
):
    """Run one production mixed-precision solve from ``initial_head_f64``.

    Every timed invocation starts from the caller-supplied head (benchmarks:
    the original DEM); all defect-correction iterations and K-cycles are
    inside this single call.
    """
    session = get_mixed_fast_session(
        model,
        bc_values_f64=bc_values_f64,
        gh_head_f64=gh_head_f64,
        R_f64=R_f64,
        config=config,
    )
    return session.solve(
        initial_head_f64,
        inner_kcycles=int(config.inner_kcycles),
        max_outer=int(config.max_outer),
        nu_pre=int(config.nu_pre),
        nu_post=int(config.nu_post),
        nu_coarse=int(config.nu_coarse),
        omega=float(config.omega),
        smoother=str(config.smoother),
        cheby_lambda_min=float(config.cheby_lambda_min),
        cheby_lambda_max=float(config.cheby_lambda_max),
        rel_tol=float(config.rel_tol),
        abs_tol_min=float(config.abs_tol_min),
        dh_rms_tol=float(config.dh_rms_tol),
        dh_max_tol=config.dh_max_tol,
    )


@dataclass
class FastLevel:
    """Face-conductance arrays + reduction workspace for one hierarchy level."""

    nx: int
    ny: int
    Te: Any
    Tw: Any
    Tn: Any
    Ts: Any
    diag: Any
    partials: Any  # fp64 per-block partial buffer
    out: Any  # fp64 combined scalar
    n_partials: int
    level: Any  # underlying production hierarchy level (buffers/masks)


def _fill_face_level(faces: dict, level: Any, wp_dtype, device: str) -> None:
    """(Re)fill face-conductance arrays from a hierarchy level's current state."""
    nxL, nyL = int(level.nx), int(level.ny)
    build_kernel = mf3._mf3_build_faces_f32 if wp_dtype is wp.float32 else mf3._mf3_build_faces_f64
    wp.launch(
        kernel=build_kernel,
        dim=(nyL, nxL),
        inputs=[
            level.T_wp, level.active_wp, level.gh_mask_wp, level.ghb_factor_wp,
            faces["Te"], faces["Tw"], faces["Tn"], faces["Ts"], faces["diag"],
            nxL, nyL,
        ],
        device=device,
    )


def build_face_level(model: Any, level: Any, wp_dtype, device: str) -> FastLevel:
    """Allocate and fill face-conductance arrays for one hierarchy level."""
    nxL, nyL = int(level.nx), int(level.ny)
    shape = (nyL, nxL)
    faces = {k: wp.zeros(shape, dtype=wp_dtype, device=device)
             for k in ("Te", "Tw", "Tn", "Ts", "diag")}
    _fill_face_level(faces, level, wp_dtype, device)
    n_cells = nxL * nyL
    n_partials = (n_cells + _BLOCK - 1) // _BLOCK
    return FastLevel(
        nx=nxL, ny=nyL,
        Te=faces["Te"], Tw=faces["Tw"], Tn=faces["Tn"], Ts=faces["Ts"],
        diag=faces["diag"],
        partials=wp.zeros(n_partials, dtype=wp.float64, device=device),
        out=wp.zeros(1, dtype=wp.float64, device=device),
        n_partials=n_partials,
        level=level,
    )


def solve_kcycle_fast_device_buffers(
    *,
    model: Any,
    x_wp,
    rhs_wp,
    T_wp,
    active_wp,
    bc_mask_wp,
    bc_values_wp,
    levels,
    fast_levels: list[FastLevel],
    solve_controls: dict,
) -> None:
    """Fixed-work K-cycle on the correction equation, fast-kernel variant.

    Same two-descent + per-level Krylov structure as production
    ``solve_kcycle_device_buffers`` with ``fixed_work_no_scalar_reads=True``
    (no convergence testing, no device scalar reads).  Buffer ownership:
    level-0 field pointers are rewired to the caller's arrays and restored
    by the caller, exactly like the production device-buffer path.
    """
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module

    WP_FLOAT = kernel_module.WP_FLOAT
    _chebyshev_relaxation_sequence = kernel_module._chebyshev_relaxation_sequence
    add_correction_kernel = kernel_module.add_correction_kernel
    axpy_active_scalar_2dmask_kernel = kernel_module.axpy_active_scalar_2dmask_kernel
    axpy_active_scalar_kernel = kernel_module.axpy_active_scalar_kernel
    compute_safe_alpha_kernel = kernel_module.compute_safe_alpha_kernel
    copy_field_kernel = kernel_module.copy_field_kernel
    prolong_bilinear_any_kernel = kernel_module.prolong_bilinear_any_kernel
    restrict_blockavg_kernel = kernel_module.restrict_blockavg_kernel
    device = model.device_str

    if WP_FLOAT is wp.float32:
        jacobi_k = mf3._mf3_jacobi_f32
        residual_k = mf3._mf3_residual_f32
        dot_k = mf3._mf3_dot_partials_f32
        applyA_dot_k = mf3._mf3_applyA_dot_partials_f32
    else:
        jacobi_k = mf3._mf3_jacobi_f64
        residual_k = mf3._mf3_residual_f64
        dot_k = mf3._mf3_dot_partials_f64
        applyA_dot_k = mf3._mf3_applyA_dot_partials_f64

    nu_pre = int(solve_controls.get("nu_pre", 2))
    nu_post = int(solve_controls.get("nu_post", 2))
    nu_coarse = int(solve_controls.get("nu_coarse", 30))
    omega = float(solve_controls.get("omega", 0.7))
    smoother_mode = str(solve_controls.get("smoother", "chebyshev")).strip().lower()
    cheby_lambda_min = float(solve_controls.get("cheby_lambda_min", 0.1))
    cheby_lambda_max = float(solve_controls.get("cheby_lambda_max", 2.0))

    if smoother_mode == "chebyshev":
        pre_omegas = _chebyshev_relaxation_sequence(nu_pre, cheby_lambda_min, cheby_lambda_max)
        post_omegas = _chebyshev_relaxation_sequence(nu_post, cheby_lambda_min, cheby_lambda_max)
    else:
        pre_omegas = tuple(omega for _ in range(nu_pre))
        post_omegas = tuple(omega for _ in range(nu_post))
    if len(pre_omegas) == 0:
        pre_omegas = (float(omega),)
    if len(post_omegas) == 0:
        post_omegas = (float(omega),)

    def smooth(fl: FastLevel, omegas) -> None:
        level = fl.level
        dimL = (fl.ny, fl.nx)
        x_in = level.x_wp
        x_out = level.Ax_wp
        for omega_step in omegas:
            wp.launch(
                kernel=jacobi_k,
                dim=dimL,
                inputs=[
                    level.b_wp, x_in, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                    level.active_wp, level.bc_mask_wp, level.bc_values_wp,
                    float(omega_step), fl.nx, fl.ny, x_out,
                ],
                device=device,
            )
            x_in, x_out = x_out, x_in
        if x_in is not level.x_wp:
            wp.launch(kernel=copy_field_kernel, dim=dimL,
                      inputs=[x_in, level.x_wp, fl.nx, fl.ny], device=device)

    def residual(fl: FastLevel, x_arr, b_arr, r_arr) -> None:
        level = fl.level
        wp.launch(
            kernel=residual_k,
            dim=(fl.ny, fl.nx),
            inputs=[
                x_arr, b_arr, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                level.active_wp, level.bc_mask_wp, r_arr, fl.nx, fl.ny,
            ],
            device=device,
        )

    def kcycle(level_id: int) -> None:
        fl = fast_levels[level_id]
        level = fl.level
        nxL, nyL = fl.nx, fl.ny
        dimL = (nyL, nxL)

        smooth(fl, pre_omegas)
        residual(fl, level.x_wp, level.b_wp, level.r_wp)

        if level_id == len(fast_levels) - 1:
            # Coarsest: fixed Jacobi block (no PCG, no reductions).
            smooth(fl, tuple(float(omega) for _ in range(nu_coarse)))
            return

        fc = fast_levels[level_id + 1]
        coarse = fc.level
        nxC, nyC = fc.nx, fc.ny
        dimC = (nyC, nxC)

        wp.launch(
            kernel=restrict_blockavg_kernel,
            dim=dimC,
            inputs=[level.r_wp, level.active_wp, level.bc_mask_wp, coarse.b_wp,
                    nxL, nyL, nxC, nyC],
            device=device,
        )
        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        coarse_is_coarsest = (level_id + 1) == len(fast_levels) - 1
        if coarse_is_coarsest:
            wp.launch(kernel=copy_field_kernel, dim=dimC,
                      inputs=[coarse.x_wp, coarse.e_wp, nxC, nyC], device=device)
            z1_wp = coarse.e_wp
        else:
            wp.launch(kernel=copy_field_kernel, dim=dimC,
                      inputs=[coarse.x_wp, coarse.z_wp, nxC, nyC], device=device)
            z1_wp = coarse.z_wp

        # r1 = b - A z1, then second descent on r1
        residual(fc, z1_wp, coarse.b_wp, coarse.r_wp)
        wp.launch(kernel=copy_field_kernel, dim=dimC,
                  inputs=[coarse.r_wp, coarse.b_wp, nxC, nyC], device=device)

        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        # alpha = (r1 . z2) / (z2 . A z2); z1 += alpha * z2
        n_c = nxC * nyC
        wp.launch(kernel=dot_k, dim=n_c,
                  inputs=[coarse.b_wp, coarse.x_wp, coarse.active_wp,
                          coarse.bc_mask_wp, fc.partials, nxC, nyC, _BLOCK],
                  device=device)
        wp.launch(kernel=mf3._mf3_combine_partials_kernel, dim=1,
                  inputs=[fc.partials, coarse.rho_buf, fc.n_partials], device=device)
        wp.launch(kernel=applyA_dot_k, dim=n_c,
                  inputs=[coarse.x_wp, fc.Te, fc.Tw, fc.Tn, fc.Ts, fc.diag,
                          coarse.active_wp, coarse.bc_mask_wp, fc.partials,
                          nxC, nyC, _BLOCK],
                  device=device)
        wp.launch(kernel=mf3._mf3_combine_partials_kernel, dim=1,
                  inputs=[fc.partials, coarse.pAp_buf, fc.n_partials], device=device)
        wp.launch(kernel=compute_safe_alpha_kernel, dim=1,
                  inputs=[coarse.rho_buf, coarse.pAp_buf, coarse.alpha_buf],
                  device=device)

        if len(coarse.active_wp.shape) == 1:
            _axpy_k = axpy_active_scalar_kernel
        else:
            _axpy_k = axpy_active_scalar_2dmask_kernel
        wp.launch(kernel=_axpy_k, dim=dimC,
                  inputs=[z1_wp, coarse.x_wp, coarse.active_wp, coarse.bc_mask_wp,
                          coarse.alpha_buf, nxC, nyC],
                  device=device)

        wp.launch(kernel=prolong_bilinear_any_kernel, dim=dimL,
                  inputs=[z1_wp, level.e_wp, nxL, nyL, nxC, nyC], device=device)
        wp.launch(kernel=add_correction_kernel, dim=dimL,
                  inputs=[level.x_wp, level.e_wp, level.active_wp, level.bc_mask_wp,
                          level.bc_values_wp, nxL, nyL],
                  device=device)

        smooth(fl, post_omegas)

    lvl0 = levels[0]
    lvl0.x_wp = x_wp
    lvl0.b_wp = rhs_wp
    lvl0.T_wp = T_wp
    lvl0.storage_diag_wp = None
    lvl0.active_wp = active_wp
    lvl0.bc_mask_wp = bc_mask_wp
    lvl0.bc_values_wp = bc_values_wp

    for _ in range(int(solve_controls.get("max_cycles", 1))):
        kcycle(0)


class MixedPrecisionFastSession(MixedPrecisionDefectCorrectionSession):
    """Production mixed-precision session with fast FP32 K-cycle correction
    and block-reduced FP64 outer kernels."""

    def __init__(self, model: Any, **kwargs: Any):
        super().__init__(model, emit_experimental_warning=False, **kwargs)
        device = self.device

        # FP32 face arrays for every hierarchy level (correction operator)
        self.fast_levels = [
            build_face_level(model, level, wp.float32, device)
            for level in model.mg_levels
        ]
        # FP64 face arrays for level 0 only (authoritative outer residual);
        # built from the model's coefficient storage with the identical
        # harmonic formula, so rows are bit-identical to the un-optimized
        # FP64 residual kernel.
        lvl0 = model.mg_levels[0]
        self.face0_f64 = build_face_level(model, lvl0, wp.float64, device)

        n0 = self.nx * self.ny
        self.n_partials0 = (n0 + _BLOCK - 1) // _BLOCK
        self.partials0 = wp.zeros(self.n_partials0, dtype=wp.float64, device=device)
        self.partials0_max = wp.zeros(self.n_partials0, dtype=wp.float64, device=device)
        self.out0 = wp.zeros(1, dtype=wp.float64, device=device)

        # CUDA-graph cache for the fixed correction block (Phase 4): keyed
        # by the solve-control values that affect the launch sequence.
        self._correction_graph = None
        self._correction_graph_key = None

    def solve(self, initial_head_f64: np.ndarray, **controls: Any):
        """Solve with production mixed precision and label the result accordingly."""
        head, info = super().solve(initial_head_f64, **controls)
        production_info = dict(info)
        production_info["experimental"] = False
        production_info["production_precision_path"] = True
        return head, production_info

    def refresh_operator_faces(self) -> None:
        """Refill face-conductance arrays from the model's current hierarchy.

        Call after an in-place transmissivity update
        (``model.update_T_in_place``) and before the next :meth:`solve`.
        Arrays are refilled in place — pointers are unchanged, so the
        captured correction graph remains valid and replays against the new
        operator.  Pair with :meth:`update_rhs_f64` (the FP64 RHS carries a
        T-dependent GHB term).
        """
        for fl in self.fast_levels:
            _fill_face_level(
                {"Te": fl.Te, "Tw": fl.Tw, "Tn": fl.Tn, "Ts": fl.Ts, "diag": fl.diag},
                fl.level,
                wp.float32,
                self.device,
            )
        f0 = self.face0_f64
        _fill_face_level(
            {"Te": f0.Te, "Tw": f0.Tw, "Tn": f0.Tn, "Ts": f0.Ts, "diag": f0.diag},
            f0.level,
            wp.float64,
            self.device,
        )

    # -- graph-captured correction block (Phase 4) -----------------------------

    @staticmethod
    def _correction_controls_key(solve_controls: dict) -> tuple:
        return tuple(sorted(
            (k, v) for k, v in solve_controls.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        ))

    def _launch_correction_block(self, solve_controls: dict) -> None:
        """The full fixed correction sequence: cast defect to FP32, zero the
        correction, run the configured fast K-cycles.  Launch-stable: no
        host reads, no allocation, no shape changes — safe to capture."""
        from .mixed_precision import _mp_cast_r64_to_r32_kernel

        model = self.model
        wp.launch(
            kernel=_mp_cast_r64_to_r32_kernel,
            dim=(self.ny, self.nx),
            inputs=[self.r64, self.r32, self.nx, self.ny],
            device=self.device,
        )
        self.delta32.fill_(wp.float32(0.0))

        solve_kcycle_fast_device_buffers(
            model=model,
            x_wp=self.delta32,
            rhs_wp=self.r32,
            T_wp=model.T_wp,
            active_wp=model.active_wp,
            bc_mask_wp=model.bc_mask_wp,
            bc_values_wp=self.zero_bc32,
            levels=model.mg_levels,
            fast_levels=self.fast_levels,
            solve_controls=solve_controls,
        )

    def _inner_correction_block(self, solve_controls: dict) -> None:
        key = self._correction_controls_key(solve_controls)
        lvl0 = self.model.mg_levels[0]
        front = (
            lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.storage_diag_wp,
            lvl0.active_wp, lvl0.bc_mask_wp, lvl0.bc_values_wp,
        )
        try:
            if self._correction_graph is not None and self._correction_graph_key == key:
                wp.capture_launch(self._correction_graph)
                return
            # Capture once (launches do not execute during capture), restore
            # the level-0 wiring, then replay so this outer step still does
            # its work.
            with wp.ScopedCapture() as cap:
                self._launch_correction_block(solve_controls)
            self._correction_graph = cap.graph
            self._correction_graph_key = key
        finally:
            (
                lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.storage_diag_wp,
                lvl0.active_wp, lvl0.bc_mask_wp, lvl0.bc_values_wp,
            ) = front
        if self._correction_graph is None:
            # Capture disabled (e.g. launch-profiler null capture): the block
            # already executed eagerly inside the capture context.
            return
        wp.capture_launch(self._correction_graph)

    def _true_residual(self) -> float:
        f0 = self.face0_f64
        n0 = self.nx * self.ny
        wp.launch(
            kernel=mf3._mf3_outer_residual_f64,
            dim=n0,
            inputs=[
                self.h64, self.b64, f0.Te, f0.Tw, f0.Tn, f0.Ts, f0.diag,
                self.model.active_wp, self.model.bc_mask_wp, self.r64,
                self.partials0, self.nx, self.ny, _BLOCK,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=mf3._mf3_combine_partials_kernel,
            dim=1,
            inputs=[self.partials0, self.rTr_buf, self.n_partials0],
            device=self.device,
        )
        return float(self.rTr_buf.numpy()[0])

    def _accumulate(self) -> tuple[float, float]:
        n0 = self.nx * self.ny
        wp.launch(
            kernel=mf3._mf3_accumulate_f64,
            dim=n0,
            inputs=[
                self.h64, self.delta32, self.model.active_wp,
                self.model.bc_mask_wp, self.bc_values64,
                self.partials0, self.partials0_max, self.nx, self.ny, _BLOCK,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=mf3._mf3_combine_partials_kernel,
            dim=1,
            inputs=[self.partials0, self.dh_sq_buf, self.n_partials0],
            device=self.device,
        )
        wp.launch(
            kernel=mf3._mf3_combine_partials_max_kernel,
            dim=1,
            inputs=[self.partials0_max, self.dh_max_buf, self.n_partials0],
            device=self.device,
        )
        dh_max = float(self.dh_max_buf.numpy()[0])
        dh_rms = float(np.sqrt(max(float(self.dh_sq_buf.numpy()[0]), 0.0) / float(self.n_free)))
        return dh_max, dh_rms

    # -- FP64 outer pieces (block-reduced) ------------------------------------


__all__ = [
    "EXPERIMENTAL",
    "FastLevel",
    "MixedFastConfig",
    "MixedPrecisionFastSession",
    "build_face_level",
    "get_mixed_fast_session",
    "solve_kcycle_fast_device_buffers",
    "solve_mixed_fast",
]
