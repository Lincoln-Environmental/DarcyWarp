# SPDX-License-Identifier: AGPL-3.0-only
"""Fast face-array operator for the 2D transient unconfined device fast path.

Phase A of ``UNCONFINED_FAST_PLAN.md``: port the steady-confined fast-path
kernel engineering (``solvers/face_kernels_f64.py`` +
``solvers/fast_confined_kcycle.py``) into the production transient unconfined
Picard/K-cycle device driver (``solvers/transient_unconfined.py``).

What this module provides:

* ``face_build_storage_f64_kernel`` — ``face_build_f64_kernel`` plus the
  transient storage diagonal folded into ``diag`` (identical harmonic-mean
  formula and addition order as the classic transient kernels, so the face
  operator is bit-identical to the classic stencil on the same inputs).  The
  face build replaces the fine and per-level ``M_inv`` rebuilds on this
  path (only the coarsest level keeps a classic ``M_inv`` rebuild for its
  PCG sweep).
* ``face_dual_residual_f64_kernel`` — face form of
  ``compute_dual_residual_kernel`` (warped_darcy.py) with per-block partial
  reductions instead of per-thread FP64 atomics.  Same formula: flow
  residual ``b - A h`` and head-equivalent residual ``r / diag`` with storage
  in ``diag``, both summed as rTr over free cells.
* ``face_check_dh_dual_residual_f64_kernel`` — face form of
  ``kcycle_check_dh_and_dual_residual_kernel`` with block-reduced partials
  (x_prev snapshot updated for ALL cells; dh stats + dual rTr on free cells
  only — exact classic semantics).
* ``TransientFaceLevel`` / ``ensure_transient_face_levels`` /
  ``refresh_transient_face_levels`` — per-MG-level face arrays + partial
  buffers, allocated once per (levels identity, shapes) and refreshed IN
  PLACE per Picard outer iteration (pointer-stable for future CUDA-graph
  capture).  Coarse ``T``/``storage_diag`` values still come from the
  existing ``coarsen_transient_operator_level_kernel``.
* ``solve_kcycle_face_transient_device_buffers`` — fixed-work (and optional
  scalar-info) K-cycle on caller-supplied buffers using the face kernels:
  same two-descent + per-level 2-term Krylov structure and smoother
  controls as classic ``solve_kcycle_device_buffers``.  The coarsest level
  keeps the classic PCG sweep (with a per-outer coarsest-only ``M_inv``
  rebuild): at ``nu_coarse=1`` that PCG step is one safe-alpha
  preconditioned Richardson update on the smallest grid, so it costs no
  measurable time but keeps the inner trajectory algorithmically identical
  to the classic path.  (A Jacobi-block coarsest, as used by the confined
  fast path, was tried first: it shifted accepted heads by ~6e-6 m —
  inside the strict acceptance basin, but above the 1e-6 m parity target
  for this port.)  Acceptance gates are unchanged and always evaluated
  with the face dual residual, which matches the classic check formulas
  to round-off.

GHB inputs are part of the face build: ``C_gh`` is included in the diagonal
and the matching ``C_gh * gh_head`` term is included in the transient RHS.
The classic transient device path remains the reference path and does not
support GHB; production GHB runs therefore select the face operator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .face_kernels_f64 import (
    _BLOCK,
    applyA_dot_partials_f64_kernel,
    combine_partials_kernel,
    combine_partials_max_kernel,
    dot_partials_f64_kernel,
    face_check_dh_residual_f64_kernel,
    face_jacobi_f64_kernel,
    face_residual_f64_kernel,
)

# ---------------------------------------------------------------------------
# Kernels (explicit FP64, mirroring solvers/face_kernels_f64.py style)
# ---------------------------------------------------------------------------


@wp.kernel
def face_build_storage_f64_kernel(
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
    """Face conductances + diagonal in FP64 with the transient storage
    diagonal folded in (``diag = sum(faces) + C_gh + storage_diag``).

    Identical harmonic formula and addition order to the classic transient
    kernels (``compute_dual_residual_kernel`` etc.); identity row for
    isolated cells."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        Te[j, i] = wp.float64(0.0)
        Tw[j, i] = wp.float64(0.0)
        Tn[j, i] = wp.float64(0.0)
        Ts[j, i] = wp.float64(0.0)
        diag[j, i] = wp.float64(1.0)
        return

    tiny = wp.float64(1.0e-12)
    T_c = wp.float64(T_field[j, i])

    t_e = wp.float64(0.0)
    t_w = wp.float64(0.0)
    t_n = wp.float64(0.0)
    t_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = t_e + t_w + t_n + t_s + C_gh + wp.float64(storage_diag[j, i])
    if sum_T < tiny:
        Te[j, i] = wp.float64(0.0)
        Tw[j, i] = wp.float64(0.0)
        Tn[j, i] = wp.float64(0.0)
        Ts[j, i] = wp.float64(0.0)
        diag[j, i] = wp.float64(1.0)
    else:
        Te[j, i] = t_e
        Tw[j, i] = t_w
        Tn[j, i] = t_n
        Ts[j, i] = t_s
        diag[j, i] = sum_T


@wp.kernel
def face_dual_residual_f64_kernel(
    x: wp.array(ndim=2),
    b: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    partials_flow: wp.array(dtype=wp.float64, ndim=1),
    partials_head: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """Dual residual, face-array form.  Mirrors
    ``compute_dual_residual_kernel`` semantics exactly (flow residual
    ``b - A h`` and head-equivalent residual ``r / diag``, both summed as rTr
    over free cells only) with per-block partial reductions instead of
    per-thread atomics.  ``diag`` already carries the storage term; the
    face build maps tiny-diagonal cells to the identity row (diag = 1), which
    reproduces the classic ``diagA < tiny`` branch."""
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span

    flow_acc = wp.float64(0.0)
    head_acc = wp.float64(0.0)
    if active[j, i] != 0 and bc_mask[j, i] == 0:
        hC = wp.float64(x[j, i])
        ax = wp.float64(diag[j, i]) * hC
        t_e = wp.float64(Te[j, i])
        t_w = wp.float64(Tw[j, i])
        t_n = wp.float64(Tn[j, i])
        t_s = wp.float64(Ts[j, i])
        if t_e > wp.float64(0.0):
            ax = ax - t_e * wp.float64(x[j, i + 1])
        if t_w > wp.float64(0.0):
            ax = ax - t_w * wp.float64(x[j, i - 1])
        if t_n > wp.float64(0.0):
            ax = ax - t_n * wp.float64(x[j - 1, i])
        if t_s > wp.float64(0.0):
            ax = ax - t_s * wp.float64(x[j + 1, i])
        flow_residual = wp.float64(b[j, i]) - ax
        head_residual = flow_residual / wp.float64(diag[j, i])
        flow_acc = flow_residual * flow_residual
        head_acc = head_residual * head_residual
    wp.atomic_add(partials_flow, block, flow_acc)
    wp.atomic_add(partials_head, block, head_acc)


@wp.kernel
def face_check_dh_dual_residual_f64_kernel(
    x: wp.array(ndim=2),
    x_prev: wp.array(ndim=2),
    b: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    partials_dh_sq: wp.array(dtype=wp.float64, ndim=1),
    partials_dh_max: wp.array(dtype=wp.float64, ndim=1),
    partials_flow: wp.array(dtype=wp.float64, ndim=1),
    partials_head: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """Picard outer convergence check, face-array form.  Mirrors
    ``kcycle_check_dh_and_dual_residual_kernel`` semantics exactly (x_prev
    snapshot updated for ALL cells; dh stats and dual residuals on free
    cells only) with per-block partial reductions instead of per-thread
    atomics."""
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span

    x_new = wp.float64(x[j, i])
    x_old = wp.float64(x_prev[j, i])
    x_prev[j, i] = x[j, i]

    sq = wp.float64(0.0)
    mx = wp.float64(0.0)
    fr = wp.float64(0.0)
    hr = wp.float64(0.0)
    if active[j, i] != 0 and bc_mask[j, i] == 0:
        dh = x_new - x_old
        sq = dh * dh
        mx = wp.abs(dh)

        hC = x_new
        ax = wp.float64(diag[j, i]) * hC
        t_e = wp.float64(Te[j, i])
        t_w = wp.float64(Tw[j, i])
        t_n = wp.float64(Tn[j, i])
        t_s = wp.float64(Ts[j, i])
        if t_e > wp.float64(0.0):
            ax = ax - t_e * wp.float64(x[j, i + 1])
        if t_w > wp.float64(0.0):
            ax = ax - t_w * wp.float64(x[j, i - 1])
        if t_n > wp.float64(0.0):
            ax = ax - t_n * wp.float64(x[j - 1, i])
        if t_s > wp.float64(0.0):
            ax = ax - t_s * wp.float64(x[j + 1, i])
        flow_residual = wp.float64(b[j, i]) - ax
        head_residual = flow_residual / wp.float64(diag[j, i])
        fr = flow_residual * flow_residual
        hr = head_residual * head_residual
    wp.atomic_add(partials_dh_sq, block, sq)
    wp.atomic_max(partials_dh_max, block, mx)
    wp.atomic_add(partials_flow, block, fr)
    wp.atomic_add(partials_head, block, hr)




@wp.kernel
def build_transient_rhs_ghb_f64_kernel(
    recharge_rate: wp.array(ndim=2),
    T_field: wp.array(ndim=2),
    storage_diag: wp.array(ndim=2),
    head_prev: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_head: wp.array(ndim=2),
    ghb_factor: wp.array(ndim=2),
    dx: wp.float64,
    nx: int,
    ny: int,
    rhs_out: wp.array(ndim=2),
):
    """Transient RHS with the GHB injection term, face-path variant of
    ``build_transient_rhs_from_storage_kernel`` (warped_darcy.py).

    ``rhs = R*dx^2 + storage_diag*h_prev + C_gh*gh_head`` with
    ``C_gh = T_c*ghb_factor`` — the identical formula and guard semantics as
    the host path (``build_rhs_kernel`` / ``build_rhs_fd_like`` at
    head_scale=1.0, plus the backward-Euler storage term from
    ``_prepare_5point_transient_terms``).  ``C_gh`` uses the CURRENT outer
    iteration's T(h), exactly like the host path, which rebuilds its RHS from
    the freshly updated ``T_wp`` every Picard iteration.  The matching diag
    term ``C_gh`` is folded into the operator by
    ``face_build_storage_f64_kernel``; both are rebuilt together every
    outer iteration, so diag and RHS stay consistent."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0:
        rhs_out[j, i] = wp.float64(0.0)
        return
    if bc_mask[j, i] != 0:
        rhs_out[j, i] = wp.float64(bc_values[j, i])
        return

    rhs = (
        wp.float64(recharge_rate[j, i]) * dx * dx
        + wp.float64(storage_diag[j, i]) * wp.float64(head_prev[j, i])
    )
    if gh_mask[j, i] != 0:
        T_c = wp.float64(T_field[j, i])
        ghbf = wp.float64(ghb_factor[j, i])
        if T_c > wp.float64(0.0) and ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            rhs = rhs + T_c * ghbf * wp.float64(gh_head[j, i])
    rhs_out[j, i] = rhs


# ---------------------------------------------------------------------------
# Face-level storage and refresh
# ---------------------------------------------------------------------------


class TransientFaceLevel:
    """Face-conductance arrays + reduction workspace for one hierarchy level
    (transient variant: ``diag`` carries the storage diagonal)."""

    __slots__ = ("nx", "ny", "Te", "Tw", "Tn", "Ts", "diag",
                 "partials", "partials_b", "partials_c", "partials_d",
                 "out", "n_partials")

    def __init__(self, level, device: str):
        self.nx = int(level.nx)
        self.ny = int(level.ny)
        shape = (self.ny, self.nx)
        self.Te = wp.zeros(shape, dtype=wp.float64, device=device)
        self.Tw = wp.zeros(shape, dtype=wp.float64, device=device)
        self.Tn = wp.zeros(shape, dtype=wp.float64, device=device)
        self.Ts = wp.zeros(shape, dtype=wp.float64, device=device)
        self.diag = wp.zeros(shape, dtype=wp.float64, device=device)
        n_cells = self.nx * self.ny
        self.n_partials = (n_cells + _BLOCK - 1) // _BLOCK
        self.partials = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.partials_b = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.partials_c = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.partials_d = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.out = wp.zeros(1, dtype=wp.float64, device=device)


def _launch_face_build(fl: TransientFaceLevel, level, device: str) -> None:
    wp.launch(
        kernel=face_build_storage_f64_kernel,
        dim=(fl.ny, fl.nx),
        inputs=[
            level.T_wp, level.active_wp, level.gh_mask_wp, level.ghb_factor_wp,
            level.storage_diag_wp,
            fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag, fl.nx, fl.ny,
        ],
        device=device,
    )


def ensure_transient_face_levels(model: Any, levels) -> list[TransientFaceLevel]:
    """Build (or reuse cached) transient face-conductance levels.

    Cache validity mirrors ``ensure_fast_face_levels``: keyed on the levels
    list identity, the level-0 object identity, and per-level shapes.  Value
    staleness is NOT tracked here — the transient driver refreshes face
    values explicitly every Picard outer iteration via
    ``refresh_transient_face_levels`` (in place, pointer-stable).
    """
    cache = getattr(model, "_transient_face_cache", None)
    if (
        cache is not None
        and cache["levels_id"] == id(levels)
        and cache["level0"] is levels[0]
        and len(cache["faces"]) == len(levels)
        and all(
            (fl.ny, fl.nx) == (int(level.ny), int(level.nx))
            for fl, level in zip(cache["faces"], levels)
        )
    ):
        return cache["faces"]
    device = str(model.device_str)
    faces = [TransientFaceLevel(level, device) for level in levels]
    model._transient_face_cache = {
        "levels_id": id(levels),
        "level0": levels[0],
        "faces": faces,
    }
    return faces


def refresh_transient_face_levels(
    model: Any,
    levels,
    face_levels: list[TransientFaceLevel],
    *,
    level0_only: bool = False,
) -> None:
    """Recompute face conductances in place from the current level values.

    Level 0 is built from ``levels[0].T_wp`` + ``levels[0].storage_diag_wp``
    (the driver's fine transmissivity and secant-storage diagonal).  Coarse
    levels are first refreshed with the existing
    ``coarsen_transient_operator_level_kernel`` (same dynamic coarse
    operator as the classic path) and then face-built from the coarsened
    values.  This replaces the fine and per-level ``M_inv`` rebuilds on the
    device transient fast path; the coarsest level additionally keeps a
    classic ``M_inv`` rebuild (one small launch) for its classic PCG sweep.
    """
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module

    coarsen_transient_operator_level_kernel = kernel_module.coarsen_transient_operator_level_kernel
    build_diag_preconditioner_kernel = kernel_module.build_diag_preconditioner_kernel
    device = str(model.device_str)

    _launch_face_build(face_levels[0], levels[0], device)
    if not level0_only:
        for lid in range(1, len(levels)):
            fine = levels[lid - 1]
            coarse = levels[lid]
            if getattr(fine, "storage_diag_wp", None) is None or getattr(coarse, "storage_diag_wp", None) is None:
                raise RuntimeError("transient device hierarchy is missing storage diagonal buffers")
            wp.launch(
                kernel=coarsen_transient_operator_level_kernel,
                dim=(int(coarse.ny), int(coarse.nx)),
                inputs=[
                    fine.T_wp,
                    fine.storage_diag_wp,
                    fine.active_wp,
                    coarse.active_wp,
                    coarse.bc_mask_wp,
                    int(fine.nx),
                    int(fine.ny),
                    int(coarse.nx),
                    int(coarse.ny),
                    coarse.T_wp,
                    coarse.storage_diag_wp,
                ],
                device=device,
            )
            _launch_face_build(face_levels[lid], coarse, device)
        # Coarsest-level classic M_inv for the classic PCG sweep (one small
        # launch; the coarsest grid is tiny by construction).
        coarsest = levels[-1]
        wp.launch(
            kernel=build_diag_preconditioner_kernel,
            dim=(int(coarsest.ny), int(coarsest.nx)),
            inputs=[
                coarsest.T_wp,
                coarsest.active_wp,
                coarsest.bc_mask_wp,
                coarsest.gh_mask_wp,
                coarsest.ghb_factor_wp,
                coarsest.storage_diag_wp,
                int(1 if bool(model.use_ghb) else 0),
                int(coarsest.nx),
                int(coarsest.ny),
                coarsest.M_inv_wp,
            ],
            device=device,
        )


# ---------------------------------------------------------------------------
# Fast K-cycle on caller-supplied device buffers (face operator)
# ---------------------------------------------------------------------------


def solve_kcycle_face_transient_device_buffers(
    *,
    model: Any,
    x_wp,
    rhs_wp,
    T_wp,
    storage_diag_wp,
    active_wp,
    bc_mask_wp,
    bc_values_wp,
    levels,
    face_levels: list[TransientFaceLevel],
    solve_controls: dict,
    return_scalar_info: bool = False,
    graph_cache: dict | None = None,
) -> dict:
    """K-cycle on the face operator, device-buffer variant for the transient
    unconfined driver.

    Same two-descent + per-level 2-term Krylov structure, smoother controls
    (Chebyshev/Jacobi with the same defaults), and fixed-work semantics as
    classic ``solve_kcycle_device_buffers`` — but with face-array smoothers
    and residuals, two-stage block reductions, and a Jacobi-block coarsest
    level (replacing PCG).  Level-0 field pointers are rewired to the
    caller's arrays exactly like the classic device-buffer path (callers
    re-wire on the next call; nothing is restored).

    With ``return_scalar_info=False`` (production inner blocks) the solve is
    fixed-work with zero device scalar reads.  With
    ``return_scalar_info=True`` (the startup confined pre-solve) it mirrors
    the classic initial-residual-relative convergence test, check cadence,
    and dh safeguards, and returns the classic info-dict keys.

    Phase B (``UNCONFINED_FAST_PLAN.md``): when ``graph_cache`` is a dict
    (and the device is CUDA), the fixed-work path captures exactly ONE
    K-cycle per key — (level-0 buffer wiring identity, level shapes, nu/
    smoother/omega structure) — and replays it ``max_cycles`` times.  The
    adaptive inner controller's variable block size therefore needs no
    block-size keying: a block of N cycles is N replays of the same graph,
    bit-identical to N eager cycles (same kernels, same order, same
    reduction order).  Any capture failure is recorded in the cache
    (``False`` sentinel) and falls back to eager launches with identical
    semantics.  Scalar-info mode captures only the fixed K-cycle sequence;
    convergence checks and all host readbacks remain outside the graph.
    """
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module

    self = model
    WP_FLOAT = kernel_module.WP_FLOAT
    _chebyshev_relaxation_sequence = kernel_module._chebyshev_relaxation_sequence
    add_correction_kernel = kernel_module.add_correction_kernel
    axpy_active_scalar_2dmask_kernel = kernel_module.axpy_active_scalar_2dmask_kernel
    axpy_active_scalar_kernel = kernel_module.axpy_active_scalar_kernel
    check_rtr_converged_kernel = kernel_module.check_rtr_converged_kernel
    compute_safe_alpha_kernel = kernel_module.compute_safe_alpha_kernel
    copy_field_kernel = kernel_module.copy_field_kernel
    prolong_bilinear_any_kernel = kernel_module.prolong_bilinear_any_kernel
    restrict_blockavg_kernel = kernel_module.restrict_blockavg_kernel
    zero_scalar_kernel = kernel_module.zero_scalar_kernel
    # Classic kernels kept for the coarsest-level PCG sweep (trajectory
    # parity with the classic device-buffer K-cycle).
    init_pcg_with_A_kernel = kernel_module.init_pcg_with_A_kernel
    apply_A_and_pAp_kernel = kernel_module.apply_A_and_pAp_kernel
    compute_alpha_kernel = kernel_module.compute_alpha_kernel
    update_x_r_z_rho_rTr_kernel = kernel_module.update_x_r_z_rho_rTr_kernel
    compute_beta_and_update_rho_kernel = kernel_module.compute_beta_and_update_rho_kernel
    update_p_kernel = kernel_module.update_p_kernel
    device = self.device_str

    # Same control keys/defaults as classic solve_kcycle_device_buffers.
    max_cycles_i = int(solve_controls.get("max_cycles", 20))
    nu_pre = int(solve_controls.get("nu_pre", 2))
    nu_post = int(solve_controls.get("nu_post", 2))
    nu_coarse = int(solve_controls.get("nu_coarse", 30))
    omega = float(solve_controls.get("omega", 0.8))
    rel_tol = float(solve_controls.get("rel_tol", 5.0e-7))
    abs_tol_min = float(solve_controls.get("abs_tol_min", 5.0e-7))

    dh_rms_tol_f = solve_controls.get("dh_rms_tol", 1.0e-4)
    if dh_rms_tol_f is not None:
        dh_rms_tol_f = float(dh_rms_tol_f)
    dh_max_tol = solve_controls.get("dh_max_tol", None)
    if dh_max_tol is not None:
        dh_max_tol = float(dh_max_tol)

    smoother_mode = str(solve_controls.get("smoother", "chebyshev")).strip().lower()
    cheby_lambda_min = float(solve_controls.get("cheby_lambda_min", 0.05))
    cheby_lambda_max = float(solve_controls.get("cheby_lambda_max", 1.95))
    coarse_operator_mode = str(
        solve_controls.get("coarse_operator_mode", "device_refreshed_dynamic_coarse_operator")
    )

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

    lvl0 = levels[0]
    nx0 = int(lvl0.nx)
    ny0 = int(lvl0.ny)
    dim0 = (ny0, nx0)

    # Wire buffers (same as the classic device-buffer path).
    lvl0.x_wp = x_wp
    lvl0.b_wp = rhs_wp
    lvl0.T_wp = T_wp
    lvl0.storage_diag_wp = storage_diag_wp
    lvl0.active_wp = active_wp
    lvl0.bc_mask_wp = bc_mask_wp
    lvl0.bc_values_wp = bc_values_wp

    wp.launch(
        kernel=copy_field_kernel,
        dim=dim0,
        inputs=[lvl0.x_wp, lvl0.x_prev_wp, nx0, ny0],
        device=device,
    )

    for k in range(1, len(levels)):
        levels[k].x_wp.fill_(WP_FLOAT(0.0))
        levels[k].b_wp.fill_(WP_FLOAT(0.0))
        levels[k].r_wp.fill_(WP_FLOAT(0.0))
        levels[k].Ax_wp.fill_(WP_FLOAT(0.0))
        levels[k].e_wp.fill_(WP_FLOAT(0.0))
        levels[k].z_wp.fill_(WP_FLOAT(0.0))
        levels[k].p_wp.fill_(WP_FLOAT(0.0))
        levels[k].Ap_wp.fill_(WP_FLOAT(0.0))
        levels[k].rTr_buf.fill_(0.0)
        levels[k].rho_buf.fill_(0.0)
        levels[k].rho_new_buf.fill_(0.0)
        levels[k].pAp_buf.fill_(0.0)
        levels[k].alpha_buf.fill_(0.0)
        levels[k].beta_buf.fill_(0.0)
        levels[k].converged_flag.fill_(0)
        if getattr(levels[k], "dh_max_buf", None) is not None:
            levels[k].dh_max_buf.fill_(0.0)
        if getattr(levels[k], "x_prev_wp", None) is not None:
            levels[k].x_prev_wp.fill_(WP_FLOAT(0.0))

    gpu_scalar_sync_count = 0

    n_free0 = int(np.count_nonzero((self.active_host != 0) & (self.bc_mask_host == 0)))
    if n_free0 <= 0:
        return {
            "converged": True,
            "n_cycles_used": 0,
            "r_rms_end": 0.0,
            "h_rms_end": 0.0,
            "gpu_scalar_synchronization_count": 0,
            "coarse_operator_mode": coarse_operator_mode,
            "fine_operator_residual_checked": True,
            "implementation": "face_f64",
        }

    f0 = face_levels[0]
    n0 = nx0 * ny0

    def face_residual_into(fl, x_arr, b_arr, r_arr, level) -> None:
        wp.launch(
            kernel=face_residual_f64_kernel,
            dim=(fl.ny, fl.nx),
            inputs=[x_arr, b_arr, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                    level.active_wp, level.bc_mask_wp, r_arr, fl.nx, fl.ny],
            device=device,
        )

    def smooth(fl, level, omegas) -> None:
        nxL, nyL = fl.nx, fl.ny
        dimL = (nyL, nxL)
        x_in = level.x_wp
        x_out = level.Ax_wp
        for omega_step in omegas:
            wp.launch(
                kernel=face_jacobi_f64_kernel,
                dim=dimL,
                inputs=[level.b_wp, x_in, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                        level.active_wp, level.bc_mask_wp, level.bc_values_wp,
                        float(omega_step), nxL, nyL, x_out],
                device=device,
            )
            x_in, x_out = x_out, x_in
        if x_in is not level.x_wp:
            wp.launch(kernel=copy_field_kernel, dim=dimL,
                      inputs=[x_in, level.x_wp, nxL, nyL], device=device)

    def pcg_solve_level_classic(level, max_iter_level: int) -> None:
        """Coarsest-level PCG sweep, identical to the classic device-buffer
        K-cycle (same kernels, same M_inv-based preconditioner, same
        storage-aware operator) so the inner trajectory matches classic."""
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rho_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)

        wp.launch(
            kernel=init_pcg_with_A_kernel,
            dim=dimL,
            inputs=[
                level.x_wp, level.b_wp, level.T_wp, level.active_wp, level.bc_mask_wp,
                level.gh_mask_wp, level.ghb_factor_wp, level.storage_diag_wp,
                level.M_inv_wp, level.Ap_wp, level.r_wp, level.z_wp, level.p_wp,
                level.rho_buf, level.rTr_buf, nxL, nyL,
            ],
            device=device,
        )

        for _ in range(int(max_iter_level)):
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.pAp_buf], device=device)
            wp.launch(
                kernel=apply_A_and_pAp_kernel,
                dim=dimL,
                inputs=[
                    level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                    level.ghb_factor_wp, level.storage_diag_wp,
                    level.p_wp, level.Ap_wp, level.pAp_buf, nxL, nyL,
                ],
                device=device,
            )
            wp.launch(
                kernel=compute_alpha_kernel,
                dim=1,
                inputs=[level.rho_buf, level.pAp_buf, level.alpha_buf],
                device=device,
            )
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rho_new_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)
            wp.launch(
                kernel=update_x_r_z_rho_rTr_kernel,
                dim=dimL,
                inputs=[
                    level.x_wp, level.r_wp, level.z_wp, level.p_wp, level.Ap_wp,
                    level.M_inv_wp, level.active_wp, level.bc_mask_wp,
                    level.alpha_buf, level.rho_new_buf, level.rTr_buf, nxL, nyL,
                ],
                device=device,
            )
            wp.launch(
                kernel=compute_beta_and_update_rho_kernel,
                dim=1,
                inputs=[level.rho_buf, level.rho_new_buf, level.beta_buf],
                device=device,
            )
            wp.launch(
                kernel=update_p_kernel,
                dim=dimL,
                inputs=[level.p_wp, level.z_wp, level.active_wp, level.bc_mask_wp,
                        level.beta_buf, nxL, nyL],
                device=device,
            )

    def kcycle(level_id: int) -> None:
        fl = face_levels[level_id]
        level = levels[level_id]
        nxL, nyL = fl.nx, fl.ny
        dimL = (nyL, nxL)

        smooth(fl, level, pre_omegas)
        face_residual_into(fl, level.x_wp, level.b_wp, level.r_wp, level)

        if level_id == len(levels) - 1:
            # Coarsest: classic PCG sweep (trajectory parity with classic).
            pcg_solve_level_classic(level, int(nu_coarse))
            return

        fc = face_levels[level_id + 1]
        coarse = levels[level_id + 1]
        nxC, nyC = fc.nx, fc.ny
        dimC = (nyC, nxC)

        wp.launch(kernel=restrict_blockavg_kernel, dim=dimC,
                  inputs=[level.r_wp, level.active_wp, level.bc_mask_wp, coarse.b_wp,
                          nxL, nyL, nxC, nyC],
                  device=device)
        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        coarse_is_coarsest = (level_id + 1) == len(levels) - 1
        if coarse_is_coarsest:
            wp.launch(kernel=copy_field_kernel, dim=dimC,
                      inputs=[coarse.x_wp, coarse.e_wp, nxC, nyC], device=device)
            z1_wp = coarse.e_wp
        else:
            wp.launch(kernel=copy_field_kernel, dim=dimC,
                      inputs=[coarse.x_wp, coarse.z_wp, nxC, nyC], device=device)
            z1_wp = coarse.z_wp

        # r1 = b - A z1, then second descent on r1
        face_residual_into(fc, z1_wp, coarse.b_wp, coarse.r_wp, coarse)
        wp.launch(kernel=copy_field_kernel, dim=dimC,
                  inputs=[coarse.r_wp, coarse.b_wp, nxC, nyC], device=device)

        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        # alpha = (r1 . z2) / (z2 . A z2); z1 += alpha * z2
        n_c = nxC * nyC
        wp.launch(kernel=dot_partials_f64_kernel, dim=n_c,
                  inputs=[coarse.b_wp, coarse.x_wp, coarse.active_wp,
                          coarse.bc_mask_wp, fc.partials, nxC, nyC, _BLOCK],
                  device=device)
        wp.launch(kernel=combine_partials_kernel, dim=1,
                  inputs=[fc.partials, coarse.rho_buf, fc.n_partials], device=device)
        wp.launch(kernel=applyA_dot_partials_f64_kernel, dim=n_c,
                  inputs=[coarse.x_wp, fc.Te, fc.Tw, fc.Tn, fc.Ts, fc.diag,
                          coarse.active_wp, coarse.bc_mask_wp, fc.partials,
                          nxC, nyC, _BLOCK],
                  device=device)
        wp.launch(kernel=combine_partials_kernel, dim=1,
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

        smooth(fl, level, post_omegas)

    if not return_scalar_info:
        # Fixed-work preconditioner mode: exactly ``max_cycles_i`` K-cycles,
        # no convergence testing, no device scalar reads.
        if graph_cache is None or not str(device).startswith("cuda"):
            for _ in range(max_cycles_i):
                kcycle(0)
        else:
            graph_key = (
                "face_transient_kcycle_v1",
                id(x_wp),
                id(rhs_wp),
                id(bc_values_wp),
                id(levels),
                int(len(levels)),
                tuple((int(l.ny), int(l.nx)) for l in levels),
                int(nu_pre),
                int(nu_post),
                int(nu_coarse),
                str(smoother_mode),
                tuple(float(v) for v in pre_omegas),
                tuple(float(v) for v in post_omegas),
            )
            entry = graph_cache.get(graph_key)
            if entry is None:
                # First block with this key: capture one K-cycle (capture
                # does not execute), then replay max_cycles_i times.  On
                # any capture failure run the full block eagerly instead.
                graph = None
                executed_eagerly = False
                try:
                    with wp.ScopedCapture() as cap:
                        kcycle(0)
                    graph = cap.graph
                except Exception:
                    graph = None
                else:
                    # Null capture (e.g. profiler): launches already ran
                    # eagerly inside the capture context (same contract as
                    # mixed_fast._inner_correction_block).
                    executed_eagerly = graph is None
                if graph is not None:
                    graph_cache[graph_key] = graph
                    for _ in range(max_cycles_i):
                        wp.capture_launch(graph)
                else:
                    graph_cache[graph_key] = False
                    remaining = max_cycles_i - (1 if executed_eagerly else 0)
                    for _ in range(max(int(remaining), 0)):
                        kcycle(0)
            elif entry is False:
                # Capture previously failed for this key: eager fallback.
                for _ in range(max_cycles_i):
                    kcycle(0)
            else:
                for _ in range(max_cycles_i):
                    wp.capture_launch(entry)
        return {
            "converged": False,
            "n_cycles_used": int(max_cycles_i),
            "r_rms_end": None,
            "h_rms_end": None,
            "dh_rms_lastcheck": None,
            "dh_max_lastcheck": None,
            "tol_abs": None,
            "gpu_scalar_synchronization_count": 0,
            "coarse_operator_mode": coarse_operator_mode,
            "fine_operator_residual_checked": True,
            "implementation": "face_f64",
        }

    # Scalar-info mode (startup confined pre-solve): classic convergence
    # semantics with face-kernel checks.
    face_residual_into(f0, lvl0.x_wp, lvl0.b_wp, lvl0.r_wp, lvl0)
    wp.launch(kernel=dot_partials_f64_kernel, dim=n0,
              inputs=[lvl0.r_wp, lvl0.r_wp, lvl0.active_wp, lvl0.bc_mask_wp,
                      f0.partials, nx0, ny0, _BLOCK],
              device=device)
    wp.launch(kernel=combine_partials_kernel, dim=1,
              inputs=[f0.partials, lvl0.rTr_buf, f0.n_partials], device=device)
    rTr0 = float(lvl0.rTr_buf.numpy()[0])
    gpu_scalar_sync_count += 1
    r_rms0 = float(np.sqrt(max(rTr0, 0.0) / float(n_free0)))
    tol_abs = float(max(abs_tol_min, rel_tol * r_rms0))
    thr_rTr = float((tol_abs * tol_abs) * float(n_free0))

    n_cycles_used = 0
    converged = False
    check_every = int(solve_controls.get("check_every_no", 10))
    dh_rms_lastcheck = 0.0
    dh_max_lastcheck = 0.0

    if rTr0 <= thr_rTr:
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.dh_max_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rho_buf], device=device)
        return {
            "converged": True,
            "n_cycles_used": 0,
            "r_rms_end": float(r_rms0),
            "h_rms_end": 0.0,
            "dh_rms_lastcheck": 0.0,
            "dh_max_lastcheck": 0.0,
            "tol_abs": float(tol_abs),
            "gpu_scalar_synchronization_count": int(gpu_scalar_sync_count),
            "coarse_operator_mode": coarse_operator_mode,
            "fine_operator_residual_checked": True,
            "implementation": "face_f64",
        }

    # Scalar convergence decisions stay on the host, but the fixed K-cycle
    # launch sequence can still be captured and replayed between decisions.
    # The first replay is explicit after successful capture; a null capture is
    # treated as one already-executed eager cycle.  This keeps exact-once
    # behavior for capture, fallback, and graph replay.  Capture is deliberately
    # deferred until after the initial convergence short-circuit above.
    graph_entry = None
    graph_null_executed = False
    graph_key = None
    if graph_cache is not None and str(device).startswith("cuda"):
        graph_key = (
            "face_transient_kcycle_scalar_v1",
            id(lvl0.x_wp),
            id(lvl0.b_wp),
            id(lvl0.bc_values_wp),
            id(levels),
            int(len(levels)),
            tuple((int(l.ny), int(l.nx)) for l in levels),
            int(nu_pre),
            int(nu_post),
            int(nu_coarse),
            str(smoother_mode),
            tuple(float(v) for v in pre_omegas),
            tuple(float(v) for v in post_omegas),
        )
        cached_graph = graph_cache.get(graph_key)
        if cached_graph is not None and cached_graph is not False:
            graph_entry = cached_graph
        elif cached_graph is False:
            graph_null_executed = False
        else:
            try:
                with wp.ScopedCapture() as capture:
                    kcycle(0)
            except Exception:
                graph_cache[graph_key] = False
            else:
                if capture.graph is None:
                    graph_null_executed = True
                    graph_cache[graph_key] = False
                else:
                    graph_entry = capture.graph
                    graph_cache[graph_key] = graph_entry

    def run_one_cycle(cycle_index: int) -> None:
        if graph_entry is not None:
            wp.capture_launch(graph_entry)
        elif graph_null_executed and int(cycle_index) == 0:
            return
        else:
            kcycle(0)

    for cyc in range(max_cycles_i):
        n_cycles_used = cyc + 1
        run_one_cycle(cyc)

        if (cyc % check_every) != (check_every - 1):
            continue

        wp.launch(
            kernel=face_check_dh_residual_f64_kernel,
            dim=n0,
            inputs=[lvl0.x_wp, lvl0.x_prev_wp, lvl0.b_wp,
                    f0.Te, f0.Tw, f0.Tn, f0.Ts, f0.diag,
                    lvl0.active_wp, lvl0.bc_mask_wp,
                    f0.partials, f0.partials_b, f0.partials_c,
                    nx0, ny0, _BLOCK],
            device=device,
        )
        wp.launch(kernel=combine_partials_kernel, dim=1,
                  inputs=[f0.partials, lvl0.rho_buf, f0.n_partials], device=device)
        wp.launch(kernel=combine_partials_max_kernel, dim=1,
                  inputs=[f0.partials_b, lvl0.dh_max_buf, f0.n_partials], device=device)
        wp.launch(kernel=combine_partials_kernel, dim=1,
                  inputs=[f0.partials_c, lvl0.rTr_buf, f0.n_partials], device=device)
        wp.launch(kernel=check_rtr_converged_kernel, dim=1,
                  inputs=[lvl0.rTr_buf, wp.float64(thr_rTr), lvl0.converged_flag],
                  device=device)

        dh2 = float(lvl0.rho_buf.numpy()[0])
        gpu_scalar_sync_count += 1
        dh_rms_lastcheck = float(np.sqrt(max(dh2, 0.0) / float(n_free0)))
        dh_max_lastcheck = float(lvl0.dh_max_buf.numpy()[0])
        gpu_scalar_sync_count += 1

        dh_ok = True
        if dh_max_tol is not None and dh_rms_tol_f is not None:
            dh_ok = dh_max_lastcheck <= float(dh_max_tol) and dh_rms_lastcheck <= float(dh_rms_tol_f)

        res_ok = int(lvl0.converged_flag.numpy()[0]) != 0
        gpu_scalar_sync_count += 1

        if res_ok and dh_ok:
            converged = True
            break

    # Final flow residual RMS (face residual + block reduction).
    face_residual_into(f0, lvl0.x_wp, lvl0.b_wp, lvl0.r_wp, lvl0)
    wp.launch(kernel=dot_partials_f64_kernel, dim=n0,
              inputs=[lvl0.r_wp, lvl0.r_wp, lvl0.active_wp, lvl0.bc_mask_wp,
                      f0.partials, nx0, ny0, _BLOCK],
              device=device)
    wp.launch(kernel=combine_partials_kernel, dim=1,
              inputs=[f0.partials, lvl0.rTr_buf, f0.n_partials], device=device)
    rTr_end = float(lvl0.rTr_buf.numpy()[0])
    gpu_scalar_sync_count += 1
    r_rms_end = float(np.sqrt(max(rTr_end, 0.0) / float(n_free0)))

    # Final head-equivalent residual RMS (face dual residual, head partial).
    wp.launch(
        kernel=face_dual_residual_f64_kernel,
        dim=n0,
        inputs=[lvl0.x_wp, lvl0.b_wp,
                f0.Te, f0.Tw, f0.Tn, f0.Ts, f0.diag,
                lvl0.active_wp, lvl0.bc_mask_wp,
                f0.partials, f0.partials_b,
                nx0, ny0, _BLOCK],
        device=device,
    )
    wp.launch(kernel=combine_partials_kernel, dim=1,
              inputs=[f0.partials, f0.out, f0.n_partials], device=device)
    wp.launch(kernel=combine_partials_kernel, dim=1,
              inputs=[f0.partials_b, lvl0.rTr_buf, f0.n_partials], device=device)
    hrTr_end = float(lvl0.rTr_buf.numpy()[0])
    gpu_scalar_sync_count += 1
    h_rms_end = float(np.sqrt(max(hrTr_end, 0.0) / float(n_free0)))

    return {
        "converged": bool(converged),
        "n_cycles_used": int(n_cycles_used),
        "r_rms_end": float(r_rms_end),
        "h_rms_end": float(h_rms_end),
        "dh_rms_lastcheck": float(dh_rms_lastcheck),
        "dh_max_lastcheck": float(dh_max_lastcheck),
        "tol_abs": float(tol_abs),
        "gpu_scalar_synchronization_count": int(gpu_scalar_sync_count),
        "coarse_operator_mode": coarse_operator_mode,
        "fine_operator_residual_checked": True,
        "implementation": "face_f64",
    }


__all__ = [
    "TransientFaceLevel",
    "ensure_transient_face_levels",
    "refresh_transient_face_levels",
    "solve_kcycle_face_transient_device_buffers",
    "face_build_storage_f64_kernel",
    "face_dual_residual_f64_kernel",
    "face_check_dh_dual_residual_f64_kernel",
    "build_transient_rhs_ghb_f64_kernel",
]
