# SPDX-License-Identifier: AGPL-3.0-only
"""EXPERIMENTAL fixed V-cycle correction for mixed-precision defect correction.

Status: **experimental, opt-in, non-production** (campaign Phase 2; see
``MIXED_PRECISION_CAMPAIGN.md``).  Not part of the solver registry; reachable
by no alias.

Hypothesis under test (H1): a K-cycle is too expensive for use as an
*approximate* correction inside FP64 iterative refinement — its second
recursive descent and Krylov combination double the work of a plain V-cycle.
A fixed FP32 V-cycle (single descent, no inner convergence testing, no scalar
reads) should give better FP64-residual reduction per millisecond.

The driver deliberately reuses the production WP_FLOAT kernels
(smoother/residual/restriction/prolongation) so that Phase 2 isolates the
*cycle structure* question; kernel-level arithmetic precision and
reduction-cost changes belong to Phase 3.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .mixed_precision import (
    EXPERIMENTAL,
    MixedPrecisionDefectCorrectionSession,
)


@wp.kernel
def _alpha_minus_one_kernel(alpha: wp.array(dtype=wp.float64, ndim=1)):
    """alpha <- alpha - 1 (device-side; lets axpy rescale z1 by alpha)."""
    alpha[0] = alpha[0] - wp.float64(1.0)


def solve_vcycle_device_buffers(
    *,
    model: Any,
    x_wp,
    rhs_wp,
    T_wp,
    active_wp,
    bc_mask_wp,
    bc_values_wp,
    levels,
    solve_controls: dict,
) -> None:
    """One fixed-work FP32 V-cycle on the correction equation A x = rhs.

    Single descent, no convergence testing, no device scalar reads.  The
    coarsest level is handled with a fixed block of extra smoothing sweeps
    (no PCG, no dot products, no per-thread-atomic reductions beyond the one
    unused rTr accumulation inside the production residual kernel).

    Buffer ownership mirrors ``solve_kcycle_device_buffers``: level-0 field
    pointers are rewired to the caller's arrays and restored by the caller.
    """
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module

    WP_FLOAT = kernel_module.WP_FLOAT
    _chebyshev_relaxation_sequence = kernel_module._chebyshev_relaxation_sequence
    add_correction_kernel = kernel_module.add_correction_kernel
    apply_A_and_pAp_no_storage_kernel = kernel_module.apply_A_and_pAp_no_storage_kernel
    axpy_active_scalar_2dmask_kernel = kernel_module.axpy_active_scalar_2dmask_kernel
    axpy_active_scalar_kernel = kernel_module.axpy_active_scalar_kernel
    compute_residual_no_storage_kernel = kernel_module.compute_residual_no_storage_kernel
    compute_safe_alpha_kernel = kernel_module.compute_safe_alpha_kernel
    copy_field_kernel = kernel_module.copy_field_kernel
    dot_active_kernel = kernel_module.dot_active_kernel
    jacobi_applyA_fused_no_storage_kernel = kernel_module.jacobi_applyA_fused_no_storage_kernel
    prolong_bilinear_any_kernel = kernel_module.prolong_bilinear_any_kernel
    restrict_blockavg_kernel = kernel_module.restrict_blockavg_kernel
    zero_scalar_kernel = kernel_module.zero_scalar_kernel
    device = model.device_str

    nu_pre = int(solve_controls.get("nu_pre", 2))
    nu_post = int(solve_controls.get("nu_post", 2))
    nu_coarse = int(solve_controls.get("nu_coarse", 30))
    omega = float(solve_controls.get("omega", 0.7))
    per_level_krylov = bool(solve_controls.get("per_level_krylov", False))
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

    def smooth(level, omegas) -> None:
        """Fixed smoothing sweeps, alternating x_wp <-> Ax_wp like production."""
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)
        x_in = level.x_wp
        x_out = level.Ax_wp
        for omega_step in omegas:
            wp.launch(
                kernel=jacobi_applyA_fused_no_storage_kernel,
                dim=dimL,
                inputs=[
                    level.T_wp, level.active_wp, level.bc_mask_wp,
                    level.gh_mask_wp, level.ghb_factor_wp,
                    level.b_wp, x_in, level.M_inv_wp, level.bc_values_wp,
                    float(omega_step), nxL, nyL, x_out,
                ],
                device=device,
            )
            x_in, x_out = x_out, x_in
        if x_in is not level.x_wp:
            wp.launch(
                kernel=copy_field_kernel,
                dim=dimL,
                inputs=[x_in, level.x_wp, nxL, nyL],
                device=device,
            )

    def vcycle(level_id: int) -> None:
        level = levels[level_id]
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)

        smooth(level, pre_omegas)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)
        wp.launch(
            kernel=compute_residual_no_storage_kernel,
            dim=dimL,
            inputs=[
                level.x_wp, level.b_wp, level.T_wp, level.active_wp,
                level.bc_mask_wp, level.gh_mask_wp, level.ghb_factor_wp,
                level.r_wp, level.rTr_buf, nxL, nyL,
            ],
            device=device,
        )

        if level_id == len(levels) - 1:
            # Coarsest: fixed block of extra smoothing sweeps (no PCG).
            smooth(level, tuple(float(omega) for _ in range(nu_coarse)))
            return

        coarse = levels[level_id + 1]
        nxC = int(coarse.nx)
        nyC = int(coarse.ny)
        dimC = (nyC, nxC)

        wp.launch(
            kernel=restrict_blockavg_kernel,
            dim=dimC,
            inputs=[level.r_wp, level.active_wp, level.bc_mask_wp, coarse.b_wp,
                    nxL, nyL, nxC, nyC],
            device=device,
        )
        coarse.x_wp.fill_(WP_FLOAT(0.0))
        vcycle(level_id + 1)

        if per_level_krylov:
            # Energy-optimal rescale of the coarse correction before
            # prolongation: alpha = (b_c . z1) / (z1 . A_c z1), applied as
            # z1 <- z1 + (alpha - 1) * z1 via the existing axpy kernel.
            # This is the K-cycle's overshoot safeguard without its second
            # recursive descent.
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[coarse.rho_buf], device=device)
            wp.launch(
                kernel=dot_active_kernel,
                dim=dimC,
                inputs=[coarse.b_wp, coarse.x_wp, coarse.active_wp,
                        coarse.bc_mask_wp, coarse.rho_buf, nxC, nyC],
                device=device,
            )
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[coarse.pAp_buf], device=device)
            wp.launch(
                kernel=apply_A_and_pAp_no_storage_kernel,
                dim=dimC,
                inputs=[coarse.T_wp, coarse.active_wp, coarse.bc_mask_wp,
                        coarse.gh_mask_wp, coarse.ghb_factor_wp,
                        coarse.x_wp, coarse.Ax_wp, coarse.pAp_buf, nxC, nyC],
                device=device,
            )
            wp.launch(
                kernel=compute_safe_alpha_kernel,
                dim=1,
                inputs=[coarse.rho_buf, coarse.pAp_buf, coarse.alpha_buf],
                device=device,
            )
            # alpha_buf <- alpha - 1 in-place is not available; reuse the
            # production axpy form z1 += alpha * z1 with (alpha - 1) computed
            # on device by a dedicated tiny kernel.
            wp.launch(
                kernel=_alpha_minus_one_kernel,
                dim=1,
                inputs=[coarse.alpha_buf],
                device=device,
            )
            if len(coarse.active_wp.shape) == 1:
                _axpy_k = axpy_active_scalar_kernel
            else:
                _axpy_k = axpy_active_scalar_2dmask_kernel
            wp.launch(
                kernel=_axpy_k,
                dim=dimC,
                inputs=[coarse.x_wp, coarse.x_wp, coarse.active_wp,
                        coarse.bc_mask_wp, coarse.alpha_buf, nxC, nyC],
                device=device,
            )

        wp.launch(
            kernel=prolong_bilinear_any_kernel,
            dim=dimL,
            inputs=[coarse.x_wp, level.e_wp, nxL, nyL, nxC, nyC],
            device=device,
        )
        wp.launch(
            kernel=add_correction_kernel,
            dim=dimL,
            inputs=[level.x_wp, level.e_wp, level.active_wp, level.bc_mask_wp,
                    level.bc_values_wp, nxL, nyL],
            device=device,
        )

        smooth(level, post_omegas)

    lvl0 = levels[0]
    nx0 = int(lvl0.nx)
    ny0 = int(lvl0.ny)

    # Wire level-0 buffers to the caller's correction arrays.
    lvl0.x_wp = x_wp
    lvl0.b_wp = rhs_wp
    lvl0.T_wp = T_wp
    lvl0.storage_diag_wp = None
    lvl0.active_wp = active_wp
    lvl0.bc_mask_wp = bc_mask_wp
    lvl0.bc_values_wp = bc_values_wp

    n_cycles = int(solve_controls.get("max_cycles", 1))
    for _ in range(n_cycles):
        vcycle(0)


class MixedPrecisionVcycleSession(MixedPrecisionDefectCorrectionSession):
    """EXPERIMENTAL mixed-precision session using fixed FP32 V-cycles.

    Identical FP64 outer loop to the K-cycle-based session; only the inner
    correction block differs.  ``inner_kcycles`` in the solve controls selects
    the number of fixed V-cycles per outer refinement step.
    """

    def _inner_correction_block(self, solve_controls: dict) -> None:
        from .mixed_precision import _mp_cast_r64_to_r32_kernel

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
            solve_vcycle_device_buffers(
                model=model,
                x_wp=self.delta32,
                rhs_wp=self.r32,
                T_wp=model.T_wp,
                active_wp=model.active_wp,
                bc_mask_wp=model.bc_mask_wp,
                bc_values_wp=self.zero_bc32,
                levels=model.mg_levels,
                solve_controls=solve_controls,
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


__all__ = [
    "EXPERIMENTAL",
    "MixedPrecisionVcycleSession",
    "solve_vcycle_device_buffers",
]
