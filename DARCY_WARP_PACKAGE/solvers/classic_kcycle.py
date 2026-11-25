# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared fixed-work recursive classic K-cycle and coarse PCG operations."""

from __future__ import annotations

import numpy as np


def build_classic_kcycle(
    *, levels, device, storage_for_level, pre_omegas, post_omegas, nu_coarse,
    n_free0, recovery_history_every=None,
):
    """Create per-solve K-cycle and PCG callables over preallocated buffers.

    The cycle uses two recursively preconditioned coarse corrections, followed
    by the safe scalar Krylov combination. All operations are capturable when
    the supplied level buffers are stable. The coarse PCG solve uses fixed
    work; only explicit fine-grid recovery performs scalar readbacks.

    :param levels: Ordered multigrid levels, finest first.
    :param device: Warp execution device.
    :param storage_for_level: Callable taking a level and returning a pair
        (storage active, storage diagonal buffer). The caller controls whether
        that level uses transient kernels, even when other levels do not.
    :param pre_omegas: Point relaxation weights before coarse correction.
    :param post_omegas: Point relaxation weights after coarse correction.
    :param nu_coarse: Fixed number of coarsest-grid PCG iterations.
    :param n_free0: Fine-grid free-cell count for recovery RMS diagnostics.
    :param recovery_history_every: Optional iteration interval for recording
        fine-grid recovery history.
    :returns: Recursive cycle callable and level PCG callable.
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.warped_darcy import (
        WP_FLOAT,
        add_correction_kernel,
        apply_A_and_pAp_kernel,
        apply_A_and_pAp_no_storage_kernel,
        axpy_active_scalar_2dmask_kernel,
        axpy_active_scalar_kernel,
        compute_alpha_kernel,
        compute_beta_and_update_rho_kernel,
        compute_residual_kernel,
        compute_residual_no_storage_kernel,
        compute_safe_alpha_kernel,
        copy_field_kernel,
        dot_active_kernel,
        init_pcg_with_A_kernel,
        init_pcg_with_A_no_storage_kernel,
        jacobi_applyA_fused_kernel,
        jacobi_applyA_fused_no_storage_kernel,
        prolong_bilinear_any_kernel,
        restrict_blockavg_kernel,
        update_p_kernel,
        update_x_r_z_rho_rTr_kernel,
        zero_scalar_kernel,
    )

    def pcg_solve_level(level, max_iter_level: int, residual_target_squared=None, recovery_history=None):
        """Apply coarse PCG, or recover a linear fine solve to a target.

        :param level: Multigrid level with writable PCG work buffers.
        :param max_iter_level: Maximum PCG iterations for this invocation.
        :param residual_target_squared: Optional squared flow-residual target.
            None performs fixed work without a device scalar read.
        :param recovery_history: Optional list receiving convergence samples.
        :returns: Number of PCG iterations executed.
        """
        storage_active, storage_diagonal = storage_for_level(level)
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rho_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)

        _ipcga_k = init_pcg_with_A_kernel if storage_active else init_pcg_with_A_no_storage_kernel
        _ipcga_in = [
            level.x_wp, level.b_wp, level.T_wp, level.active_wp, level.bc_mask_wp,
            level.gh_mask_wp, level.ghb_factor_wp,
        ]
        if storage_active:
            _ipcga_in.append(storage_diagonal)
        _ipcga_in += [
            level.M_inv_wp, level.Ap_wp, level.r_wp, level.z_wp, level.p_wp,
            level.rho_buf, level.rTr_buf, nxL, nyL,
        ]
        wp.launch(kernel=_ipcga_k, dim=dimL, inputs=_ipcga_in, device=device)

        iterations = 0
        for iteration in range(int(max_iter_level)):
            iterations = iteration + 1
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.pAp_buf], device=device)
            _aap_k = apply_A_and_pAp_kernel if storage_active else apply_A_and_pAp_no_storage_kernel
            _aap_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if storage_active:
                _aap_in.append(storage_diagonal)
            _aap_in += [level.p_wp, level.Ap_wp, level.pAp_buf, nxL, nyL]
            wp.launch(kernel=_aap_k, dim=dimL, inputs=_aap_in, device=device)

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
                    level.x_wp,
                    level.r_wp,
                    level.z_wp,
                    level.p_wp,
                    level.Ap_wp,
                    level.M_inv_wp,
                    level.active_wp,
                    level.bc_mask_wp,
                    level.alpha_buf,
                    level.rho_new_buf,
                    level.rTr_buf,
                    nxL,
                    nyL,
                ],
                device=device,
            )


            if residual_target_squared is not None:
                residual_squared = float(level.rTr_buf.numpy()[0])
                if recovery_history_every is not None and (
                    iterations % recovery_history_every == 0 or
                    residual_squared <= residual_target_squared or iterations == max_iter_level
                ):
                    recovery_history.append(dict(iter=iterations, rms_res_phys=float(np.sqrt(max(residual_squared, 0.) / n_free0))))
                if not np.isfinite(residual_squared) or residual_squared <= residual_target_squared:
                    break

            wp.launch(
                kernel=compute_beta_and_update_rho_kernel,
                dim=1,
                inputs=[level.rho_buf, level.rho_new_buf, level.beta_buf],
                device=device,
            )

            wp.launch(
                kernel=update_p_kernel,
                dim=dimL,
                inputs=[
                    level.p_wp,
                    level.z_wp,
                    level.active_wp,
                    level.bc_mask_wp,
                    level.beta_buf,
                    nxL,
                    nyL,
                ],
                device=device,
            )
        return iterations

    def kcycle(level_id: int):
        """Execute one recursive two-correction Krylov multigrid visit.

        :param level_id: Index into levels, with zero denoting the fine grid.
        """
        level = levels[level_id]
        storage_active, storage_diagonal = storage_for_level(level)
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)

        x_tmp_wp = level.Ax_wp
        x_in = level.x_wp
        x_out = x_tmp_wp

        for omega_step in pre_omegas:
            _jac_k = jacobi_applyA_fused_kernel if storage_active else jacobi_applyA_fused_no_storage_kernel
            _jac_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if storage_active:
                _jac_in.append(storage_diagonal)
            _jac_in += [
                level.b_wp, x_in, level.M_inv_wp, level.bc_values_wp,
                float(omega_step), nxL, nyL, x_out,
            ]
            wp.launch(kernel=_jac_k, dim=dimL, inputs=_jac_in, device=device)
            tmp = x_in
            x_in = x_out
            x_out = tmp

        if x_in is not level.x_wp:
            wp.launch(kernel=copy_field_kernel, dim=dimL, inputs=[x_in, level.x_wp, nxL, nyL], device=device)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)
        _cr_k = compute_residual_kernel if storage_active else compute_residual_no_storage_kernel
        _cr_in = [
            level.x_wp, level.b_wp, level.T_wp, level.active_wp, level.bc_mask_wp,
            level.gh_mask_wp, level.ghb_factor_wp,
        ]
        if storage_active:
            _cr_in.append(storage_diagonal)
        _cr_in += [level.r_wp, level.rTr_buf, nxL, nyL]
        wp.launch(kernel=_cr_k, dim=dimL, inputs=_cr_in, device=device)

        if level_id == (len(levels) - 1):
            pcg_solve_level(level=level, max_iter_level=int(nu_coarse))
            return

        coarse = levels[level_id + 1]
        coarse_storage_active, coarse_storage_diagonal = storage_for_level(coarse)
        nxC = int(coarse.nx)
        nyC = int(coarse.ny)
        dimC = (nyC, nxC)

        wp.launch(
            kernel=restrict_blockavg_kernel,
            dim=dimC,
            inputs=[level.r_wp, level.active_wp,
                    level.bc_mask_wp,coarse.b_wp,
                    nxL, nyL, nxC, nyC],
            device=device,
        )

        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        coarse_is_coarsest = (level_id + 1) == (len(levels) - 1)
        if coarse_is_coarsest:
            wp.launch(kernel=copy_field_kernel, dim=dimC, inputs=[coarse.x_wp, coarse.e_wp, nxC, nyC],
                      device=device)
            z1_wp = coarse.e_wp
        else:
            wp.launch(kernel=copy_field_kernel, dim=dimC, inputs=[coarse.x_wp, coarse.z_wp, nxC, nyC],
                      device=device)
            z1_wp = coarse.z_wp

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[coarse.rTr_buf], device=device)
        _ccr_k = compute_residual_kernel if coarse_storage_active else compute_residual_no_storage_kernel
        _ccr_in = [
            z1_wp, coarse.b_wp, coarse.T_wp, coarse.active_wp, coarse.bc_mask_wp,
            coarse.gh_mask_wp, coarse.ghb_factor_wp,
        ]
        if coarse_storage_active:
            _ccr_in.append(coarse_storage_diagonal)
        _ccr_in += [coarse.r_wp, coarse.rTr_buf, nxC, nyC]
        wp.launch(kernel=_ccr_k, dim=dimC, inputs=_ccr_in, device=device)

        wp.launch(kernel=copy_field_kernel, dim=dimC, inputs=[coarse.r_wp, coarse.b_wp, nxC, nyC], device=device)
        r1_wp = coarse.b_wp

        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[coarse.rho_buf], device=device)
        wp.launch(
            kernel=dot_active_kernel,
            dim=dimC,
            inputs=[r1_wp, coarse.x_wp, coarse.active_wp, coarse.bc_mask_wp, coarse.rho_buf, nxC, nyC],
            device=device,
        )

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[coarse.pAp_buf], device=device)
        _caap_k = apply_A_and_pAp_kernel if coarse_storage_active else apply_A_and_pAp_no_storage_kernel
        _caap_in = [
            coarse.T_wp, coarse.active_wp, coarse.bc_mask_wp, coarse.gh_mask_wp,
            coarse.ghb_factor_wp,
        ]
        if coarse_storage_active:
            _caap_in.append(coarse_storage_diagonal)
        _caap_in += [coarse.x_wp, coarse.Ax_wp, coarse.pAp_buf, nxC, nyC]
        wp.launch(kernel=_caap_k, dim=dimC, inputs=_caap_in, device=device)

        wp.launch(
            kernel=compute_safe_alpha_kernel,
            dim=1,
            inputs=[coarse.rho_buf, coarse.pAp_buf, coarse.alpha_buf],
            device=device,
        )

        active_is_1d = (len(coarse.active_wp.shape) == 1)
        if active_is_1d:
            wp.launch(
                kernel=axpy_active_scalar_kernel,
                dim=dimC,
                inputs=[z1_wp, coarse.x_wp, coarse.active_wp, coarse.bc_mask_wp, coarse.alpha_buf, nxC, nyC],
                device=device,
            )
        else:
            wp.launch(
                kernel=axpy_active_scalar_2dmask_kernel,
                dim=dimC,
                inputs=[z1_wp, coarse.x_wp, coarse.active_wp, coarse.bc_mask_wp, coarse.alpha_buf, nxC, nyC],
                device=device,
            )

        wp.launch(
            kernel=prolong_bilinear_any_kernel,
            dim=dimL,
            inputs=[z1_wp, level.e_wp, nxL, nyL, nxC, nyC],
            device=device,
        )
        wp.launch(
            kernel=add_correction_kernel,
            dim=dimL,
            inputs=[level.x_wp, level.e_wp, level.active_wp, level.bc_mask_wp, level.bc_values_wp, nxL, nyL],
            device=device,
        )

        x_tmp_wp = level.Ax_wp
        x_in = level.x_wp
        x_out = x_tmp_wp

        for omega_step in post_omegas:
            _jac_k = jacobi_applyA_fused_kernel if storage_active else jacobi_applyA_fused_no_storage_kernel
            _jac_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if storage_active:
                _jac_in.append(storage_diagonal)
            _jac_in += [
                level.b_wp, x_in, level.M_inv_wp, level.bc_values_wp,
                float(omega_step), nxL, nyL, x_out,
            ]
            wp.launch(kernel=_jac_k, dim=dimL, inputs=_jac_in, device=device)
            tmp = x_in
            x_in = x_out
            x_out = tmp

        if x_in is not level.x_wp:
            wp.launch(kernel=copy_field_kernel, dim=dimL, inputs=[x_in, level.x_wp, nxL, nyL], device=device)

    return kcycle, pcg_solve_level
