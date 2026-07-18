# SPDX-License-Identifier: AGPL-3.0-only
"""Geometric multigrid K-cycle solver backend."""

from __future__ import annotations

from typing import Any

from .base import SolverContext


class ConfinedKCycleBackend:
    """Use the shared K-cycle hierarchy and device work buffers."""

    name = "confined_kcycle"

    def solve(self, context: SolverContext, **kwargs: Any):
        kwargs["unconfined"] = False
        return context.run_kcycle(**kwargs)


def solve_kcycle_device_buffers(
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
    solve_controls,
    return_scalar_info=True,
):
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module
    self = model
    wp = kernel_module.wp
    np = kernel_module.np
    WP_FLOAT = kernel_module.WP_FLOAT
    _chebyshev_relaxation_sequence = kernel_module._chebyshev_relaxation_sequence
    add_correction_kernel = kernel_module.add_correction_kernel
    apply_A_and_pAp_kernel = kernel_module.apply_A_and_pAp_kernel
    apply_A_and_pAp_no_storage_kernel = kernel_module.apply_A_and_pAp_no_storage_kernel
    axpy_active_scalar_2dmask_kernel = kernel_module.axpy_active_scalar_2dmask_kernel
    axpy_active_scalar_kernel = kernel_module.axpy_active_scalar_kernel
    check_rtr_converged_kernel = kernel_module.check_rtr_converged_kernel
    compute_alpha_kernel = kernel_module.compute_alpha_kernel
    compute_beta_and_update_rho_kernel = kernel_module.compute_beta_and_update_rho_kernel
    compute_head_residual_kernel = kernel_module.compute_head_residual_kernel
    compute_head_residual_no_storage_kernel = kernel_module.compute_head_residual_no_storage_kernel
    compute_residual_kernel = kernel_module.compute_residual_kernel
    compute_residual_no_storage_kernel = kernel_module.compute_residual_no_storage_kernel
    compute_safe_alpha_kernel = kernel_module.compute_safe_alpha_kernel
    copy_field_kernel = kernel_module.copy_field_kernel
    dot_active_kernel = kernel_module.dot_active_kernel
    init_pcg_with_A_kernel = kernel_module.init_pcg_with_A_kernel
    init_pcg_with_A_no_storage_kernel = kernel_module.init_pcg_with_A_no_storage_kernel
    jacobi_applyA_fused_kernel = kernel_module.jacobi_applyA_fused_kernel
    jacobi_applyA_fused_no_storage_kernel = kernel_module.jacobi_applyA_fused_no_storage_kernel
    kcycle_check_dh_and_residual_kernel = kernel_module.kcycle_check_dh_and_residual_kernel
    kcycle_check_dh_and_residual_no_storage_kernel = kernel_module.kcycle_check_dh_and_residual_no_storage_kernel
    prolong_bilinear_any_kernel = kernel_module.prolong_bilinear_any_kernel
    reset_kcycle_check_buffers_kernel = kernel_module.reset_kcycle_check_buffers_kernel
    restrict_blockavg_kernel = kernel_module.restrict_blockavg_kernel
    update_p_kernel = kernel_module.update_p_kernel
    update_x_r_z_rho_rTr_kernel = kernel_module.update_x_r_z_rho_rTr_kernel
    zero_scalar_kernel = kernel_module.zero_scalar_kernel
    device = self.device_str

    max_cycles_i = int(solve_controls.get("max_cycles", 20))
    nu_pre = int(solve_controls.get("nu_pre", 2))
    nu_post = int(solve_controls.get("nu_post", 2))
    nu_coarse = int(solve_controls.get("nu_coarse", 30))
    omega = float(solve_controls.get("omega", 0.8))
    rel_tol = float(solve_controls.get("rel_tol", 5.0e-7))
    abs_tol_min = float(solve_controls.get("abs_tol_min", 5.0e-7))

    dh_rms_tol_f = solve_controls.get("dh_rms_tol", 1.0e-4)
    if dh_rms_tol_f is not None: dh_rms_tol_f = float(dh_rms_tol_f)
    dh_max_tol = solve_controls.get("dh_max_tol", None)
    if dh_max_tol is not None: dh_max_tol = float(dh_max_tol)

    smoother_mode = str(solve_controls.get("smoother", "chebyshev")).strip().lower()
    cheby_lambda_min = float(solve_controls.get("cheby_lambda_min", 0.05))
    cheby_lambda_max = float(solve_controls.get("cheby_lambda_max", 1.95))
    coarse_operator_mode = str(
        solve_controls.get("coarse_operator_mode", "stale_approximate_preconditioner")
    )

    if smoother_mode == "chebyshev":
        pre_omegas = _chebyshev_relaxation_sequence(nu_pre, cheby_lambda_min, cheby_lambda_max)
        post_omegas = _chebyshev_relaxation_sequence(nu_post, cheby_lambda_min, cheby_lambda_max)
    else:
        pre_omegas = tuple(omega for _ in range(nu_pre))
        post_omegas = tuple(omega for _ in range(nu_post))
    if len(pre_omegas) == 0: pre_omegas = (float(omega),)
    if len(post_omegas) == 0: post_omegas = (float(omega),)

    lvl0 = levels[0]
    nx0 = int(lvl0.nx)
    ny0 = int(lvl0.ny)
    dim0 = (ny0, nx0)

    # Wire buffers
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

    # We must count free cells
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
        }

    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
    if storage_diag_wp is not None:
        _cr_k = compute_residual_kernel
        _cr_in = [
            lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
            lvl0.gh_mask_wp, lvl0.ghb_factor_wp, storage_diag_wp,
            lvl0.r_wp, lvl0.rTr_buf, nx0, ny0
        ]
    else:
        _cr_k = compute_residual_no_storage_kernel
        _cr_in = [
            lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
            lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
            lvl0.r_wp, lvl0.rTr_buf, nx0, ny0
        ]
    wp.launch(kernel=_cr_k, dim=dim0, inputs=_cr_in, device=device)
    rTr0 = float(lvl0.rTr_buf.numpy()[0])
    gpu_scalar_sync_count += 1
    r_rms0 = float(np.sqrt(max(rTr0, 0.0) / float(n_free0)))
    tol_abs = float(max(abs_tol_min, rel_tol * r_rms0))
    thr_rTr = float((tol_abs * tol_abs) * float(n_free0))

    if rTr0 <= thr_rTr:
        return {"converged": True, "n_cycles_used": 0, "r_rms_end": r_rms0}

    def pcg_solve_level(level, max_iter_level: int):
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rho_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)

        _ipcga_in = [
            level.x_wp, level.b_wp, level.T_wp, level.active_wp, level.bc_mask_wp,
            level.gh_mask_wp, level.ghb_factor_wp,
        ]
        if storage_diag_wp is not None and level is lvl0:
            _ipcga_k = init_pcg_with_A_kernel
            _ipcga_in.append(storage_diag_wp)
        elif getattr(level, "storage_diag_wp", None) is not None:
            _ipcga_k = init_pcg_with_A_kernel
            _ipcga_in.append(level.storage_diag_wp)
        else:
            _ipcga_k = init_pcg_with_A_no_storage_kernel

        _ipcga_in += [
            level.M_inv_wp, level.Ap_wp, level.r_wp, level.z_wp, level.p_wp,
            level.rho_buf, level.rTr_buf, nxL, nyL,
        ]
        wp.launch(kernel=_ipcga_k, dim=dimL, inputs=_ipcga_in, device=device)

        for _ in range(int(max_iter_level)):
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.pAp_buf], device=device)

            _aap_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if storage_diag_wp is not None and level is lvl0:
                _aap_k = apply_A_and_pAp_kernel
                _aap_in.append(storage_diag_wp)
            elif getattr(level, "storage_diag_wp", None) is not None:
                _aap_k = apply_A_and_pAp_kernel
                _aap_in.append(level.storage_diag_wp)
            else:
                _aap_k = apply_A_and_pAp_no_storage_kernel
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

    def kcycle(level_id: int):
        level = levels[level_id]
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)

        x_tmp_wp = level.Ax_wp
        x_in = level.x_wp
        x_out = x_tmp_wp

        for omega_step in pre_omegas:
            _jac_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if storage_diag_wp is not None and level is lvl0:
                _jac_k = jacobi_applyA_fused_kernel
                _jac_in.append(storage_diag_wp)
            elif getattr(level, "storage_diag_wp", None) is not None:
                _jac_k = jacobi_applyA_fused_kernel
                _jac_in.append(level.storage_diag_wp)
            else:
                _jac_k = jacobi_applyA_fused_no_storage_kernel
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
        _cr_in = [
            level.x_wp, level.b_wp, level.T_wp, level.active_wp, level.bc_mask_wp,
            level.gh_mask_wp, level.ghb_factor_wp,
        ]
        if storage_diag_wp is not None and level is lvl0:
            _cr_k = compute_residual_kernel
            _cr_in.append(storage_diag_wp)
        elif getattr(level, "storage_diag_wp", None) is not None:
            _cr_k = compute_residual_kernel
            _cr_in.append(level.storage_diag_wp)
        else:
            _cr_k = compute_residual_no_storage_kernel
        _cr_in += [level.r_wp, level.rTr_buf, nxL, nyL]
        wp.launch(kernel=_cr_k, dim=dimL, inputs=_cr_in, device=device)

        if level_id == (len(levels) - 1):
            pcg_solve_level(level=level, max_iter_level=int(nu_coarse))
            return

        coarse = levels[level_id + 1]
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
        _ccr_in = [
            z1_wp, coarse.b_wp, coarse.T_wp, coarse.active_wp, coarse.bc_mask_wp,
            coarse.gh_mask_wp, coarse.ghb_factor_wp,
        ]
        if getattr(coarse, "storage_diag_wp", None) is not None:
            _ccr_k = compute_residual_kernel
            _ccr_in.append(coarse.storage_diag_wp)
        else:
            _ccr_k = compute_residual_no_storage_kernel
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
        _caap_in = [
            coarse.T_wp, coarse.active_wp, coarse.bc_mask_wp, coarse.gh_mask_wp,
            coarse.ghb_factor_wp,
        ]
        if getattr(coarse, "storage_diag_wp", None) is not None:
            _caap_k = apply_A_and_pAp_kernel
            _caap_in.append(coarse.storage_diag_wp)
        else:
            _caap_k = apply_A_and_pAp_no_storage_kernel
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
            _jac_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if storage_diag_wp is not None and level is lvl0:
                _jac_k = jacobi_applyA_fused_kernel
                _jac_in.append(storage_diag_wp)
            elif getattr(level, "storage_diag_wp", None) is not None:
                _jac_k = jacobi_applyA_fused_kernel
                _jac_in.append(level.storage_diag_wp)
            else:
                _jac_k = jacobi_applyA_fused_no_storage_kernel
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

    n_cycles_used = 0
    converged = False
    check_every = solve_controls.get("check_every_no", 10)

    dh_rms_lastcheck = 0.0
    dh_max_lastcheck = 0.0

    if rTr0 <= float(thr_rTr):
        converged = True
        n_cycles_used = 0
        # Also populate buffers so check is okay
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.dh_max_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rho_buf], device=device)
        return {
            "converged": True,
            "n_cycles_used": 0,
            "r_rms_end": r_rms0,
            "h_rms_end": 0.0,
            "dh_rms_lastcheck": 0.0,
            "dh_max_lastcheck": 0.0,
            "tol_abs": tol_abs,
        }

    if not return_scalar_info:
        for cyc in range(max_cycles_i):
            n_cycles_used = cyc + 1
            kcycle(0)
        return {
            "converged": False,
            "n_cycles_used": int(n_cycles_used),
            "r_rms_end": None,
            "h_rms_end": None,
            "dh_rms_lastcheck": None,
            "dh_max_lastcheck": None,
            "tol_abs": None,
            "gpu_scalar_synchronization_count": 0,
            "coarse_operator_mode": coarse_operator_mode,
            "fine_operator_residual_checked": True,
        }

    for cyc in range(max_cycles_i):
        n_cycles_used = cyc + 1
        kcycle(0)

        if (cyc % int(check_every)) != (int(check_every) - 1):
            continue

        wp.launch(
            kernel=reset_kcycle_check_buffers_kernel,
            dim=1,
            inputs=[lvl0.rho_buf, lvl0.dh_max_buf, lvl0.rTr_buf, lvl0.converged_flag],
            device=device,
        )
        _kc_in = [
            lvl0.x_wp, lvl0.x_prev_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp,
            lvl0.bc_mask_wp, lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
        ]
        if storage_diag_wp is not None:
            _kc_k = kcycle_check_dh_and_residual_kernel
            _kc_in.append(storage_diag_wp)
        else:
            _kc_k = kcycle_check_dh_and_residual_no_storage_kernel
        _kc_in += [lvl0.rho_buf, lvl0.dh_max_buf, lvl0.rTr_buf, int(1 if self.use_ghb else 0), nx0, ny0]
        wp.launch(kernel=_kc_k, dim=dim0, inputs=_kc_in, device=device)
        wp.launch(
            kernel=check_rtr_converged_kernel,
            dim=1,
            inputs=[lvl0.rTr_buf, thr_rTr, lvl0.converged_flag],
# __K_CYCLE_SOURCE_INSERTION_POINT__
