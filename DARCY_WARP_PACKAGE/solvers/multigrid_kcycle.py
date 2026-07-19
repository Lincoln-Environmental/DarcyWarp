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
        return solve_multigrid_kcycle_backend(model=context.model, **kwargs)


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
    fixed_work_no_scalar_reads=False,
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

    if return_scalar_info or not fixed_work_no_scalar_reads:
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
    else:
        # Fixed-work preconditioner mode: deliberately avoid device scalar
        # reads and convergence decisions.  The caller requested exactly
        # ``max_cycles_i`` K-cycles.
        rTr0 = float("inf")
        r_rms0 = float("inf")
        tol_abs = float("nan")
        thr_rTr = float("nan")

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
            device=device,
        )

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

    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
    if storage_diag_wp is not None:
        _cr_k = compute_residual_kernel
        _cr_in = [
            lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
            lvl0.gh_mask_wp, lvl0.ghb_factor_wp, storage_diag_wp,
            lvl0.r_wp, lvl0.rTr_buf, nx0, ny0,
        ]
    else:
        _cr_k = compute_residual_no_storage_kernel
        _cr_in = [
            lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
            lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
            lvl0.r_wp, lvl0.rTr_buf, nx0, ny0,
        ]
    wp.launch(kernel=_cr_k, dim=dim0, inputs=_cr_in, device=device)
    rTr_end = float(lvl0.rTr_buf.numpy()[0])
    gpu_scalar_sync_count += 1
    r_rms_end = float(np.sqrt(max(rTr_end, 0.0) / float(n_free0)))

    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
    if storage_diag_wp is not None:
        _hr_k = compute_head_residual_kernel
        _hr_in = [
            lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
            lvl0.gh_mask_wp, lvl0.ghb_factor_wp, storage_diag_wp,
            lvl0.r_wp, lvl0.rTr_buf, nx0, ny0,
        ]
    else:
        _hr_k = compute_head_residual_no_storage_kernel
        _hr_in = [
            lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
            lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
            lvl0.r_wp, lvl0.rTr_buf, nx0, ny0,
        ]
    wp.launch(kernel=_hr_k, dim=dim0, inputs=_hr_in, device=device)
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
    }

def solve_multigrid_kcycle_backend(
        model: Any,
        max_cycles: int = 20,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 30,
        omega: float = 0.8,
        rel_tol: float = 5.0e-7,
        abs_tol_min: float = 5.0e-7,
        initial_head: np.ndarray | None = None,
        aq_thickness: float | np.ndarray | None = None,
        gh_alpha: float | np.ndarray | None = None,
        max_levels: int = 5,
        return_info: bool = True,
        check_every_no: int = 10,
        dh_rms_tol: float | None  = 1.0e-4,
        dh_max_tol: float | None = None,
        dh_max_factor: float = 5.0,
        min_coarse_cells: int | None = 500,
        fallback_to_pcg: bool = True,
        divergence_cycle_start: int = 100,
        divergence_residual_factor: float = 3.0,
        fallback_pcg_max_iter: int | None = None,
        fallback_pcg_history_every: int | None = None,
        smoother: str = "chebyshev",
        cheby_lambda_min: float = 0.05,
        cheby_lambda_max: float = 1.95,
        unconfined: bool = False,
        K_field: np.ndarray | None = None,
        zbot_field: np.ndarray | None = None,
        ztop_field: np.ndarray | None = None,
        max_outer_iterations: int | None = None,
        omega_min: float = 0.05,
        omega_max: float = 0.75,
        chebyshev_enabled: bool = True,
        chebyshev_order: int = 3,
        chebyshev_lambda_min_fraction: float = 0.1,
        chebyshev_reset_on_residual_increase: bool = True,
        chebyshev_rejection_factor: float = 1.2,
        min_saturated_thickness: float | None = None,
        initial_saturated_thickness: float = 10.0,
        max_head_change_per_outer_iteration: float = 5.0,
        hclose: float | None = None,
        dry_cell_flag_threshold: float = 0.1,
        unconfined_min_sat: float | None = None,
        unconfined_max_picard_iter: int | None = None,
        unconfined_relax: float | None = None,
        unconfined_head_tol: float | None = None,
        residual_floor_tol: float | None = 1.0e-4,
        inner_head_residual_tol: float | None = None,
        unconfined_inner_max_cycles_early: int = 10,
        unconfined_inner_max_cycles_middle: int = 25,
        unconfined_inner_max_cycles_late: int = 60,
        unconfined_inner_late_dh: float = 1.0e-2,
        unconfined_inner_middle_dh: float = 1.0,
        inner_forcing_eta: float = 0.10,
        inner_head_residual_tol_min: float | None = None,
        inner_head_residual_tol_max: float = 1.0e-2,
        inner_picard_scale_max_fraction: float = 0.10,
        chebyshev_reset_factor: float = 1.2,
        chebyshev_minor_increase_patience: int = 2,
        transmissivity_relaxation_enabled: bool = False,
        transmissivity_relaxation_early: float = 0.25,
        transmissivity_relaxation_middle: float = 0.50,
        transmissivity_relaxation_late: float = 1.00,
        transmissivity_relaxation_middle_iteration: int = 5,
        transmissivity_relaxation_late_iteration: int = 15,
        unconfined_startup_mode: str = "initial_head",
        unconfined_pre_solve_iterations: int = 3,
        transient: bool = False,
        storage_coeff=None,
        dt=None,
        head_prev=None,
        refresh_diag_with_transient_storage: bool = True,
        storage_reference: str = "previous_period",
        unconfined_storage_mode_2d: str | None = None,
        sy: float | None = None,
        ss: float | None = None,
        accept_on_head_change_only: bool = False,
        practical_picard_acceptance_enabled: bool = False,
        min_practical_outer_iterations: int = 20,
        practical_residual_tol: float = 1.0e-4,
        practical_dh_rms_tol: float = 3.0e-3,
        practical_storage_diag_change_rms_tol: float = 30.0,
        save_transient_diagnostics: bool = False,
):
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module
    globals().update(kernel_module.__dict__)
    self = model
    """
    K-cycle multigrid using your existing hierarchy (self.mg_levels).

    Uses correction scheme (coarse RHS is restricted residual) and 2-term Krylov accel:
        z1 = B(b)
        r1 = b - A z1
        z2 = B(r1)
        alpha = (r1^T z2) / (z2^T A z2)
        e = z1 + alpha z2

    Optional hierarchy control:
      - min_coarse_cells: stop geometric coarsening before nx*ny drops below this.

    Optional robustness control:
      - fall back to fine-grid PCG if the checked residual grows well above the
        initial residual after a configurable number of K-cycles.

    Optional unconfined Picard controls:
      - residual_floor_tol: for unconfined solves, the inner linear residual
        threshold below which an outer Picard iteration may be accepted when
        the outer head change is small, even if the strict inner residual
        tolerance was not met. Set to None to disable practical convergence.
      - inner_head_residual_tol: head-equivalent residual tolerance for
        deciding whether an inner solve is usable for a Picard update.
        Defaults to the Picard head tolerance (hclose) for unconfined solves.
      - unconfined_inner_max_cycles_early/middle/late: adaptive K-cycle
        limits for early, middle, and late Picard outer iterations, selected
        based on the previous accepted nonlinear head-change measure.
      - unconfined_inner_late_dh/middle_dh: thresholds (meters) that select
        the adaptive inner-cycle limit.
      - inner_forcing_eta: fraction of the current Picard update scale used
        as a dynamic, inexact inner head-equivalent residual tolerance.
      - inner_head_residual_tol_min/max: bounds for the dynamic inner
        tolerance. Defaults to hclose and 1e-2 m respectively.
      - inner_picard_scale_max_fraction: fraction of the max Picard update
        included in the update-scale estimate.
      - chebyshev_reset_factor: multiplier on previous_measure that triggers
        a Chebyshev reset (was effectively 1.0).
      - chebyshev_minor_increase_patience: number of minor outer residual
        increases tolerated before resetting Chebyshev state.
      - transmissivity_relaxation_enabled and *_early/middle/late: optional
        under-relaxation of T(h) updates during early Picard iterations.
      - unconfined_startup_mode: "initial_head" keeps current behaviour;
        "confined_pre_solve" runs one fixed-T confined solve to warm-start;
        "unconfined_pre_solve" runs a few Picard sub-iterations that rebuild
        transmissivity from the current head (unconfined linearisation),
        controlled by unconfined_pre_solve_iterations.
      - storage_reference: "previous_period" keeps transient storage fixed
        from the caller-supplied storage_coeff. "current_picard" is a
        diagnostic path that rebuilds 2D unconfined storage from the current
        Picard head using sy/ss and unconfined_storage_mode_2d.
      - practical_picard_acceptance_enabled: optional production acceptance
        path for secant-Sy replay. Keeps strict Picard convergence metrics,
        but allows the nonlinear loop to stop when the head field and the
        storage linearisation have practically stabilised.
    """

    # Track whether a transient storage diagonal is in use so build_hierarchy
    # can skip per-level zero-storage device allocations for steady solves.
    storage_was_active = bool(self._storage_active)
    self._storage_active = bool(transient)

    # Normalize tolerances (treat None as disabled)
    dh_rms_tol_f = None if dh_rms_tol is None else float(dh_rms_tol)

    if dh_max_tol is None:
        dh_max_tol = None if dh_rms_tol_f is None else float(dh_max_factor) * dh_rms_tol_f
    else:
        dh_max_tol = float(dh_max_tol)

    if float(self.head_scale) != 1.0:
        raise ValueError(
            "K-cycle runs in physical head units only. "
            "Set head_scale=1.0 for K-cycle, or use PCG / 2-level MG if you want scaling."
        )

    if (aq_thickness is not None) or (gh_alpha is not None):
        self.update_ghb_factor_in_place(
            aq_thickness=aq_thickness,
            gh_alpha=gh_alpha,
        )

    smoother_mode = str(smoother).strip().lower()
    if smoother_mode not in {"chebyshev", "jacobi"}:
        raise ValueError("smoother must be 'chebyshev' or 'jacobi'.")
    if smoother_mode == "chebyshev":
        pre_omegas = _chebyshev_relaxation_sequence(
            order=int(nu_pre),
            lambda_min=float(cheby_lambda_min),
            lambda_max=float(cheby_lambda_max),
        )
        post_omegas = _chebyshev_relaxation_sequence(
            order=int(nu_post),
            lambda_min=float(cheby_lambda_min),
            lambda_max=float(cheby_lambda_max),
        )
    else:
        omega_f = float(omega)
        pre_omegas = tuple(omega_f for _ in range(int(nu_pre)))
        post_omegas = tuple(omega_f for _ in range(int(nu_post)))
    if len(pre_omegas) == 0:
        pre_omegas = (float(omega),)
    if len(post_omegas) == 0:
        post_omegas = (float(omega),)

    if bool(unconfined):
        return solve_unconfined_picard(model=self, state=locals())


    if bool(transient):
        # --- TRANSIENT STORAGE PREP ---
        dummy_rhs = np.zeros_like(self.T_field_host)
        _, new_sdiag, _, _, _ = _prepare_5point_transient_terms(
            rhs=dummy_rhs,
            storage_diag=None,
            active=self.active_host,
            bc_mask=self.bc_mask_host,
            bc_values=self.bc_values_host,
            transient=transient,
            storage_coeff=storage_coeff,
            dt=dt,
            head_prev=head_prev,
            initial_head=initial_head,
            dx=float(self.dx),
        )
        if not hasattr(self, "storage_diag_host") or self.storage_diag_host is None:
            self.storage_diag_host = np.zeros_like(self.T_field_host)
            self.storage_diag_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=self.device_str)

        hierarchy_missing_storage = False
        if self.mg_levels is not None and len(self.mg_levels) > 0:
            if getattr(self.mg_levels[-1], "storage_diag_wp", None) is None:
                hierarchy_missing_storage = True

        if np.any(self.storage_diag_host != new_sdiag) or not storage_was_active or hierarchy_missing_storage:
            self.storage_diag_host[...] = new_sdiag
            wp.copy(self.storage_diag_wp, wp.array(self.storage_diag_host, dtype=WP_FLOAT, device="cpu"))
            self._update_fine_diag_preconditioner()
            if refresh_diag_with_transient_storage or not storage_was_active or hierarchy_missing_storage:
                self._operator_dirty = True
                self._kcycle_graph = None
        # ------------------------------
    else:
        cleared_stale_storage = self._clear_transient_storage_state()
        if cleared_stale_storage or storage_was_active:
            self._update_fine_diag_preconditioner()
            self._operator_dirty = True
            self._kcycle_graph = None

    if not hasattr(self, "_kcycle_graph"):
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

    if self._operator_dirty or self.mg_levels is None:
        self.build_hierarchy(
            max_levels=int(max_levels),
            min_coarse_n=4,
            min_coarse_cells=min_coarse_cells,
        )

    levels = self.mg_levels
    if levels is None or len(levels) < 1:
        raise RuntimeError("No multigrid levels available. build_hierarchy() failed.")

    max_cycles_i = int(max_cycles)
    fallback_to_pcg_b = bool(fallback_to_pcg)
    divergence_cycle_start_i = max(1, int(divergence_cycle_start))
    divergence_residual_factor_f = float(divergence_residual_factor)
    if divergence_residual_factor_f <= 0.0:
        raise ValueError("divergence_residual_factor must be positive.")

    if fallback_pcg_max_iter is None:
        fallback_pcg_max_iter_i = max(5000, 50 * max_cycles_i)
    else:
        fallback_pcg_max_iter_i = int(fallback_pcg_max_iter)
        if fallback_pcg_max_iter_i < 1:
            raise ValueError("fallback_pcg_max_iter must be >= 1 when provided.")

    fallback_pcg_history_every_i = None if fallback_pcg_history_every is None else int(fallback_pcg_history_every)
    if fallback_pcg_history_every_i is not None and fallback_pcg_history_every_i <= 0:
        fallback_pcg_history_every_i = None

    device = self.device_str

    # Ensure every level has gh_mask_wp and ghb_factor_wp (allocate once if missing).
    for lvl in levels:
        shape = (int(lvl.ny), int(lvl.nx))
        if getattr(lvl, "gh_mask_wp", None) is None:
            lvl.gh_mask_wp = wp.zeros(shape, dtype=wp.int32, device=device)
        if getattr(lvl, "ghb_factor_wp", None) is None:
            lvl.ghb_factor_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)

    lvl0 = levels[0]
    ny0 = int(lvl0.ny)
    nx0 = int(lvl0.nx)
    dim0 = (ny0, nx0)

    # No allocations in solve: require hierarchy to have buffers.
    required = (
        "b_wp",
        "x_wp",
        "r_wp",
        "Ax_wp",
        "e_wp",
        "rho_buf",
        "converged_flag",
        "rTr_buf",
        "x_prev_wp",
        "dh_max_buf",
    )
    for name in required:
        if getattr(lvl0, name, None) is None:
            raise RuntimeError(
                f"Level 0 missing {name}. Ensure build_hierarchy() allocates all level buffers."
            )

    if tuple(lvl0.b_wp.shape) != (ny0, nx0) or tuple(lvl0.x_wp.shape) != (ny0, nx0):
        raise RuntimeError("Level 0 buffers have wrong shape. Rebuild hierarchy for this geometry.")

    # Solver-level CPU staging buffer for the initial head guess.
    if (
            not hasattr(self, "_kcycle_stage_x")
            or self._kcycle_stage_x is None
            or tuple(self._kcycle_stage_x.shape) != (ny0, nx0)
    ):
        self._kcycle_stage_x = wp.zeros((ny0, nx0), dtype=WP_FLOAT, device="cpu")

    # Finest RHS assembled via selected backend.
    self._build_rhs_fine(lvl0.b_wp)

    if bool(transient):
        # --- TRANSIENT RHS PREP ---
        b_eff, _, _, _, _ = _prepare_5point_transient_terms(
            rhs=lvl0.b_wp.numpy(),
            storage_diag=None,
            active=self.active_host,
            bc_mask=self.bc_mask_host,
            bc_values=self.bc_values_host,
            transient=transient,
            storage_coeff=storage_coeff,
            dt=dt,
            head_prev=head_prev,
            initial_head=initial_head,
            dx=float(self.dx),
        )
        if not hasattr(self, "_kcycle_stage_b") or self._kcycle_stage_b is None:
            self._kcycle_stage_b = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device="cpu")
        self._kcycle_stage_b.numpy()[...] = b_eff
        wp.copy(lvl0.b_wp, self._kcycle_stage_b)
        lvl0.storage_diag_host = self.storage_diag_host
        lvl0.storage_diag_wp = self.storage_diag_wp
        # --------------------------
    else:
        lvl0.storage_diag_host = None

    # Initial guess (host), then copy into persistent lvl0.x_wp
    x0 = np.zeros((ny0, nx0), dtype=NP_FLOAT)

    if initial_head is not None:
        init_arr = np.asarray(initial_head, dtype=NP_FLOAT)
        if init_arr.shape != (ny0, nx0):
            raise ValueError(f"initial_head must have shape ({ny0}, {nx0}), got {init_arr.shape}")
        x0[:, :] = init_arr

    bc_idx = np.asarray(self.bc_mask_host, dtype=np.int32) != 0
    x0[bc_idx] = np.asarray(self.bc_values_host, dtype=NP_FLOAT)[bc_idx]
    x0[np.asarray(self.active_host, dtype=np.int32) == 0] = NP_FLOAT(0.0)

    stage_x_np = self._kcycle_stage_x.numpy()
    stage_x_np[...] = x0
    wp.copy(lvl0.x_wp, self._kcycle_stage_x)

    # Snapshot initial x for dvclose-like metrics
    wp.launch(
        kernel=copy_field_kernel,
        dim=dim0,
        inputs=[lvl0.x_wp, lvl0.x_prev_wp, nx0, ny0],
        device=device,
    )

    # Zero coarse level buffers (still standalone; no reallocs)
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

    active_host_i32 = np.asarray(self.active_host, dtype=np.int32)
    bc_host_i32 = np.asarray(self.bc_mask_host, dtype=np.int32)

    free_mask = (active_host_i32 != 0) & (bc_host_i32 == 0)
    n_free0 = int(np.count_nonzero(free_mask))
    if n_free0 <= 0:
        head_out = lvl0.x_wp.numpy()
        info = {"solver_type": "kcycle", "n_cycles_used": 0, "converged": True}
        return (head_out, info) if return_info else head_out

    # Initial residual for tol computation (one scalar readback per solve)
    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
    _cr_k = compute_residual_kernel if self._storage_active else compute_residual_no_storage_kernel
    _cr_in = [
        lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
        lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
    ]
    if self._storage_active:
        _cr_in.append(lvl0.storage_diag_wp)
    _cr_in += [lvl0.r_wp, lvl0.rTr_buf, nx0, ny0]
    wp.launch(kernel=_cr_k, dim=dim0, inputs=_cr_in, device=device)
    rTr0 = float(lvl0.rTr_buf.numpy()[0])
    r_rms0 = float(np.sqrt(max(rTr0, 0.0) / float(n_free0)))
    tol_abs = float(max(abs_tol_min, rel_tol * r_rms0))
    thr_rTr = wp.float64((tol_abs * tol_abs) * float(n_free0))

    def pcg_solve_level(level, max_iter_level: int):
        nxL = int(level.nx)
        nyL = int(level.ny)
        dimL = (nyL, nxL)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rho_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)

        _ipcga_k = init_pcg_with_A_kernel if self._storage_active else init_pcg_with_A_no_storage_kernel
        _ipcga_in = [
            level.x_wp, level.b_wp, level.T_wp, level.active_wp, level.bc_mask_wp,
            level.gh_mask_wp, level.ghb_factor_wp,
        ]
        if self._storage_active:
            _ipcga_in.append(level.storage_diag_wp)
        _ipcga_in += [
            level.M_inv_wp, level.Ap_wp, level.r_wp, level.z_wp, level.p_wp,
            level.rho_buf, level.rTr_buf, nxL, nyL,
        ]
        wp.launch(kernel=_ipcga_k, dim=dimL, inputs=_ipcga_in, device=device)

        for _ in range(int(max_iter_level)):
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.pAp_buf], device=device)
            _aap_k = apply_A_and_pAp_kernel if self._storage_active else apply_A_and_pAp_no_storage_kernel
            _aap_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if self._storage_active:
                _aap_in.append(level.storage_diag_wp)
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
            _jac_k = jacobi_applyA_fused_kernel if self._storage_active else jacobi_applyA_fused_no_storage_kernel
            _jac_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if self._storage_active:
                _jac_in.append(level.storage_diag_wp)
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
        _cr_k = compute_residual_kernel if self._storage_active else compute_residual_no_storage_kernel
        _cr_in = [
            level.x_wp, level.b_wp, level.T_wp, level.active_wp, level.bc_mask_wp,
            level.gh_mask_wp, level.ghb_factor_wp,
        ]
        if self._storage_active:
            _cr_in.append(level.storage_diag_wp)
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
        _ccr_k = compute_residual_kernel if self._storage_active else compute_residual_no_storage_kernel
        _ccr_in = [
            z1_wp, coarse.b_wp, coarse.T_wp, coarse.active_wp, coarse.bc_mask_wp,
            coarse.gh_mask_wp, coarse.ghb_factor_wp,
        ]
        if self._storage_active:
            _ccr_in.append(coarse.storage_diag_wp)
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
        _caap_k = apply_A_and_pAp_kernel if self._storage_active else apply_A_and_pAp_no_storage_kernel
        _caap_in = [
            coarse.T_wp, coarse.active_wp, coarse.bc_mask_wp, coarse.gh_mask_wp,
            coarse.ghb_factor_wp,
        ]
        if self._storage_active:
            _caap_in.append(coarse.storage_diag_wp)
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
            _jac_k = jacobi_applyA_fused_kernel if self._storage_active else jacobi_applyA_fused_no_storage_kernel
            _jac_in = [
                level.T_wp, level.active_wp, level.bc_mask_wp, level.gh_mask_wp,
                level.ghb_factor_wp,
            ]
            if self._storage_active:
                _jac_in.append(level.storage_diag_wp)
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

    # Outer cycles
    n_cycles_used = 0
    converged = False

    check_every = check_every_no  # reduce sync frequency; set to 1 for debugging

    graph_key = [
        "kcycle",
        int(len(levels)),
        tuple((int(l.ny), int(l.nx)) for l in levels),
        int(nu_pre),
        int(nu_post),
        int(nu_coarse),
        str(smoother_mode),
        tuple(float(v) for v in pre_omegas),
        tuple(float(v) for v in post_omegas),
        float(omega),
        bool(self._storage_active),
    ]
    if not self.trust_ghb_params_for_graph:
        graph_key.append(float(self.gh_alpha))
        graph_key.append(float(self.aq_thickness))

    graph_key = tuple(graph_key)

    graph_built_this_call = False
    use_cuda_graph = str(device).startswith("cuda")

    dh_rms_lastcheck = float("nan")
    dh_max_lastcheck = float("nan")
    history: list[dict[str, float | int | bool | None]] = [
        {
            "cycle": 0,
            "r_rms": float(r_rms0),
            "tol_abs": float(tol_abs),
            "dh_rms": None,
            "dh_max": None,
            "res_ok": None,
            "dh_ok": None,
        }
    ]

    for cyc in range(max_cycles_i):
        n_cycles_used = cyc + 1

        if not use_cuda_graph:
            kcycle(0)
        elif self._kcycle_graph is None or self._kcycle_graph_shape != graph_key:
            with wp.ScopedCapture() as cap:
                kcycle(0)
            self._kcycle_graph = cap.graph
            self._kcycle_graph_shape = graph_key
            graph_built_this_call = True
        else:
            wp.capture_launch(self._kcycle_graph)

        if (cyc % int(check_every)) != (int(check_every) - 1):
            continue

        # (A) dvclose-like diagnostics and (B) flux residual check in one pass.
        wp.launch(
            kernel=reset_kcycle_check_buffers_kernel,
            dim=1,
            inputs=[lvl0.rho_buf, lvl0.dh_max_buf, lvl0.rTr_buf, lvl0.converged_flag],
            device=device,
        )
        _kc_k = kcycle_check_dh_and_residual_kernel if self._storage_active else kcycle_check_dh_and_residual_no_storage_kernel
        _kc_in = [
            lvl0.x_wp, lvl0.x_prev_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp,
            lvl0.bc_mask_wp, lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
        ]
        if self._storage_active:
            _kc_in.append(lvl0.storage_diag_wp)
        _kc_in += [lvl0.rho_buf, lvl0.dh_max_buf, lvl0.rTr_buf, int(1 if self.use_ghb else 0), nx0, ny0]  # rho_buf=dh2, rTr_buf=residual
        wp.launch(kernel=_kc_k, dim=dim0, inputs=_kc_in, device=device)
        wp.launch(
            kernel=check_rtr_converged_kernel,
            dim=1,
            inputs=[lvl0.rTr_buf, thr_rTr, lvl0.converged_flag],
            device=device,
        )

        dh2 = float(lvl0.rho_buf.numpy()[0])
        dh_rms_lastcheck = float(np.sqrt(max(dh2, 0.0) / float(n_free0)))
        dh_max_lastcheck = float(lvl0.dh_max_buf.numpy()[0])

        dh_ok = True
        if dh_max_tol is not None and dh_rms_tol is not None:
            dh_ok = dh_max_lastcheck <= float(dh_max_tol) and dh_rms_lastcheck <= float(dh_rms_tol)

        res_ok = int(lvl0.converged_flag.numpy()[0]) != 0
        rTr_check = float(lvl0.rTr_buf.numpy()[0])
        r_rms_check = float(np.sqrt(max(rTr_check, 0.0) / float(n_free0)))
        history.append(
            {
                "cycle": int(n_cycles_used),
                "r_rms": float(r_rms_check),
                "tol_abs": float(tol_abs),
                "dh_rms": float(dh_rms_lastcheck),
                "dh_max": float(dh_max_lastcheck),
                "res_ok": bool(res_ok),
                "dh_ok": bool(dh_ok),
            }
        )

        if res_ok and dh_ok:
            converged = True
            break

        if (
            fallback_to_pcg_b
            and n_cycles_used >= divergence_cycle_start_i
            and r_rms_check > (divergence_residual_factor_f * r_rms0)
        ):
            fallback_head0 = np.asarray(lvl0.x_wp.numpy(), dtype=NP_FLOAT)
            head_pcg, info_pcg = self._solve_pcg_device_loop(
                max_iter=int(fallback_pcg_max_iter_i),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                initial_head=fallback_head0,
                history_every=fallback_pcg_history_every_i,
            )
            info_pcg = dict(info_pcg)
            info_pcg["fallback_from"] = "kcycle"
            info_pcg["fallback_reason"] = "diverging_residual"
            info_pcg["fallback_trigger_cycle"] = int(n_cycles_used)
            info_pcg["fallback_trigger_r_rms"] = float(r_rms_check)
            info_pcg["fallback_trigger_threshold"] = float(divergence_residual_factor_f * r_rms0)
            info_pcg["kcycle_history_before_fallback"] = list(history)
            info_pcg["kcycle_coarsening_diagnostics"] = [dict(item) for item in self._mg_coarsening_diagnostics]
            return (head_pcg, info_pcg) if return_info else head_pcg

    # Final head pullback
    head_out = lvl0.x_wp.numpy()

    # Final flux residual RMS for reporting
    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
    _cr_k = compute_residual_kernel if self._storage_active else compute_residual_no_storage_kernel
    _cr_in = [
        lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
        lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
    ]
    if self._storage_active:
        _cr_in.append(lvl0.storage_diag_wp)
    _cr_in += [lvl0.r_wp, lvl0.rTr_buf, nx0, ny0]
    wp.launch(kernel=_cr_k, dim=dim0, inputs=_cr_in, device=device)
    rTr_end = float(lvl0.rTr_buf.numpy()[0])
    r_rms_end = float(np.sqrt(max(rTr_end, 0.0) / float(n_free0)))

    # Head-equivalent residual RMS for reporting
    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
    _hr_k = compute_head_residual_kernel if self._storage_active else compute_head_residual_no_storage_kernel
    _hr_in = [
        lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
        lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
    ]
    if self._storage_active:
        _hr_in.append(lvl0.storage_diag_wp)
    _hr_in += [lvl0.r_wp, lvl0.rTr_buf, nx0, ny0]  # r stores r_h [m]; rTr_buf sums r_h^2
    wp.launch(kernel=_hr_k, dim=dim0, inputs=_hr_in, device=device)
    hrTr_end = float(lvl0.rTr_buf.numpy()[0])
    h_rms_end = float(np.sqrt(max(hrTr_end, 0.0) / float(n_free0)))

    # For dvclose-like metrics: the last check is the meaningful "end" value
    dh_rms_end = float(dh_rms_lastcheck)
    dh_max_end = float(dh_max_lastcheck)

    info = {
        "solver_type": "kcycle",
        "n_levels": int(len(levels)),
        "max_cycles": int(max_cycles),
        "n_cycles_used": int(n_cycles_used),
        "nu_pre": int(nu_pre),
        "nu_post": int(nu_post),
        "nu_coarse": int(nu_coarse),
        "smoother": str(smoother_mode),
        "omega": float(omega),
        "cheby_lambda_min": float(cheby_lambda_min) if smoother_mode == "chebyshev" else float("nan"),
        "cheby_lambda_max": float(cheby_lambda_max) if smoother_mode == "chebyshev" else float("nan"),
        "cheby_pre_omegas": [float(v) for v in pre_omegas],
        "cheby_post_omegas": [float(v) for v in post_omegas],
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "tol_abs": float(tol_abs),
        "r_rms0": float(r_rms0),
        "r_rms_end": float(r_rms_end),
        "h_rms_end": float(h_rms_end),
        "dh_rms_lastcheck": float(dh_rms_lastcheck),
        "dh_max_lastcheck": float(dh_max_lastcheck),
        "dh_rms_end": float(dh_rms_end),
        "dh_max_end": float(dh_max_end),
        "converged": bool(converged),
        "aq_thickness": float(self.aq_thickness),
        "use_ghb": bool(self.use_ghb),
        "diag_preconditioner_backend": self._diag_backend_env_or_default(),
        "cuda_graph_reused": bool((not graph_built_this_call) and (self._kcycle_graph is not None)),
        "cuda_graph_built_this_call": bool(graph_built_this_call),
        "check_every": int(check_every),
        "min_coarse_cells": None if min_coarse_cells is None else int(min_coarse_cells),
        "coarsening_diagnostics": [dict(item) for item in self._mg_coarsening_diagnostics],
        "update_T_profile_last": None if self._last_update_T_profile is None else dict(self._last_update_T_profile),
        "update_T_profile_totals": None if self._update_T_profile_totals is None else dict(self._update_T_profile_totals),
    }
    if (not history) or int(history[-1]["cycle"]) != int(n_cycles_used):
        history.append(
            {
                "cycle": int(n_cycles_used),
                "r_rms": float(r_rms_end),
                "tol_abs": float(tol_abs),
                "dh_rms": float(dh_rms_end) if np.isfinite(dh_rms_end) else None,
                "dh_max": float(dh_max_end) if np.isfinite(dh_max_end) else None,
                "res_ok": bool(r_rms_end <= float(tol_abs)),
                "dh_ok": (
                    None
                    if (dh_rms_tol_f is None or dh_max_tol is None)
                    else bool(
                        np.isfinite(dh_rms_end)
                        and np.isfinite(dh_max_end)
                        and dh_rms_end <= float(dh_rms_tol_f)
                        and dh_max_end <= float(dh_max_tol)
                    )
                ),
            }
        )
    info["history"] = history

    return (head_out, info) if return_info else head_out
