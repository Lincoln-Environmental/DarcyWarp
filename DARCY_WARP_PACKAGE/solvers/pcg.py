# SPDX-License-Identifier: AGPL-3.0-only
"""Confined, steady-state preconditioned conjugate-gradient backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .base import SolverContext


class ConfinedPCGBackend:
    """Adapter for the established device PCG implementation."""

    name = "confined_pcg"

    def solve(self, context: SolverContext, **kwargs: Any):
        max_iter = int(kwargs.pop("pcg_max_iter", kwargs.pop("max_iter", 250)))
        rel_tol = float(kwargs.pop("rel_tol", 5.0e-7))
        abs_tol_min = float(kwargs.pop("abs_tol_min", 5.0e-7))
        initial_head = kwargs.pop("initial_head", None)
        history_every = kwargs.pop("history_every", None)
        if kwargs:
            raise TypeError(
                "unused solve kwargs for solver='confined_pcg': "
                f"{sorted(kwargs.keys())}"
            )
        return context.run_pcg(
            max_iter=max_iter,
            rel_tol=rel_tol,
            abs_tol_min=abs_tol_min,
            initial_head=initial_head,
            history_every=history_every,
        )


def solve_pcg_device_loop(
    *,
    model: Any,
    max_iter: int,
    rel_tol: float,
    abs_tol_min: float,
    initial_head: np.ndarray | None,
    history_every: int | None = None,
):
    """Established device PCG loop, extracted without changing launch order.

    ``model`` is a deliberately temporary compatibility bridge for the model's
    already-owned Warp buffers and assembly helpers.  It is removed once those
    helpers are represented by the typed resource/workspace context.
    """
    from DARCY_WARP_PACKAGE import warped_darcy as kernels

    if model.T_field_host is None:
        raise RuntimeError("build_from_truth_inputs must be called before solve().")
    device = model.device_str
    nx = int(model.nx)
    ny = int(model.ny)
    model._pcg_build_rhs_and_upload()
    model._pcg_initialize_guess_and_upload(initial_head=initial_head)
    model._pcg_reset_work_vectors()
    dim = (ny, nx)
    wp.launch(kernel=kernels.zero_scalar_kernel, dim=1, inputs=[model.rho_buf], device=device)
    wp.launch(kernel=kernels.zero_scalar_kernel, dim=1, inputs=[model.rTr_buf], device=device)
    wp.launch(
        kernel=kernels.init_pcg_with_A_no_storage_kernel,
        dim=dim,
        inputs=[
            model.x_wp, model.b_wp, model.T_wp, model.active_wp, model.bc_mask_wp,
            model.gh_mask_wp, model.ghb_factor_wp, model.M_inv_wp, model.Ap_wp,
            model.r_wp, model.z_wp, model.p_wp, model.rho_buf, model.rTr_buf, nx, ny,
        ],
        device=device,
    )
    wp.launch(
        kernel=kernels.enforce_constraints_kernel,
        dim=dim,
        inputs=[
            model.x_wp, model.r_wp, model.z_wp, model.p_wp, model.active_wp,
            model.bc_mask_wp, model.bc_values_wp, float(model.head_scale),
        ],
        device=device,
    )
    r_tr0 = float(model.rTr_buf.numpy()[0])
    r_rms0_scaled = (
        float(np.sqrt(r_tr0 / float(model.n_active)))
        if model.n_active > 0 and r_tr0 > 0.0
        else 0.0
    )
    r_rms0_phys = r_rms0_scaled * float(model.head_scale)
    abs_tol_scaled = float(abs_tol_min) / float(model.head_scale)
    tol_abs_scaled = max(abs_tol_scaled, float(rel_tol) * r_rms0_scaled)
    n_iter_used = 0
    converged = False
    history_every_i = None if history_every is None else int(history_every)
    if history_every_i is not None and history_every_i <= 0:
        history_every_i = None
    history: list[dict[str, float | int | bool]] = []
    if history_every_i is not None:
        history.append({
            "iter": 0,
            "rms_res_phys": float(r_rms0_phys),
            "tol_abs_phys": float(tol_abs_scaled * float(model.head_scale)),
        })
    for iteration in range(int(max_iter)):
        n_iter_used = iteration + 1
        wp.launch(kernel=kernels.zero_scalar_kernel, dim=1, inputs=[model.pAp_buf], device=device)
        wp.launch(
            kernel=kernels.apply_A_and_pAp_no_storage_kernel,
            dim=dim,
            inputs=[
                model.T_wp, model.active_wp, model.bc_mask_wp, model.gh_mask_wp,
                model.ghb_factor_wp, model.p_wp, model.Ap_wp, model.pAp_buf, nx, ny,
            ],
            device=device,
        )
        wp.launch(
            kernel=kernels.compute_alpha_kernel,
            dim=1,
            inputs=[model.rho_buf, model.pAp_buf, model.alpha_buf],
            device=device,
        )
        wp.launch(kernel=kernels.zero_scalar_kernel, dim=1, inputs=[model.rho_new_buf], device=device)
        wp.launch(kernel=kernels.zero_scalar_kernel, dim=1, inputs=[model.rTr_buf], device=device)
        wp.launch(
            kernel=kernels.update_x_r_z_rho_rTr_kernel,
            dim=dim,
            inputs=[
                model.x_wp, model.r_wp, model.z_wp, model.p_wp, model.Ap_wp,
                model.M_inv_wp, model.active_wp, model.bc_mask_wp, model.alpha_buf,
                model.rho_new_buf, model.rTr_buf, nx, ny,
            ],
            device=device,
        )
        wp.launch(
            kernel=kernels.check_convergence_kernel,
            dim=1,
            inputs=[model.rTr_buf, int(model.n_active), float(tol_abs_scaled), model.converged_flag],
            device=device,
        )
        if history_every_i is not None and (
            n_iter_used % history_every_i == 0 or n_iter_used == int(max_iter)
        ):
            r_tr_now = float(model.rTr_buf.numpy()[0]) if model.n_active > 0 else 0.0
            r_rms_now_scaled = (
                float(np.sqrt(r_tr_now / float(model.n_active)))
                if model.n_active > 0 and r_tr_now >= 0.0
                else 0.0
            )
            history.append({
                "iter": int(n_iter_used),
                "rms_res_phys": float(r_rms_now_scaled * float(model.head_scale)),
                "tol_abs_phys": float(tol_abs_scaled * float(model.head_scale)),
            })
        if int(model.converged_flag.numpy()[0]) == 1:
            converged = True
            break
        wp.launch(
            kernel=kernels.compute_beta_and_update_rho_kernel,
            dim=1,
            inputs=[model.rho_buf, model.rho_new_buf, model.beta_buf],
            device=device,
        )
        wp.launch(
            kernel=kernels.update_p_kernel,
            dim=dim,
            inputs=[model.p_wp, model.z_wp, model.active_wp, model.bc_mask_wp, model.beta_buf, nx, ny],
            device=device,
        )
    head = model.x_wp.numpy() * float(model.head_scale)
    r_tr_final = float(model.rTr_buf.numpy()[0]) if model.n_active > 0 else 0.0
    r_rms_final_scaled = (
        float(np.sqrt(r_tr_final / float(model.n_active)))
        if model.n_active > 0 and r_tr_final >= 0.0
        else 0.0
    )
    r_rms_final_phys = r_rms_final_scaled * float(model.head_scale)
    tol_abs_phys = float(tol_abs_scaled) * float(model.head_scale)
    info = {
        "solver_type": "pcg", "nx": int(model.nx), "ny": int(model.ny),
        "n_cells_total": int(model.nx * model.ny), "n_iter_used": int(n_iter_used),
        "max_iter": int(max_iter), "converged": bool(converged), "rel_tol": float(rel_tol),
        "abs_tol_min_phys": float(abs_tol_min), "tol_abs_phys": float(tol_abs_phys),
        "head_scale": float(model.head_scale), "rms_res_initial_phys": float(r_rms0_phys),
        "rms_res_final_phys": float(r_rms_final_phys),
    }
    if history_every_i is not None:
        if not history or int(history[-1]["iter"]) != int(n_iter_used):
            history.append({
                "iter": int(n_iter_used), "rms_res_phys": float(r_rms_final_phys),
                "tol_abs_phys": float(tol_abs_phys),
            })
        info["history_every"] = int(history_every_i)
        info["history"] = history
    return head, info
