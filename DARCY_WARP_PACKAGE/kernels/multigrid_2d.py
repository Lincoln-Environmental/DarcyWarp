# SPDX-License-Identifier: AGPL-3.0-only
"""Restriction, prolongation, smoothing, and Krylov kernel exports."""

from __future__ import annotations


_NAMES = {
    "add_correction_kernel", "axpy_active_scalar_2dmask_kernel",
    "compute_alpha_kernel", "compute_beta_and_update_rho_kernel",
    "copy_field_kernel", "dot_active_kernel", "init_pcg_kernel",
    "init_pcg_with_A_kernel", "init_pcg_with_A_no_storage_kernel",
    "jacobi_applyA_fused_kernel", "jacobi_applyA_fused_no_storage_kernel",
    "prolong_bilinear_any_kernel", "restrict_blockavg_kernel", "update_p_kernel",
    "update_x_r_z_rho_rTr_kernel", "zero_field_kernel", "zero_scalar_kernel",
}


def __getattr__(name: str):
    if name not in _NAMES:
        raise AttributeError(name)
    from DARCY_WARP_PACKAGE import warped_darcy

    return getattr(warped_darcy, name)

