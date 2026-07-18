# SPDX-License-Identifier: AGPL-3.0-only
"""Operator and residual kernel compatibility exports.

Kernel definitions remain byte-for-byte in the compatibility module until the
linear-solver extraction moves them as one verified family.
"""

from __future__ import annotations


_NAMES = {
    "apply_A_kernel", "apply_A_and_pAp_kernel", "apply_A_and_pAp_no_storage_kernel",
    "build_diag_preconditioner_kernel", "build_diag_preconditioner_no_storage_kernel",
    "build_rhs_fd_like", "build_rhs_kernel", "compute_head_residual_kernel",
    "compute_head_residual_no_storage_kernel", "compute_residual_kernel",
    "compute_residual_no_storage_kernel", "enforce_constraints_kernel",
}


def __getattr__(name: str):
    if name not in _NAMES:
        raise AttributeError(name)
    from DARCY_WARP_PACKAGE import warped_darcy

    return getattr(warped_darcy, name)

