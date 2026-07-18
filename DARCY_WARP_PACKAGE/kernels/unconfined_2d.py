# SPDX-License-Identifier: AGPL-3.0-only
"""Unconfined transmissivity, storage, and Picard-update kernel exports."""

from __future__ import annotations


_NAMES = {
    "apply_relaxed_clipped_picard_update_kernel", "apply_relaxed_correction_kernel",
    "build_transient_rhs_from_storage_kernel", "clamp_unconfined_head_kernel",
    "head_update_rms_and_snapshot_kernel", "update_secant_sy_storage_kernel",
    "update_unconfined_transmissivity_from_head_kernel",
}


def __getattr__(name: str):
    if name not in _NAMES:
        raise AttributeError(name)
    from DARCY_WARP_PACKAGE import warped_darcy

    return getattr(warped_darcy, name)
