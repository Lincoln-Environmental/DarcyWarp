# SPDX-License-Identifier: AGPL-3.0-only
"""Shared hierarchy data types for geometric multigrid K-cycle backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LinearGridLevel:
    """Fine/coarse linear-operator view used by the legacy two-level path."""

    T_wp: Any
    active_wp: Any
    bc_mask_wp: Any
    gh_mask_wp: Any
    gh_width_wp: Any
    ghb_factor_wp: Any
    storage_diag_wp: Any = None
    M_inv_wp: Any = None
    nx: int = 0
    ny: int = 0
    dx: float = 0.0


@dataclass(slots=True)
class MGLevel:
    """One model-owned multigrid level and its persistent Warp work arrays."""

    level_id: int
    nx: int
    ny: int
    dx: float
    n_active: int
    T_host: Any
    R_host: Any
    active_host: Any
    bc_mask_host: Any
    bc_values_host: Any
    gh_mask_host: Any
    gh_head_host: Any
    gh_width_host: Any
    ghb_factor_host: Any
    storage_diag_host: Any
    T_wp: Any
    R_wp: Any
    active_wp: Any
    bc_mask_wp: Any
    bc_values_wp: Any
    gh_mask_wp: Any
    gh_head_wp: Any
    gh_width_wp: Any
    ghb_factor_wp: Any
    storage_diag_wp: Any
    M_inv_wp: Any
    x_wp: Any
    b_wp: Any
    r_wp: Any
    Ax_wp: Any
    e_wp: Any
    z_wp: Any
    p_wp: Any
    Ap_wp: Any
    rTr_buf: Any
    rho_buf: Any
    rho_new_buf: Any
    pAp_buf: Any
    alpha_buf: Any
    beta_buf: Any
    converged_flag: Any
    x_prev_wp: Any = None
    dh_max_buf: Any = None
