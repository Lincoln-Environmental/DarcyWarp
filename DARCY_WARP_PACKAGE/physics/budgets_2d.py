# SPDX-License-Identifier: AGPL-3.0-only
"""2D discrete operator budget evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mass_balance_budget(
    T_field: np.ndarray,
    R_field: np.ndarray,
    head: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    dx: float,
    gh_mask: np.ndarray | None = None,
    gh_head: np.ndarray | None = None,
    gh_width: np.ndarray | None = None,
    ghb_factor: np.ndarray | None = None,
    gh_alpha: float = 1.0,
    aq_thickness: float = 1.0,
    case: str | None = None,
) -> pd.DataFrame:
    """Compute the MF6-like discrete recharge, CHD, and GHB budget.

    This is the original vectorised formulation, retained verbatim in sign and
    interface ordering while moving it out of the model implementation.
    """
    T = np.asarray(T_field, dtype=np.float64)
    R = np.asarray(R_field, dtype=np.float64)
    h = np.asarray(head, dtype=np.float64)
    act = np.asarray(active, dtype=np.int32) != 0
    bc = np.asarray(bc_mask, dtype=np.int32) != 0
    bc_v = np.asarray(bc_values, dtype=np.float64)
    ny, nx = T.shape
    if h.shape != (ny, nx):
        raise ValueError("head shape mismatch")
    if R.shape != (ny, nx):
        raise ValueError("R_field shape mismatch")

    dx_f = float(dx)
    tiny = 1.0e-12
    h_use = np.array(h, copy=True)
    h_use[~act] = 0.0
    h_use[bc] = bc_v[bc]
    cell_is_interior = act & (~bc)

    r_cell = np.zeros((ny, nx), dtype=np.float64)
    r_cell[cell_is_interior] = R[cell_is_interior] * dx_f * dx_f
    rcha_in = float(np.sum(np.maximum(r_cell, 0.0)))
    rcha_out = float(np.sum(np.maximum(-r_cell, 0.0)))

    act_l = act[:, :-1]
    act_r = act[:, 1:]
    conn_e = act_l & act_r
    t_l = T[:, :-1]
    t_r = T[:, 1:]
    denom_e = t_l + t_r
    cond_e = np.zeros((ny, nx - 1), dtype=np.float64)
    valid_e = conn_e & (t_l > 0.0) & (t_r > 0.0) & (denom_e > tiny)
    cond_e[valid_e] = 2.0 * t_l[valid_e] * t_r[valid_e] / denom_e[valid_e]
    q_l_to_r = cond_e * (h_use[:, :-1] - h_use[:, 1:])
    bc_l = bc[:, :-1]
    bc_r = bc[:, 1:]
    q_int_to_bc_l = np.where(conn_e & (~bc_l) & bc_r, q_l_to_r, 0.0)
    q_int_to_bc_r = np.where(conn_e & bc_l & (~bc_r), -q_l_to_r, 0.0)
    chd_out = float(np.sum(np.maximum(q_int_to_bc_l, 0.0))) + float(np.sum(np.maximum(q_int_to_bc_r, 0.0)))
    chd_in = float(np.sum(np.maximum(-q_int_to_bc_l, 0.0))) + float(np.sum(np.maximum(-q_int_to_bc_r, 0.0)))

    act_t = act[:-1, :]
    act_b = act[1:, :]
    conn_s = act_t & act_b
    t_t = T[:-1, :]
    t_b = T[1:, :]
    denom_s = t_t + t_b
    cond_s = np.zeros((ny - 1, nx), dtype=np.float64)
    valid_s = conn_s & (t_t > 0.0) & (t_b > 0.0) & (denom_s > tiny)
    cond_s[valid_s] = 2.0 * t_t[valid_s] * t_b[valid_s] / denom_s[valid_s]
    q_t_to_b = cond_s * (h_use[:-1, :] - h_use[1:, :])
    bc_t = bc[:-1, :]
    bc_b = bc[1:, :]
    q_int_to_bc_t = np.where(conn_s & (~bc_t) & bc_b, q_t_to_b, 0.0)
    q_int_to_bc_b = np.where(conn_s & bc_t & (~bc_b), -q_t_to_b, 0.0)
    chd_out += float(np.sum(np.maximum(q_int_to_bc_t, 0.0))) + float(np.sum(np.maximum(q_int_to_bc_b, 0.0)))
    chd_in += float(np.sum(np.maximum(-q_int_to_bc_t, 0.0))) + float(np.sum(np.maximum(-q_int_to_bc_b, 0.0)))
    chd_net_out_positive = chd_out - chd_in

    ghb_in = 0.0
    ghb_out = 0.0
    ghb_net_out_positive = 0.0
    if (gh_mask is not None) and (gh_head is not None) and ((gh_width is not None) or (ghb_factor is not None)):
        ghm = np.asarray(gh_mask, dtype=np.int32) != 0
        ghe = np.asarray(gh_head, dtype=np.float64)
        if ghe.shape != (ny, nx):
            raise ValueError("gh_head shape mismatch")
        if ghb_factor is not None:
            ghbf = np.asarray(ghb_factor, dtype=np.float64)
            if ghbf.shape != (ny, nx):
                raise ValueError("ghb_factor shape mismatch")
            c_gh = (T * ghbf).astype(np.float64, copy=False)
            gh_ok = np.isfinite(ghbf) & (ghbf > 0.0)
        else:
            ghw = np.asarray(gh_width, dtype=np.float64)
            if ghw.shape != (ny, nx):
                raise ValueError("gh_width shape mismatch")
            if float(aq_thickness) <= 0.0:
                raise ValueError("aq_thickness must be positive")
            c_gh = (float(gh_alpha) * T / float(aq_thickness) * ghw * dx_f).astype(np.float64)
            gh_ok = np.isfinite(ghw) & (ghw > 0.0)
        mask_gh = ghm & cell_is_interior & gh_ok & np.isfinite(ghe) & np.isfinite(h_use)
        q_gh = np.zeros((ny, nx), dtype=np.float64)
        q_gh[mask_gh] = c_gh[mask_gh] * (h_use[mask_gh] - ghe[mask_gh])
        ghb_out = float(np.sum(np.maximum(q_gh, 0.0)))
        ghb_in = float(np.sum(np.maximum(-q_gh, 0.0)))
        ghb_net_out_positive = ghb_out - ghb_in

    total_in = rcha_in + chd_in + ghb_in
    total_out = rcha_out + chd_out + ghb_out
    in_minus_out = total_in - total_out
    denom = abs(total_in) + abs(total_out)
    percent_discrepancy = 0.0 if denom == 0.0 else 100.0 * in_minus_out / denom
    throughflow = 0.5 * (total_in + total_out)
    imbalance_fraction = 0.0 if throughflow == 0.0 else in_minus_out / throughflow
    return pd.DataFrame([{
        "case": "" if case is None else str(case),
        "rcha_in": rcha_in, "rcha_out": rcha_out,
        "chd_in": chd_in, "chd_out": chd_out,
        "ghb_in": ghb_in, "ghb_out": ghb_out,
        "total_in": total_in, "total_out": total_out,
        "in_minus_out": in_minus_out, "percent_discrepancy": percent_discrepancy,
        "throughflow": throughflow, "imbalance_fraction": imbalance_fraction,
        "chd_net_out_positive": chd_net_out_positive,
        "ghb_net_out_positive": ghb_net_out_positive,
    }])
