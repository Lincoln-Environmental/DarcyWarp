# SPDX-License-Identifier: AGPL-3.0-or-later
"""2D discrete operator budget evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _free_components(free, east, south):
    """Label free cells joined by positive face conductance, excluding CHD cells."""
    from scipy.ndimage import label

    neighbours_e = free[:, :-1] & free[:, 1:]
    neighbours_s = free[:-1, :] & free[1:, :]
    if not (np.any(neighbours_e & (east == 0.0)) or np.any(neighbours_s & (south == 0.0))):
        return label(free)
    # Insert face slots so a zero conductance can split adjacent active cells.
    ny, nx = free.shape
    grid = np.zeros((2 * ny - 1, 2 * nx - 1), dtype=bool)
    grid[::2, ::2] = free
    grid[::2, 1::2] = neighbours_e & (east > 0.0)
    grid[1::2, ::2] = neighbours_s & (south > 0.0)
    labels, count = label(grid)
    return labels[::2, ::2], count


def _component_sum(labels, values):
    return np.bincount(labels.ravel(), weights=values.ravel())[1:]


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
    *,
    drn_mask: np.ndarray | None = None,
    drn_elev: np.ndarray | None = None,
    drn_cond: np.ndarray | None = None,
    riv_mask: np.ndarray | None = None,
    riv_stage: np.ndarray | None = None,
    riv_rbot: np.ndarray | None = None,
    riv_cond: np.ndarray | None = None,
    return_cells: bool = False,
    component_budgets: bool = False,
) -> pd.DataFrame | tuple[dict, np.ndarray, np.ndarray, float]:
    """Compute the MF6-like discrete recharge, CHD, and GHB budget.

    Optional cell diagnostics include DRN and RIV exchange and their active
    derivatives. With return_cells, component_budgets also groups external
    exchange by hydraulically connected free cells for transient acceptance.
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

    q_gh = np.zeros_like(h_use)
    c_gh = np.zeros_like(h_use)
    gh_stage = np.zeros_like(h_use)
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
        c_gh[~mask_gh] = 0.0
        gh_stage[mask_gh] = ghe[mask_gh]
        q_gh[mask_gh] = c_gh[mask_gh] * (h_use[mask_gh] - ghe[mask_gh])
        ghb_out = float(np.sum(np.maximum(q_gh, 0.0)))
        ghb_in = float(np.sum(np.maximum(-q_gh, 0.0)))
        ghb_net_out_positive = ghb_out - ghb_in

    q_drn = None
    drn_in = 0.0
    drn_out = 0.0
    drn_net_out_positive = 0.0
    if (drn_mask is not None) and (drn_elev is not None) and (drn_cond is not None):
        dm = np.asarray(drn_mask, dtype=np.int32) != 0
        de = np.asarray(drn_elev, dtype=np.float64)
        dc = np.asarray(drn_cond, dtype=np.float64)
        for name, arr in (("drn_elev", de), ("drn_cond", dc)):
            if arr.shape != (ny, nx):
                raise ValueError(f"{name} shape mismatch")
        mask_drn = dm & cell_is_interior & np.isfinite(dc) & (dc > 0.0) & np.isfinite(de)
        q_drn = np.zeros((ny, nx), dtype=np.float64)
        q_drn[mask_drn] = dc[mask_drn] * np.maximum(h_use[mask_drn] - de[mask_drn], 0.0)
        # DRN is outflow-only by construction.
        drn_out = float(np.sum(q_drn))
        drn_net_out_positive = drn_out

    q_riv = None
    riv_in = 0.0
    riv_out = 0.0
    riv_net_out_positive = 0.0
    if (
        (riv_mask is not None) and (riv_stage is not None)
        and (riv_rbot is not None) and (riv_cond is not None)
    ):
        rm = np.asarray(riv_mask, dtype=np.int32) != 0
        rs = np.asarray(riv_stage, dtype=np.float64)
        rb = np.asarray(riv_rbot, dtype=np.float64)
        rc = np.asarray(riv_cond, dtype=np.float64)
        for name, arr in (("riv_stage", rs), ("riv_rbot", rb), ("riv_cond", rc)):
            if arr.shape != (ny, nx):
                raise ValueError(f"{name} shape mismatch")
        mask_riv = rm & cell_is_interior & np.isfinite(rc) & (rc > 0.0) & np.isfinite(rs) & np.isfinite(rb)
        q_riv = np.zeros((ny, nx), dtype=np.float64)
        q_riv[mask_riv] = rc[mask_riv] * (np.maximum(h_use[mask_riv], rb[mask_riv]) - rs[mask_riv])
        riv_out = float(np.sum(np.maximum(q_riv, 0.0)))
        riv_in = float(np.sum(np.maximum(-q_riv, 0.0)))
        riv_net_out_positive = riv_out - riv_in

    total_in = rcha_in + chd_in + ghb_in + drn_in + riv_in
    total_out = rcha_out + chd_out + ghb_out + drn_out + riv_out
    in_minus_out = total_in - total_out
    throughflow = 0.5 * (total_in + total_out)
    imbalance_fraction = 0.0 if throughflow == 0.0 else in_minus_out / throughflow
    row = {
        "case": "" if case is None else str(case),
        "rcha_in": rcha_in, "rcha_out": rcha_out,
        "chd_in": chd_in, "chd_out": chd_out,
        "ghb_in": ghb_in, "ghb_out": ghb_out,
        "drn_in": drn_in, "drn_out": drn_out, "riv_in": riv_in, "riv_out": riv_out,
        "total_in": total_in, "total_out": total_out,
        "in_minus_out": in_minus_out, "percent_discrepancy": 100.0 * imbalance_fraction,
        "throughflow": throughflow, "imbalance_fraction": imbalance_fraction,
        "chd_net_out_positive": chd_net_out_positive,
        "ghb_net_out_positive": ghb_net_out_positive,
        "drn_net_out_positive": drn_net_out_positive, "riv_net_out_positive": riv_net_out_positive,
    }
    if not return_cells:
        return pd.DataFrame([row])

    residual = r_cell - q_gh
    diagonal = c_gh.copy()
    boundary_operand = c_gh * (np.abs(h_use) + np.abs(gh_stage))
    if q_drn is not None:
        residual -= q_drn
        flowing = mask_drn & (h_use > de)
        diagonal[flowing] += dc[flowing]
        boundary_operand[flowing] += dc[flowing] * (np.abs(h_use[flowing]) + np.abs(de[flowing]))
    if q_riv is not None:
        residual -= q_riv
        connected = mask_riv & (h_use > rb)
        diagonal[connected] += rc[connected]
        boundary_operand[mask_riv] += rc[mask_riv] * (
            np.abs(np.maximum(h_use[mask_riv], rb[mask_riv])) + np.abs(rs[mask_riv])
        )

    if component_budgets:
        labels, count = _free_components(cell_is_interior, cond_e, cond_s)
        row["component_count"] = int(count)
        if count > 1:
            # Sum each boundary separately. Opposing flows in the same cell
            # still both contribute to its gross external exchange.
            external_net = residual.copy()
            external_gross = np.abs(r_cell) + np.abs(q_gh)
            if q_drn is not None:
                external_gross += np.abs(q_drn)
            if q_riv is not None:
                external_gross += np.abs(q_riv)
            component_operand = boundary_operand.copy()
            for first, second, conductance, out_first, out_second in (
                (np.s_[:, :-1], np.s_[:, 1:], cond_e, q_int_to_bc_l, q_int_to_bc_r),
                (np.s_[:-1, :], np.s_[1:, :], cond_s, q_int_to_bc_t, q_int_to_bc_b),
            ):
                external_net[first] -= out_first
                external_net[second] -= out_second
                external_gross[first] += np.abs(out_first)
                external_gross[second] += np.abs(out_second)
                operand = conductance * (np.abs(h_use[first]) + np.abs(h_use[second]))
                component_operand[first] += np.where(bc[second], operand, 0.0)
                component_operand[second] += np.where(bc[first], operand, 0.0)
            row["component_labels"] = labels
            row["component_net"] = _component_sum(labels, external_net)
            row["component_gross"] = _component_sum(labels, external_gross)
            row["component_operand"] = _component_sum(labels, component_operand)

    residual[:, :-1] -= q_l_to_r
    residual[:, 1:] += q_l_to_r
    residual[:-1, :] -= q_t_to_b
    residual[1:, :] += q_t_to_b
    diagonal[:, :-1] += cond_e
    diagonal[:, 1:] += cond_e
    diagonal[:-1, :] += cond_s
    diagonal[1:, :] += cond_s
    # Only exchanges with external boundaries contribute to domain roundoff.
    boundary_scale = np.sum(boundary_operand, dtype=np.float64)
    for conductance, first, second in (
        (cond_e, (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        (cond_s, (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        boundary_scale += np.sum(
            conductance * (bc[first] ^ bc[second]) * (np.abs(h_use[first]) + np.abs(h_use[second])),
            dtype=np.float64,
        )
    residual[~cell_is_interior] = 0.0
    return row, residual, diagonal, boundary_scale


def evaluate_picard_transient_balance(
    *, model, head, previous, conductivity, bottom, top, min_sat, dt, sy, ss,
    storage_coeff, head_tolerance, relative_tolerance=1e-3,
):
    """Evaluate the returned head with the physical recharge, faces and storage."""
    from DARCY_WARP_PACKAGE.physics.storage_2d import exact_unconfined_storage_terms

    h = np.asarray(head, dtype=np.float64)
    old = np.asarray(previous, dtype=np.float64)
    free = (model.active_host != 0) & (model.bc_mask_host == 0)
    cap = np.maximum(top - bottom, min_sat) if top is not None else None
    saturation = np.clip(h - bottom, min_sat, cap)
    if storage_coeff is None:
        storage, _, _ = exact_unconfined_storage_terms(
            head_new=h, head_old=old, bottom=bottom, top=top,
            specific_yield=sy, specific_storage=ss, dt=dt,
        )
        storage_scale = sy + ss * np.maximum(top - bottom, 0.0)
    else:
        storage = np.asarray(storage_coeff, dtype=np.float64) * (h - old) / dt
        storage_scale = storage_coeff
    area = float(model.dx) ** 2
    storage_volume = np.where(free, storage * area, 0.0)
    budget, residual, diagonal, boundary_scale = compute_mass_balance_budget(
        T_field=conductivity * saturation, R_field=model.R_field_host, head=h,
        active=model.active_host, bc_mask=model.bc_mask_host, bc_values=model.bc_values_host,
        dx=model.dx, gh_mask=model.gh_mask_host, gh_head=model.gh_head_host,
        ghb_factor=model.ghb_factor_host, return_cells=True, component_budgets=True,
    )
    residual -= storage_volume
    # Use a positive storage sensitivity for the head-equivalent residual.
    storage_diagonal = np.zeros_like(h)
    np.divide(storage_volume, h - old, out=storage_diagonal, where=h != old)
    storage_diagonal[h == old] = np.broadcast_to(storage_scale, h.shape)[h == old] * area / dt
    diagonal += storage_diagonal
    equivalent = np.divide(residual, diagonal, out=np.zeros_like(residual), where=diagonal > 0.0)
    equivalent[(diagonal <= 0.0) & (residual != 0.0)] = np.inf
    head_rms = float(np.sqrt(np.mean(equivalent[free] ** 2))) if np.any(free) else 0.0
    flow_rms = float(np.sqrt(np.mean(residual[free] ** 2))) if np.any(free) else 0.0
    storage_in = float(np.sum(np.maximum(-storage_volume, 0.0)))
    storage_out = float(np.sum(np.maximum(storage_volume, 0.0)))
    total_in = budget["total_in"] + storage_in
    total_out = budget["total_out"] + storage_out
    throughflow = 0.5 * (total_in + total_out)
    imbalance = total_in - total_out
    storage_operand = np.where(free, storage_scale * (np.abs(h) + np.abs(old)) * (area / dt), 0.0)
    eps = np.finfo(np.asarray(head).dtype).eps
    floor = float(32.0 * eps * (throughflow + boundary_scale + np.sum(storage_operand)))
    discrepancy = 100.0 * imbalance / throughflow if throughflow else 0.0
    worst_component_pct = abs(discrepancy)
    components_ok = True
    if budget["component_count"] > 1:
        labels = budget["component_labels"]
        net = budget["component_net"] - _component_sum(labels, storage_volume)
        component_throughflow = 0.5 * (budget["component_gross"] + _component_sum(labels, np.abs(storage_volume)))
        component_floor = 32.0 * eps * (
            component_throughflow + budget["component_operand"] + _component_sum(labels, storage_operand)
        )
        components_ok = np.all(np.abs(net) <= np.maximum(component_floor, relative_tolerance * component_throughflow))
        ratios = np.divide(np.abs(net), component_throughflow, out=np.zeros_like(net), where=component_throughflow > 0)
        worst_component_pct = float(100.0 * np.max(ratios))
    return dict(accepted=bool(components_ok and np.isfinite([head_rms, flow_rms, imbalance, floor]).all()
                and head_rms <= head_tolerance and abs(imbalance) <= max(floor, relative_tolerance * throughflow)),
        head_residual_rms=head_rms, flow_residual_rms=flow_rms, total_in=total_in, total_out=total_out,
        storage_in=storage_in, storage_out=storage_out, in_minus_out=imbalance,
        percent_discrepancy=discrepancy, component_count=budget["component_count"],
        worst_component_percent_discrepancy=worst_component_pct,
        flow_roundoff_floor=floor, relative_tolerance=relative_tolerance,
    )


def add_exact_storage_to_budget(
    budget: pd.DataFrame,
    storage_flux: np.ndarray,
) -> pd.DataFrame:
    """Add exact nonlinear storage uptake/release to a steady package budget.

    Positive Stage-1 storage flux is water retained in the aquifer (an outflow
    from the flow balance); negative storage flux is release (an inflow).
    """
    if len(budget) != 1:
        raise ValueError("budget must contain exactly one aggregate row.")
    storage = np.asarray(storage_flux, dtype=np.float64)
    storage_out = float(np.sum(np.maximum(storage, 0.0)))
    storage_in = float(np.sum(np.maximum(-storage, 0.0)))
    result = budget.copy()
    result.loc[result.index[0], "storage_in"] = storage_in
    result.loc[result.index[0], "storage_out"] = storage_out
    total_in = float(result.iloc[0]["total_in"]) + storage_in
    total_out = float(result.iloc[0]["total_out"]) + storage_out
    imbalance = total_in - total_out
    denominator = 0.5 * (total_in + total_out)
    result.loc[result.index[0], "total_in"] = total_in
    result.loc[result.index[0], "total_out"] = total_out
    result.loc[result.index[0], "in_minus_out"] = imbalance
    result.loc[result.index[0], "percent_discrepancy"] = (
        0.0 if denominator == 0.0 else 100.0 * imbalance / denominator
    )
    result.loc[result.index[0], "throughflow"] = 0.5 * (total_in + total_out)
    result.loc[result.index[0], "imbalance_fraction"] = (
        0.0 if (total_in + total_out) == 0.0 else imbalance / (0.5 * (total_in + total_out))
    )
    return result
