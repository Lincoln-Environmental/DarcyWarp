# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import warp as wp

from DARCY_WARP_PACKAGE.model_builder import (
    _build_domain,
    _build_dirichlet_boundary_mask,
    _build_dem,
    _model_bottom,
    build_base_fields,
    build_truth_inputs,
)
from DARCY_WARP_PACKAGE.sparse_operator import (
    build_sparse_system_fd_like as _build_sparse_system_fd_like_impl,
)

import ctypes.util


logger = logging.getLogger(__name__)


def _probe_cuda_driver() -> bool:
    """
    Best-effort CUDA runtime probe used for diagnostics only.
    """
    lib_path = ctypes.util.find_library("cuda") or "libcuda.so.1"
    try:
        libcuda = ctypes.CDLL(lib_path)

        libcuda.cuInit.argtypes = [ctypes.c_uint]
        libcuda.cuInit.restype = ctypes.c_int

        libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        libcuda.cuDeviceGetCount.restype = ctypes.c_int

        res = libcuda.cuInit(0)
        if res != 0:
            logger.debug("CUDA driver probe failed: cuInit returned %s.", res)
            return False

        count = ctypes.c_int(0)
        res2 = libcuda.cuDeviceGetCount(ctypes.byref(count))
        if res2 == 0 and count.value > 0:
            logger.debug("CUDA driver reports %d device(s).", count.value)
            return True

        logger.debug(
            "CUDA driver present, but no devices available (res2=%s, count=%s).",
            res2,
            count.value,
        )
        return False
    except OSError as exc:
        logger.debug("Could not load CUDA driver library '%s': %s", lib_path, exc)
        return False


cuda_device_found = _probe_cuda_driver()

# Optional GHB helper: use if available
try:
    from legacy_code.model_builder import _build_ghb_boundary_mask
except ImportError:
    _build_ghb_boundary_mask = None




from DARCY_WARP_PACKAGE.config import NP_FLOAT, WP_FLOAT


def _chebyshev_relaxation_sequence(
    order: int,
    lambda_min: float,
    lambda_max: float,
) -> tuple[float, ...]:
    """
    Build Chebyshev semi-iteration relaxation factors for Richardson/Jacobi updates.

    Returns a tuple of omega_k values for:
        x_{k+1} = x_k + omega_k * M^{-1}(b - A x_k)
    over an eigenvalue interval [lambda_min, lambda_max] of M^{-1}A.
    """
    m = int(order)
    if m <= 0:
        return tuple()

    lam_hi = max(float(lambda_max), 1.0e-12)
    lam_lo = max(1.0e-12, min(float(lambda_min), 0.999999 * lam_hi))

    c = 0.5 * (lam_hi + lam_lo)
    d = 0.5 * (lam_hi - lam_lo)

    if d <= 0.0:
        return tuple(float(1.0 / c) for _ in range(m))

    out: list[float] = []
    for k in range(1, m + 1):
        theta_k = np.pi * (2.0 * float(k) - 1.0) / (2.0 * float(m))
        denom = c - d * float(np.cos(theta_k))
        if denom <= 1.0e-12:
            denom = 1.0e-12
        out.append(float(1.0 / denom))

    return tuple(out)


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
    gh_alpha: float = 1.0,
    aq_thickness: float = 1.0,
    case: str | None = None,
) -> pd.DataFrame:
    """
    MF6-like discrete budget consistent with apply_A_kernel + build_rhs_fd_like.

    Reports gross IN and gross OUT per term (MF6-style), not net-only:
      - RCHA: cell flux R*dx^2 over active, non-Dirichlet cells (split by sign).
      - CHD: sum of interface fluxes across interior<->Dirichlet connections (split by sign).
      - GHB: sum of cellwise exchange flux C*(h - h_ext) over GHB cells (split by sign).

    Sign conventions for the split:
      - Term IN is positive into the domain.
      - Term OUT is positive out of the domain.

    Percent discrepancy:
        100 * (TOTAL_IN - TOTAL_OUT) / (abs(TOTAL_IN) + abs(TOTAL_OUT))
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

    # Enforce solver constraints
    h_use = np.array(h, copy=True)
    h_use[~act] = 0.0
    h_use[bc] = bc_v[bc]

    # Only active, non-Dirichlet cells contribute recharge and GHB
    cell_is_interior = act & (~bc)

    # RCHA: compute cellwise flux and split by sign (MF6-like)
    r_cell = np.zeros((ny, nx), dtype=np.float64)
    r_cell[cell_is_interior] = R[cell_is_interior] * dx_f * dx_f
    rcha_in = float(np.sum(np.maximum(r_cell, 0.0)))
    rcha_out = float(np.sum(np.maximum(-r_cell, 0.0)))

    # CHD: compute interface fluxes across interior<->Dirichlet and split by sign
    chd_in = 0.0
    chd_out = 0.0

    # East-West interfaces
    act_L = act[:, :-1]
    act_R = act[:, 1:]
    conn_E = act_L & act_R

    T_L = T[:, :-1]
    T_R = T[:, 1:]
    denom_E = T_L + T_R

    cond_E = np.zeros((ny, nx - 1), dtype=np.float64)
    valid_E = conn_E & (T_L > 0.0) & (T_R > 0.0) & (denom_E > tiny)
    cond_E[valid_E] = 2.0 * T_L[valid_E] * T_R[valid_E] / denom_E[valid_E]

    dh_E = h_use[:, :-1] - h_use[:, 1:]
    q_L_to_R = cond_E * dh_E  # positive means leaving left cell into right cell

    bc_L = bc[:, :-1]
    bc_R = bc[:, 1:]

    # interior(left) -> bc(right)
    dom_L = conn_E & (~bc_L) & bc_R
    q_int_to_bc_L = np.where(dom_L, q_L_to_R, 0.0)
    chd_out += float(np.sum(np.maximum(q_int_to_bc_L, 0.0)))
    chd_in += float(np.sum(np.maximum(-q_int_to_bc_L, 0.0)))

    # interior(right) -> bc(left)
    dom_R = conn_E & bc_L & (~bc_R)
    q_int_to_bc_R = np.where(dom_R, -q_L_to_R, 0.0)
    chd_out += float(np.sum(np.maximum(q_int_to_bc_R, 0.0)))
    chd_in += float(np.sum(np.maximum(-q_int_to_bc_R, 0.0)))

    # North-South interfaces
    act_T = act[:-1, :]
    act_B = act[1:, :]
    conn_S = act_T & act_B

    T_T = T[:-1, :]
    T_B = T[1:, :]
    denom_S = T_T + T_B

    cond_S = np.zeros((ny - 1, nx), dtype=np.float64)
    valid_S = conn_S & (T_T > 0.0) & (T_B > 0.0) & (denom_S > tiny)
    cond_S[valid_S] = 2.0 * T_T[valid_S] * T_B[valid_S] / denom_S[valid_S]

    dh_S = h_use[:-1, :] - h_use[1:, :]
    q_T_to_B = cond_S * dh_S  # positive means leaving top cell into bottom cell

    bc_T = bc[:-1, :]
    bc_B = bc[1:, :]

    # interior(top) -> bc(bottom)
    dom_T = conn_S & (~bc_T) & bc_B
    q_int_to_bc_T = np.where(dom_T, q_T_to_B, 0.0)
    chd_out += float(np.sum(np.maximum(q_int_to_bc_T, 0.0)))
    chd_in += float(np.sum(np.maximum(-q_int_to_bc_T, 0.0)))

    # interior(bottom) -> bc(top)
    dom_B = conn_S & bc_T & (~bc_B)
    q_int_to_bc_B = np.where(dom_B, -q_T_to_B, 0.0)
    chd_out += float(np.sum(np.maximum(q_int_to_bc_B, 0.0)))
    chd_in += float(np.sum(np.maximum(-q_int_to_bc_B, 0.0)))

    # Net (optional, for sanity checks)
    chd_net_out_positive = chd_out - chd_in

    # GHB: cellwise exchange flux and split by sign
    ghb_in = 0.0
    ghb_out = 0.0
    ghb_net_out_positive = 0.0

    if (gh_mask is not None) and (gh_head is not None) and (gh_width is not None):
        ghm = np.asarray(gh_mask, dtype=np.int32) != 0
        ghe = np.asarray(gh_head, dtype=np.float64)
        ghw = np.asarray(gh_width, dtype=np.float64)

        if ghe.shape != (ny, nx) or ghw.shape != (ny, nx):
            raise ValueError("gh_head/gh_width shape mismatch")
        if float(aq_thickness) <= 0.0:
            raise ValueError("aq_thickness must be positive")

        Cgh = (float(gh_alpha) * T / float(aq_thickness) * ghw * dx_f).astype(np.float64)

        mask_gh = (
            ghm
            & cell_is_interior
            & np.isfinite(ghw)
            & (ghw > 0.0)
            & np.isfinite(ghe)
            & np.isfinite(h_use)
        )

        q_gh = np.zeros((ny, nx), dtype=np.float64)
        q_gh[mask_gh] = Cgh[mask_gh] * (h_use[mask_gh] - ghe[mask_gh])  # + is outflow

        ghb_out = float(np.sum(np.maximum(q_gh, 0.0)))
        ghb_in = float(np.sum(np.maximum(-q_gh, 0.0)))
        ghb_net_out_positive = ghb_out - ghb_in

    total_in = rcha_in + chd_in + ghb_in
    total_out = rcha_out + chd_out + ghb_out
    in_minus_out = total_in - total_out

    denom = abs(total_in) + abs(total_out)
    percent_discrepancy = 0.0 if denom == 0.0 else 100.0 * in_minus_out / denom

    # Optional throughflow (handy for tables)
    throughflow = 0.5 * (total_in + total_out)
    imbalance_fraction = 0.0 if throughflow == 0.0 else in_minus_out / throughflow

    row = {
        "case": "" if case is None else str(case),
        "rcha_in": rcha_in,
        "rcha_out": rcha_out,
        "chd_in": chd_in,
        "chd_out": chd_out,
        "ghb_in": ghb_in,
        "ghb_out": ghb_out,
        "total_in": total_in,
        "total_out": total_out,
        "in_minus_out": in_minus_out,
        "percent_discrepancy": percent_discrepancy,
        "throughflow": throughflow,
        "imbalance_fraction": imbalance_fraction,
        # nets for sanity checks
        "chd_net_out_positive": chd_net_out_positive,
        "ghb_net_out_positive": ghb_net_out_positive,
    }

    return pd.DataFrame([row])


def build_coarse_level_from_fine(
    T_f: np.ndarray,
    R_f: np.ndarray,
    active_f: np.ndarray,
    bc_mask_f: np.ndarray,
    bc_values_f: np.ndarray,
    gh_mask_f: np.ndarray,
    gh_head_f: np.ndarray,
    gh_width_f: np.ndarray,
):
    ny_f, nx_f = T_f.shape
    nx_c = (nx_f + 1) // 2
    ny_c = (ny_f + 1) // 2

    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    def _pad(arr, fill_value=0):
        return np.pad(
            np.asarray(arr),
            ((0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=fill_value,
        )

    # Block counts for correct means on odd-sized grids
    valid = np.ones((ny_f, nx_f), dtype=np.float64)
    count = _pad(valid, fill_value=0.0).reshape(ny_c, 2, nx_c, 2).sum(axis=(1, 3))
    count_safe = np.maximum(count, 1.0)

    # Masks: any in 2x2 block
    active_c = (
        _pad(active_f, fill_value=0)
        .reshape(ny_c, 2, nx_c, 2)
        .max(axis=(1, 3))
        .astype(np.int32, copy=False)
    )
    bc_mask_c = (
        _pad(bc_mask_f, fill_value=0)
        .reshape(ny_c, 2, nx_c, 2)
        .max(axis=(1, 3))
        .astype(np.int32, copy=False)
    )
    gh_mask_c = (
        _pad(gh_mask_f, fill_value=0)
        .reshape(ny_c, 2, nx_c, 2)
        .max(axis=(1, 3))
        .astype(np.int32, copy=False)
    )

    # Means over valid cells in the block (not just active)
    T_sum = _pad(T_f, fill_value=0.0).astype(np.float64, copy=False).reshape(ny_c, 2, nx_c, 2).sum(axis=(1, 3))
    R_sum = _pad(R_f, fill_value=0.0).astype(np.float64, copy=False).reshape(ny_c, 2, nx_c, 2).sum(axis=(1, 3))
    T_c = (T_sum / count_safe).astype(NP_FLOAT, copy=False)
    R_c = (R_sum / count_safe).astype(NP_FLOAT, copy=False)

    # Zero out inactive coarse cells (matches loop behavior)
    inactive = active_c == 0
    if np.any(inactive):
        T_c[inactive] = NP_FLOAT(0.0)
        R_c[inactive] = NP_FLOAT(0.0)

    # GHB width: mean over block if any GHB present, else 0
    gh_width_sum = (
        _pad(gh_width_f, fill_value=0.0).astype(np.float64, copy=False).reshape(ny_c, 2, nx_c, 2).sum(axis=(1, 3))
    )
    gh_width_c = (gh_width_sum / count_safe).astype(NP_FLOAT, copy=False)
    gh_width_c[gh_mask_c == 0] = NP_FLOAT(0.0)

    bc_values_c = np.zeros((ny_c, nx_c), dtype=NP_FLOAT)
    gh_head_c = np.zeros((ny_c, nx_c), dtype=NP_FLOAT)

    return (
        T_c,
        R_c,
        active_c,
        bc_mask_c,
        bc_values_c,
        gh_mask_c,
        gh_head_c,
        gh_width_c,
    )


def build_sparse_system_fd_like(
    T_field: np.ndarray,
    R_field: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    dx: float,
    gh_mask: np.ndarray | None = None,
    gh_head: np.ndarray | None = None,
    gh_width: np.ndarray | None = None,
    gh_alpha: float = 1.0,
    aq_thickness: float = 1.0,
):
    return _build_sparse_system_fd_like_impl(
        T_field=T_field,
        R_field=R_field,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        dx=dx,
        gh_mask=gh_mask,
        gh_head=gh_head,
        gh_width=gh_width,
        gh_alpha=gh_alpha,
        aq_thickness=aq_thickness,
    )


def _coarsen_mean_edge_2x2(field_f: np.ndarray) -> np.ndarray:
    """
    2:1 coarsening by 2x2 arithmetic mean with edge padding for odd sizes.
    """
    arr_f = np.asarray(field_f, dtype=NP_FLOAT)
    ny_f, nx_f = arr_f.shape
    ny_c = (ny_f + 1) // 2
    nx_c = (nx_f + 1) // 2

    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)
    arr_p = np.pad(arr_f, ((0, pad_y), (0, pad_x)), mode="edge")

    arr_c = arr_p.reshape(ny_c, 2, nx_c, 2).mean(axis=(1, 3), dtype=np.float64)
    return arr_c.astype(NP_FLOAT, copy=False)


def _normalize_stencil_mode(stencil: str) -> str:
    mode = str(stencil).strip().lower().replace("_", "-")
    if mode in {"5", "5pt", "5-point", "2d"}:
        return "5-point"
    if mode in {"7", "7pt", "7-point", "3d", "multilayer"}:
        return "7-point"
    raise ValueError("stencil must be one of {'5-point', '7-point'}.")


def build_7point_face_conductance_from_k(
    kx_field: np.ndarray,
    ky_field: np.ndarray,
    kz_field: np.ndarray,
    active: np.ndarray,
    dx: float,
    dy: float | None = None,
    dz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D 7-point face conductances from cell-centered K fields.

    Returns (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m), each shape (nz, ny, nx),
    where e.g. tx_p[k,j,i] couples cell (k,j,i) to (k,j,i+1).
    """
    if float(dx) <= 0.0:
        raise ValueError("dx must be > 0.")
    dy_f = float(dx) if dy is None else float(dy)
    dz_f = float(dz)
    if dy_f <= 0.0 or dz_f <= 0.0:
        raise ValueError("dy and dz must be > 0.")

    Kx = np.asarray(kx_field, dtype=np.float64)
    Ky = np.asarray(ky_field, dtype=np.float64)
    Kz = np.asarray(kz_field, dtype=np.float64)
    act = np.asarray(active, dtype=np.int32) != 0

    if Kx.ndim != 3 or Ky.ndim != 3 or Kz.ndim != 3:
        raise ValueError("kx_field, ky_field, kz_field must all be 3D arrays (nz, ny, nx).")
    if Ky.shape != Kx.shape or Kz.shape != Kx.shape:
        raise ValueError("kx_field, ky_field, kz_field must have matching shapes.")
    if act.shape != Kx.shape:
        raise ValueError(f"active shape {act.shape} must match K shape {Kx.shape}.")
    if not np.all(np.isfinite(Kx)) or not np.all(np.isfinite(Ky)) or not np.all(np.isfinite(Kz)):
        raise ValueError("K fields must be finite.")
    if np.any(Kx < 0.0) or np.any(Ky < 0.0) or np.any(Kz < 0.0):
        raise ValueError("K fields must be >= 0.")

    nz, ny, nx = Kx.shape
    tiny = 1.0e-12

    tx_p = np.zeros((nz, ny, nx), dtype=np.float64)
    tx_m = np.zeros((nz, ny, nx), dtype=np.float64)
    ty_p = np.zeros((nz, ny, nx), dtype=np.float64)
    ty_m = np.zeros((nz, ny, nx), dtype=np.float64)
    tz_p = np.zeros((nz, ny, nx), dtype=np.float64)
    tz_m = np.zeros((nz, ny, nx), dtype=np.float64)

    fac_x = dy_f * dz_f / float(dx)
    fac_y = float(dx) * dz_f / dy_f
    fac_z = float(dx) * dy_f / dz_f

    if nx > 1:
        KL = Kx[:, :, :-1]
        KR = Kx[:, :, 1:]
        denom = KL + KR
        conn = act[:, :, :-1] & act[:, :, 1:] & (KL > 0.0) & (KR > 0.0) & (denom > tiny)
        cond = np.zeros_like(denom, dtype=np.float64)
        cond[conn] = 2.0 * KL[conn] * KR[conn] / denom[conn] * fac_x
        tx_p[:, :, :-1] = cond
        tx_m[:, :, 1:] = cond

    if ny > 1:
        KT = Ky[:, :-1, :]
        KB = Ky[:, 1:, :]
        denom = KT + KB
        conn = act[:, :-1, :] & act[:, 1:, :] & (KT > 0.0) & (KB > 0.0) & (denom > tiny)
        cond = np.zeros_like(denom, dtype=np.float64)
        cond[conn] = 2.0 * KT[conn] * KB[conn] / denom[conn] * fac_y
        ty_p[:, :-1, :] = cond
        ty_m[:, 1:, :] = cond

    if nz > 1:
        KU = Kz[:-1, :, :]
        KD = Kz[1:, :, :]
        denom = KU + KD
        conn = act[:-1, :, :] & act[1:, :, :] & (KU > 0.0) & (KD > 0.0) & (denom > tiny)
        cond = np.zeros_like(denom, dtype=np.float64)
        cond[conn] = 2.0 * KU[conn] * KD[conn] / denom[conn] * fac_z
        tz_p[:-1, :, :] = cond
        tz_m[1:, :, :] = cond

    return (
        tx_p.astype(NP_FLOAT, copy=False),
        tx_m.astype(NP_FLOAT, copy=False),
        ty_p.astype(NP_FLOAT, copy=False),
        ty_m.astype(NP_FLOAT, copy=False),
        tz_p.astype(NP_FLOAT, copy=False),
        tz_m.astype(NP_FLOAT, copy=False),
    )


def build_diag_preconditioner_7point(
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    storage_diag: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build diagonal Jacobi preconditioner for 3D 7-point operator.
    """
    txp = np.asarray(tx_p, dtype=np.float64)
    txm = np.asarray(tx_m, dtype=np.float64)
    typ = np.asarray(ty_p, dtype=np.float64)
    tym = np.asarray(ty_m, dtype=np.float64)
    tzp = np.asarray(tz_p, dtype=np.float64)
    tzm = np.asarray(tz_m, dtype=np.float64)
    act = np.asarray(active, dtype=np.int32)
    bc = np.asarray(bc_mask, dtype=np.int32)

    shape = txp.shape
    if txp.ndim != 3:
        raise ValueError("7-point arrays must be 3D with shape (nz, ny, nx).")
    for name, arr in (
        ("tx_m", txm),
        ("ty_p", typ),
        ("ty_m", tym),
        ("tz_p", tzp),
        ("tz_m", tzm),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} expected {shape}")

    if act.shape != shape:
        raise ValueError(f"active shape {act.shape} expected {shape}")
    if bc.shape != shape:
        raise ValueError(f"bc_mask shape {bc.shape} expected {shape}")

    for name, arr in (
        ("tx_p", txp),
        ("tx_m", txm),
        ("ty_p", typ),
        ("ty_m", tym),
        ("tz_p", tzp),
        ("tz_m", tzm),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be >= 0")

    free = (act != 0) & (bc == 0)
    diag = txp + txm + typ + tym + tzp + tzm

    if storage_diag is not None:
        sdiag = np.asarray(storage_diag, dtype=np.float64)
        if sdiag.shape != shape:
            raise ValueError(f"storage_diag shape {sdiag.shape} expected {shape}")
        if not np.all(np.isfinite(sdiag)):
            raise ValueError("storage_diag contains non-finite values")
        if np.any(sdiag < 0.0):
            raise ValueError("storage_diag must be >= 0")
        diag[free] += sdiag[free]

    tiny = 1.0e-12
    M_inv = np.ones(shape, dtype=np.float64)
    valid = free & np.isfinite(diag) & (diag > tiny)
    M_inv[valid] = 1.0 / diag[valid]
    M_inv[~free] = 1.0
    return M_inv.astype(NP_FLOAT, copy=False)



def _coarsen_mean_edge_2x2x2(field_f: np.ndarray) -> np.ndarray:
    """
    2:1 coarsening by 2x2x2 arithmetic mean with edge padding for odd sizes.
    """
    arr_f = np.asarray(field_f, dtype=NP_FLOAT)
    if arr_f.ndim != 3:
        raise ValueError("Expected 3D array for 2x2x2 coarsening.")

    nz_f, ny_f, nx_f = arr_f.shape
    nz_c = (nz_f + 1) // 2
    ny_c = (ny_f + 1) // 2
    nx_c = (nx_f + 1) // 2

    pad_z = int(2 * nz_c - nz_f)
    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    arr_p = np.pad(arr_f, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="edge")
    arr_c = arr_p.reshape(nz_c, 2, ny_c, 2, nx_c, 2).mean(axis=(1, 3, 5), dtype=np.float64)
    return arr_c.astype(NP_FLOAT, copy=False)


def _coarsen_max_edge_2x2x2(mask_f: np.ndarray) -> np.ndarray:
    """
    2:1 coarsening by 2x2x2 max pooling with edge padding for odd sizes.
    """
    arr_f = np.asarray(mask_f, dtype=np.int32)
    if arr_f.ndim != 3:
        raise ValueError("Expected 3D array for 2x2x2 coarsening.")

    nz_f, ny_f, nx_f = arr_f.shape
    nz_c = (nz_f + 1) // 2
    ny_c = (ny_f + 1) // 2
    nx_c = (nx_f + 1) // 2

    pad_z = int(2 * nz_c - nz_f)
    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    arr_p = np.pad(arr_f, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="edge")
    arr_c = arr_p.reshape(nz_c, 2, ny_c, 2, nx_c, 2).max(axis=(1, 3, 5))
    return arr_c.astype(np.int32, copy=False)


def _prepare_7point_transient_terms(
    rhs: np.ndarray,
    storage_diag: np.ndarray | None,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    transient: bool,
    storage_coeff: np.ndarray | float | None,
    dt: float | None,
    head_prev: np.ndarray | None,
    initial_head: np.ndarray | None,
    dx: float,
    dy: float | None,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """
    Prepare RHS and storage diagonal for optional confined transient 7-point solve.
    """
    b = np.asarray(rhs, dtype=NP_FLOAT).copy()
    act = np.asarray(active, dtype=np.int32)
    bcm = np.asarray(bc_mask, dtype=np.int32)
    bcv = np.asarray(bc_values, dtype=NP_FLOAT)

    shape = b.shape
    free = (act != 0) & (bcm == 0)

    if storage_diag is None:
        sdiag = np.zeros(shape, dtype=NP_FLOAT)
    else:
        sdiag = np.asarray(storage_diag, dtype=NP_FLOAT).copy()
        if sdiag.shape != shape:
            raise ValueError(f"storage_diag shape {sdiag.shape} expected {shape}")
        if not np.all(np.isfinite(sdiag)):
            raise ValueError("storage_diag must be finite.")
        if np.any(sdiag < NP_FLOAT(0.0)):
            raise ValueError("storage_diag must be >= 0.")

    sdiag[~free] = NP_FLOAT(0.0)

    if not bool(transient):
        return b, sdiag, None, float("nan")

    dt_f = float(dt) if dt is not None else float("nan")
    if not np.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError("transient=True requires dt > 0.")
    if storage_coeff is None:
        raise ValueError("transient=True requires storage_coeff.")

    dx_f = float(dx)
    dy_f = float(dx) if dy is None else float(dy)
    dz_f = float(dz)
    if dx_f <= 0.0 or dy_f <= 0.0 or dz_f <= 0.0:
        raise ValueError("dx, dy, dz must be positive for transient terms.")
    vol = np.float64(dx_f * dy_f * dz_f)

    s_in = np.asarray(storage_coeff, dtype=NP_FLOAT)
    if s_in.shape == ():
        Scoeff = np.full(shape, NP_FLOAT(s_in.reshape(()).item()), dtype=NP_FLOAT)
    else:
        if s_in.shape != shape:
            raise ValueError(f"storage_coeff shape {s_in.shape} expected {shape}")
        Scoeff = np.asarray(s_in, dtype=NP_FLOAT)

    if not np.all(np.isfinite(Scoeff)):
        raise ValueError("storage_coeff must contain finite values.")
    if np.any(Scoeff < NP_FLOAT(0.0)):
        raise ValueError("storage_coeff must be >= 0.")

    sdiag_add = (
        Scoeff.astype(np.float64, copy=False) * vol / np.float64(dt_f)
    ).astype(NP_FLOAT, copy=False)
    sdiag_add[~free] = NP_FLOAT(0.0)

    if head_prev is not None:
        h_prev = np.asarray(head_prev, dtype=NP_FLOAT).copy()
        if h_prev.shape != shape:
            raise ValueError(f"head_prev shape {h_prev.shape} expected {shape}")
    elif initial_head is not None:
        h_prev = np.asarray(initial_head, dtype=NP_FLOAT).copy()
        if h_prev.shape != shape:
            raise ValueError(f"initial_head shape {h_prev.shape} expected {shape}")
    else:
        h_prev = np.zeros(shape, dtype=NP_FLOAT)

    h_prev[bcm != 0] = bcv[bcm != 0]
    h_prev[act == 0] = NP_FLOAT(0.0)
    if not np.all(np.isfinite(h_prev)):
        raise ValueError("head_prev contains non-finite values.")

    b[free] = (
        b[free].astype(np.float64, copy=False)
        + sdiag_add[free].astype(np.float64, copy=False) * h_prev[free].astype(np.float64, copy=False)
    ).astype(NP_FLOAT, copy=False)
    sdiag[free] = (
        sdiag[free].astype(np.float64, copy=False) + sdiag_add[free].astype(np.float64, copy=False)
    ).astype(NP_FLOAT, copy=False)

    return b, sdiag, h_prev, float(dt_f)



def _solve_chebyshev_7point_3d_linear(
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
    rhs: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    initial_head: np.ndarray | None = None,
    storage_diag: np.ndarray | None = None,
    max_iter: int = 80,
    cheby_order: int = 2,
    cheby_lambda_min: float = 0.05,
    cheby_lambda_max: float = 1.95,
    rel_tol: float = 5.0e-7,
    abs_tol_min: float = 5.0e-7,
    transient: bool = False,
    storage_coeff: np.ndarray | float | None = None,
    dt: float | None = None,
    head_prev: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    dz: float = 1.0,
    device: str = "cuda:0",
    return_info: bool = True,
):
    from DARCY_WARP_PACKAGE.solvers_3d import (
        _solve_chebyshev_7point_3d_linear as _solve_chebyshev_7point_3d_linear_impl,
    )

    return _solve_chebyshev_7point_3d_linear_impl(
        tx_p=tx_p,
        tx_m=tx_m,
        ty_p=ty_p,
        ty_m=ty_m,
        tz_p=tz_p,
        tz_m=tz_m,
        rhs=rhs,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        initial_head=initial_head,
        storage_diag=storage_diag,
        max_iter=max_iter,
        cheby_order=cheby_order,
        cheby_lambda_min=cheby_lambda_min,
        cheby_lambda_max=cheby_lambda_max,
        rel_tol=rel_tol,
        abs_tol_min=abs_tol_min,
        transient=transient,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        dx=dx,
        dy=dy,
        dz=dz,
        device=device,
        return_info=return_info,
    )


def solve_chebyshev_7point_3d(
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
    rhs: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    initial_head: np.ndarray | None = None,
    storage_diag: np.ndarray | None = None,
    max_iter: int = 80,
    cheby_order: int = 2,
    cheby_lambda_min: float = 0.05,
    cheby_lambda_max: float = 1.95,
    rel_tol: float = 5.0e-7,
    abs_tol_min: float = 5.0e-7,
    transient: bool = False,
    storage_coeff: np.ndarray | float | None = None,
    dt: float | None = None,
    head_prev: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    dz: float = 1.0,
    unconfined: bool = False,
    kx_field: np.ndarray | None = None,
    ky_field: np.ndarray | None = None,
    kz_field: np.ndarray | None = None,
    zbot_field: np.ndarray | None = None,
    unconfined_min_sat: float = 0.1,
    unconfined_max_picard_iter: int = 8,
    unconfined_relax: float = 0.7,
    unconfined_head_tol: float = 1.0e-3,
    device: str = "cuda:0",
    return_info: bool = True,
):
    """
    Compatibility wrapper around the extracted 3D Chebyshev solver module.
    """
    from DARCY_WARP_PACKAGE.solvers_3d import (
        solve_chebyshev_7point_3d as solve_chebyshev_7point_3d_impl,
    )

    return solve_chebyshev_7point_3d_impl(
        tx_p=tx_p,
        tx_m=tx_m,
        ty_p=ty_p,
        ty_m=ty_m,
        tz_p=tz_p,
        tz_m=tz_m,
        rhs=rhs,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        initial_head=initial_head,
        storage_diag=storage_diag,
        max_iter=max_iter,
        cheby_order=cheby_order,
        cheby_lambda_min=cheby_lambda_min,
        cheby_lambda_max=cheby_lambda_max,
        rel_tol=rel_tol,
        abs_tol_min=abs_tol_min,
        transient=transient,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        dx=dx,
        dy=dy,
        dz=dz,
        unconfined=unconfined,
        kx_field=kx_field,
        ky_field=ky_field,
        kz_field=kz_field,
        zbot_field=zbot_field,
        unconfined_min_sat=unconfined_min_sat,
        unconfined_max_picard_iter=unconfined_max_picard_iter,
        unconfined_relax=unconfined_relax,
        unconfined_head_tol=unconfined_head_tol,
        device=device,
        return_info=return_info,
    )


def solve_multigrid_kcycle_7point_3d(
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
    rhs: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    initial_head: np.ndarray | None = None,
    storage_diag: np.ndarray | None = None,
    max_cycles: int = 20,
    nu_pre: int = 2,
    nu_post: int = 2,
    nu_coarse: int = 20,
    max_levels: int = 5,
    min_coarse_n: int = 2,
    smoother: str = "chebyshev",
    omega: float = 0.8,
    cheby_lambda_min: float = 0.05,
    cheby_lambda_max: float = 1.95,
    rel_tol: float = 5.0e-7,
    abs_tol_min: float = 5.0e-7,
    check_every_no: int = 5,
    dh_rms_tol: float | None = 1.0e-4,
    dh_max_tol: float | None = None,
    dh_max_factor: float = 5.0,
    transient: bool = False,
    storage_coeff: np.ndarray | float | None = None,
    dt: float | None = None,
    head_prev: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    dz: float = 1.0,
    unconfined: bool = False,
    kx_field: np.ndarray | None = None,
    ky_field: np.ndarray | None = None,
    kz_field: np.ndarray | None = None,
    zbot_field: np.ndarray | None = None,
    unconfined_min_sat: float = 0.1,
    unconfined_max_picard_iter: int = 8,
    unconfined_relax: float = 0.7,
    unconfined_head_tol: float = 1.0e-3,
    device: str = "cuda:0",
    return_info: bool = True,
):
    """
    Compatibility wrapper around the extracted 3D multigrid solver module.
    """
    from DARCY_WARP_PACKAGE.solvers_3d import (
        solve_multigrid_kcycle_7point_3d as solve_multigrid_kcycle_7point_3d_impl,
    )

    return solve_multigrid_kcycle_7point_3d_impl(
        tx_p=tx_p,
        tx_m=tx_m,
        ty_p=ty_p,
        ty_m=ty_m,
        tz_p=tz_p,
        tz_m=tz_m,
        rhs=rhs,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        initial_head=initial_head,
        storage_diag=storage_diag,
        max_cycles=max_cycles,
        nu_pre=nu_pre,
        nu_post=nu_post,
        nu_coarse=nu_coarse,
        max_levels=max_levels,
        min_coarse_n=min_coarse_n,
        smoother=smoother,
        omega=omega,
        cheby_lambda_min=cheby_lambda_min,
        cheby_lambda_max=cheby_lambda_max,
        rel_tol=rel_tol,
        abs_tol_min=abs_tol_min,
        check_every_no=check_every_no,
        dh_rms_tol=dh_rms_tol,
        dh_max_tol=dh_max_tol,
        dh_max_factor=dh_max_factor,
        transient=transient,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        dx=dx,
        dy=dy,
        dz=dz,
        unconfined=unconfined,
        kx_field=kx_field,
        ky_field=ky_field,
        kz_field=kz_field,
        zbot_field=zbot_field,
        unconfined_min_sat=unconfined_min_sat,
        unconfined_max_picard_iter=unconfined_max_picard_iter,
        unconfined_relax=unconfined_relax,
        unconfined_head_tol=unconfined_head_tol,
        device=device,
        return_info=return_info,
    )


@wp.kernel
def jacobi_applyA_fused_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    x_in: wp.array(dtype=WP_FLOAT, ndim=2),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    omega: float,
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    aq_thickness: float,
    x_out: wp.array(dtype=WP_FLOAT, ndim=2),
):
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        x_out[j, i] = WP_FLOAT(0.0)
        return

    if bc_mask[j, i] != 0:
        x_out[j, i] = bc_values[j, i]
        return

    T_c = wp.float64(T_field[j, i])
    hC = wp.float64(x_in[j, i])

    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)

    tiny = wp.float64(1.0e-12)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        width = wp.float64(gh_width[j, i])
        if width > wp.float64(0.0) and not wp.isnan(width):
            C_gh = (
                wp.float64(gh_alpha)
                * T_c
                / wp.float64(aq_thickness)
                * width
                * wp.float64(dx)
            )

    Sdiag = wp.float64(storage_diag[j, i])
    if Sdiag < wp.float64(0.0):
        Sdiag = wp.float64(0.0)
    sum_T = T_e + T_w + T_n + T_s + C_gh + Sdiag

    Ah = WP_FLOAT(0.0)
    if sum_T < tiny:
        Ah = WP_FLOAT(hC)
    else:
        val = sum_T * hC
        if T_e > wp.float64(0.0):
            val = val - T_e * wp.float64(x_in[j, i + 1])
        if T_w > wp.float64(0.0):
            val = val - T_w * wp.float64(x_in[j, i - 1])
        if T_n > wp.float64(0.0):
            val = val - T_n * wp.float64(x_in[j - 1, i])
        if T_s > wp.float64(0.0):
            val = val - T_s * wp.float64(x_in[j + 1, i])
        Ah = WP_FLOAT(val)

    r_ij = b[j, i] - Ah
    x_out[j, i] = WP_FLOAT(hC) + WP_FLOAT(omega) * M_inv[j, i] * r_ij

@wp.kernel
def restrict_blockavg_kernel(
        r_f: wp.array2d(dtype=WP_FLOAT),
        active_f: wp.array2d(dtype=wp.int32),
        bc_mask_f: wp.array2d(dtype=wp.int32),
        b_c: wp.array2d(dtype=WP_FLOAT),
        nx_f: int,
        ny_f: int,
        nx_c: int,
        ny_c: int,
):
    jc, ic = wp.tid()
    if jc >= ny_c or ic >= nx_c:
        return

    j0 = 2 * jc
    i0 = 2 * ic

    s = WP_FLOAT(0.0)
    n = WP_FLOAT(0.0)

    # helper inline pattern repeated 4 times, no lambdas
    if j0 < ny_f and i0 < nx_f:
        if active_f[j0, i0] != 0 and bc_mask_f[j0, i0] == 0:
            s = s + r_f[j0, i0]
            n = n + WP_FLOAT(1.0)

    if j0 < ny_f and (i0 + 1) < nx_f:
        if active_f[j0, i0 + 1] != 0 and bc_mask_f[j0, i0 + 1] == 0:
            s = s + r_f[j0, i0 + 1]
            n = n + WP_FLOAT(1.0)

    if (j0 + 1) < ny_f and i0 < nx_f:
        if active_f[j0 + 1, i0] != 0 and bc_mask_f[j0 + 1, i0] == 0:
            s = s + r_f[j0 + 1, i0]
            n = n + WP_FLOAT(1.0)

    if (j0 + 1) < ny_f and (i0 + 1) < nx_f:
        if active_f[j0 + 1, i0 + 1] != 0 and bc_mask_f[j0 + 1, i0 + 1] == 0:
            s = s + r_f[j0 + 1, i0 + 1]
            n = n + WP_FLOAT(1.0)

    if n > WP_FLOAT(0.0):
        b_c[jc, ic] = s / n
    else:
        b_c[jc, ic] = WP_FLOAT(0.0)


@wp.kernel
def prolong_bilinear_any_kernel(
    x_c: wp.array2d(dtype=WP_FLOAT),
    e_f: wp.array2d(dtype=WP_FLOAT),
    nx_f: int,
    ny_f: int,
    nx_c: int,
    ny_c: int,
):
    j, i = wp.tid()

    if j >= ny_f or i >= nx_f:
        return

    jc = j // 2
    ic = i // 2

    # fractional offsets: 0 for even index, 0.5 for odd
    fy = WP_FLOAT(0.0)
    fx = WP_FLOAT(0.0)
    if (j & 1) == 1:
        fy = WP_FLOAT(0.5)
    if (i & 1) == 1:
        fx = WP_FLOAT(0.5)

    # Neighbor coarse indices with clamping
    ic1 = ic + 1
    jc1 = jc + 1
    if ic1 >= nx_c:
        ic1 = nx_c - 1
    if jc1 >= ny_c:
        jc1 = ny_c - 1

    v00 = x_c[jc, ic]
    v10 = x_c[jc, ic1]
    v01 = x_c[jc1, ic]
    v11 = x_c[jc1, ic1]

    # Bilinear interpolation
    one = WP_FLOAT(1.0)
    w00 = (one - fx) * (one - fy)
    w10 = fx * (one - fy)
    w01 = (one - fx) * fy
    w11 = fx * fy

    e_f[j, i] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11


@wp.kernel
def add_correction_kernel(
    x_f: wp.array(dtype=WP_FLOAT, ndim=2),
    e_f: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        x_f[j, i] = WP_FLOAT(0.0)
        return

    if bc_mask[j, i] != 0:
        x_f[j, i] = bc_values[j, i]
        return

    x_f[j, i] = x_f[j, i] + e_f[j, i]


@wp.kernel
def zero_field_kernel(
    a: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    """
    :param a: field to zero, shape (ny, nx)
    :param nx: number of columns
    :param ny: number of rows
    """
    j, i = wp.tid()  # 2D thread index (row, col)

    if j >= ny or i >= nx:
        return

    a[j, i] = WP_FLOAT(0.0)

@wp.kernel
def compute_head_residual_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    aq_thickness: float,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = WP_FLOAT(0.0)
        return

    tiny = wp.float64(1.0e-12)

    T_c = wp.float64(T_field[j, i])
    hC = wp.float64(x[j, i])

    hE = wp.float64(0.0)
    hW = wp.float64(0.0)
    hN = wp.float64(0.0)
    hS = wp.float64(0.0)

    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        hE = wp.float64(x[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        hW = wp.float64(x[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        hN = wp.float64(x[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        hS = wp.float64(x[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        width = wp.float64(gh_width[j, i])
        if width > wp.float64(0.0) and not wp.isnan(width):
            C_gh = (
                wp.float64(gh_alpha)
                * T_c
                / wp.float64(aq_thickness)
                * width
                * wp.float64(dx)
            )

    Sdiag = wp.float64(storage_diag[j, i])
    if Sdiag < wp.float64(0.0):
        Sdiag = wp.float64(0.0)
    diagA = T_e + T_w + T_n + T_s + C_gh + Sdiag

    # Apply A in the same flux-integrated form as your operator
    Ax64 = wp.float64(0.0)
    if diagA < tiny:
        # Identity-row semantics: Ax = hC
        Ax64 = hC
        rh64 = wp.float64(b[j, i]) - Ax64  # metres if b was built consistently for identity rows
    else:
        Ax64 = diagA * hC
        if T_e > wp.float64(0.0):
            Ax64 = Ax64 - T_e * hE
        if T_w > wp.float64(0.0):
            Ax64 = Ax64 - T_w * hW
        if T_n > wp.float64(0.0):
            Ax64 = Ax64 - T_n * hN
        if T_s > wp.float64(0.0):
            Ax64 = Ax64 - T_s * hS

        rf64 = wp.float64(b[j, i]) - Ax64     # integrated flux residual
        rh64 = rf64 / diagA                   # Jacobi correction in metres

    r[j, i] = WP_FLOAT(rh64)
    wp.atomic_add(rTr_buf, 0, rh64 * rh64)


@wp.kernel
def compute_residual_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    aq_thickness: float,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = WP_FLOAT(0.0)
        return

    tiny = wp.float64(1.0e-12)

    T_c = wp.float64(T_field[j, i])
    hC = wp.float64(x[j, i])

    hE = wp.float64(0.0)
    hW = wp.float64(0.0)
    hN = wp.float64(0.0)
    hS = wp.float64(0.0)

    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        hE = wp.float64(x[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        hW = wp.float64(x[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        hN = wp.float64(x[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        hS = wp.float64(x[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        width = wp.float64(gh_width[j, i])
        if width > wp.float64(0.0) and not wp.isnan(width):
            C_gh = (
                wp.float64(gh_alpha)
                * T_c
                / wp.float64(aq_thickness)
                * width
                * wp.float64(dx)
            )

    Sdiag = wp.float64(storage_diag[j, i])
    if Sdiag < wp.float64(0.0):
        Sdiag = wp.float64(0.0)
    sum_T = T_e + T_w + T_n + T_s + C_gh + Sdiag

    Ax64 = wp.float64(0.0)
    if sum_T < tiny:
        Ax64 = hC
    else:
        Ax64 = sum_T * hC
        if T_e > wp.float64(0.0):
            Ax64 = Ax64 - T_e * hE
        if T_w > wp.float64(0.0):
            Ax64 = Ax64 - T_w * hW
        if T_n > wp.float64(0.0):
            Ax64 = Ax64 - T_n * hN
        if T_s > wp.float64(0.0):
            Ax64 = Ax64 - T_s * hS

    rf64 = wp.float64(b[j, i]) - Ax64
    r[j, i] = WP_FLOAT(rf64)
    wp.atomic_add(rTr_buf, 0, rf64 * rf64)


@wp.kernel
def kcycle_check_dh_and_residual_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    x_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    dh2_buf: wp.array(dtype=wp.float64, ndim=1),
    dh_max_buf: wp.array(dtype=wp.float64, ndim=1),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    aq_thickness: float,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    # Always refresh snapshot so masked cells do not accumulate stale updates.
    x_new = wp.float64(x[j, i])
    x_old = wp.float64(x_prev[j, i])
    x_prev[j, i] = x[j, i]

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        return

    dh = x_new - x_old
    abs_dh = wp.abs(dh)
    wp.atomic_add(dh2_buf, 0, dh * dh)
    wp.atomic_max(dh_max_buf, 0, abs_dh)

    tiny = wp.float64(1.0e-12)

    T_c = wp.float64(T_field[j, i])
    hC = x_new

    hE = wp.float64(0.0)
    hW = wp.float64(0.0)
    hN = wp.float64(0.0)
    hS = wp.float64(0.0)

    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        hE = wp.float64(x[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        hW = wp.float64(x[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        hN = wp.float64(x[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        hS = wp.float64(x[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        width = wp.float64(gh_width[j, i])
        if width > wp.float64(0.0) and not wp.isnan(width):
            C_gh = (
                wp.float64(gh_alpha)
                * T_c
                / wp.float64(aq_thickness)
                * width
                * wp.float64(dx)
            )

    Sdiag = wp.float64(storage_diag[j, i])
    if Sdiag < wp.float64(0.0):
        Sdiag = wp.float64(0.0)
    sum_T = T_e + T_w + T_n + T_s + C_gh + Sdiag

    Ax64 = wp.float64(0.0)
    if sum_T < tiny:
        Ax64 = hC
    else:
        Ax64 = sum_T * hC
        if T_e > wp.float64(0.0):
            Ax64 = Ax64 - T_e * hE
        if T_w > wp.float64(0.0):
            Ax64 = Ax64 - T_w * hW
        if T_n > wp.float64(0.0):
            Ax64 = Ax64 - T_n * hN
        if T_s > wp.float64(0.0):
            Ax64 = Ax64 - T_s * hS

    rf64 = wp.float64(b[j, i]) - Ax64
    wp.atomic_add(rTr_buf, 0, rf64 * rf64)


def build_rhs_fd_like(
        T_field: np.ndarray,
        R_field: np.ndarray,
        active: np.ndarray,
        bc_mask: np.ndarray,
        bc_values: np.ndarray,
        dx: NP_FLOAT,
        gh_mask: np.ndarray | None = None,
        gh_head: np.ndarray | None = None,
        gh_width: np.ndarray | None = None,
        gh_alpha: NP_FLOAT = 1.0,
        head_scale: NP_FLOAT = 1.0,
        aq_thickness: NP_FLOAT = 1.0,
) -> np.ndarray:
    """
    :param T_field: np.ndarray transmissivity [L^2/T]
    :param R_field: np.ndarray recharge [L/T]
    :param active: np.ndarray 1 active, 0 inactive
    :param bc_mask: np.ndarray 1 for Dirichlet cells
    :param bc_values: np.ndarray Dirichlet head values [L]
    :param dx: float grid size [L]
    :param gh_mask: optional GHB mask
    :param gh_head: optional GHB heads [L]
    :param gh_width: optional GHB width (assumes gh_width is a river width m)
    :param gh_alpha: GHB scaling factor
    :param head_scale: characteristic head scale, h_scaled = h / head_scale -SHOULD NOT BE USED _LEGACY_
    :param aq_thickness: aquifer thickness [L] used in C_gh
    :return: np.ndarray RHS b_scaled such that A h_scaled = b_scaled
    """
    if head_scale != 1.0:    raise ValueError('legacy head_scale != 1.0 not supported anymore only pass if you are rewriting code to account for it properly')
    T_field = np.asarray(T_field, dtype=NP_FLOAT)
    R_field = np.asarray(R_field, dtype=NP_FLOAT)
    active = np.asarray(active, dtype=np.int32)
    bc_mask = np.asarray(bc_mask, dtype=np.int32)
    bc_values = np.asarray(bc_values, dtype=NP_FLOAT)

    ny, nx = T_field.shape
    dx = float(dx)
    dx2 = dx * dx

    if head_scale <= 0.0:
        raise ValueError("head_scale must be positive.")
    if aq_thickness <= 0.0:
        raise ValueError("aq_thickness must be positive.")

    # Interior RHS: b_phys = R * dx^2, then scale by head_scale
    b = (R_field * dx2 / head_scale).astype(NP_FLOAT)

    # Optional GHB contribution: b += C_gh * h_ext / head_scale
    if gh_mask is not None and gh_head is not None and gh_width is not None:
        gh_mask_arr = np.asarray(gh_mask, dtype=np.int32)
        gh_head_arr = np.asarray(gh_head, dtype=NP_FLOAT)
        gh_width_arr = np.asarray(gh_width, dtype=NP_FLOAT)

        # Same C_gh as in Warp kernels:
        # C_gh = gh_alpha * T_c / aq_thickness * width * dx
        C_gh = (
            gh_alpha
            * T_field / float(aq_thickness)
            * gh_width_arr
            * dx
        ).astype(NP_FLOAT)

        mask_gh = (
            (gh_mask_arr != 0)
            & (active != 0)
            & np.isfinite(gh_width_arr)
            & (gh_width_arr > 0.0)
        )

        if np.any(mask_gh):
            b[mask_gh] = b[mask_gh] + C_gh[mask_gh] * (
                gh_head_arr[mask_gh] / head_scale
            )

    # Inactive: identity with b = 0 (no need to scale)
    b[active == 0] = 0.0

    # Dirichlet: A row is identity, so we need h_scaled = bc / head_scale
    bc_idx = bc_mask != 0
    b[bc_idx] = bc_values[bc_idx] / head_scale

    return b.reshape(ny, nx)


@wp.kernel
def build_rhs_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    R_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_head: wp.array(dtype=WP_FLOAT, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    head_scale: float,
    aq_thickness: float,
    b_out: wp.array(dtype=WP_FLOAT, ndim=2),
):
    """
    Assemble RHS on device using the same physics as build_rhs_fd_like.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        b_out[j, i] = WP_FLOAT(0.0)
        return

    if bc_mask[j, i] != 0:
        b_out[j, i] = WP_FLOAT(wp.float64(bc_values[j, i]) / wp.float64(head_scale))
        return

    rhs = wp.float64(R_field[j, i]) * wp.float64(dx) * wp.float64(dx) / wp.float64(head_scale)

    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        T_c = wp.float64(T_field[j, i])
        width = wp.float64(gh_width[j, i])
        if T_c > wp.float64(0.0) and width > wp.float64(0.0) and not wp.isnan(width):
            C_gh = (
                wp.float64(gh_alpha)
                * T_c
                / wp.float64(aq_thickness)
                * width
                * wp.float64(dx)
            )
            rhs = rhs + C_gh * (wp.float64(gh_head[j, i]) / wp.float64(head_scale))

    b_out[j, i] = WP_FLOAT(rhs)


@wp.kernel
def add_storage_rhs_kernel(
    b_out: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    head_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or bc_mask[j, i] != 0:
        return

    sdiag = wp.float64(storage_diag[j, i])
    if sdiag <= wp.float64(0.0):
        return

    b_out[j, i] = WP_FLOAT(wp.float64(b_out[j, i]) + sdiag * wp.float64(head_prev[j, i]))


def build_diag_preconditioner(
    T_field: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    gh_mask: np.ndarray | None = None,
    gh_width: np.ndarray | None = None,
    storage_diag: np.ndarray | None = None,
    dx: float | None = None,
    gh_alpha: float = 1.0,
    aq_thickness: float = 1.0,
) -> np.ndarray:
    """
    :param T_field: transmissivity field
    :param active: active mask
    :param bc_mask: Dirichlet mask
    :param gh_mask: optional GHB mask
    :param gh_width: optional GHB width (assumes gh_width is a river width m)
    :param storage_diag: optional transient storage diagonal contribution [integrated units]
    :param dx: cell size (required if gh_mask provided)
    :param gh_alpha: GHB scaling factor
    :param aq_thickness: aquifer thickness used in C_gh
    :return: diagonal preconditioner M_inv with dtype NP_FLOAT, consistent with kernels
    """
    T_field = np.asarray(T_field, dtype=NP_FLOAT)
    active = np.asarray(active, dtype=np.int32)
    bc_mask = np.asarray(bc_mask, dtype=np.int32)

    use_ghb = (gh_mask is not None) and (gh_width is not None)
    if use_ghb and dx is None:
        raise ValueError("dx must be provided when gh_mask/gh_width are provided.")

    if use_ghb:
        gh_mask_arr = np.asarray(gh_mask, dtype=np.int32)
        gh_width_arr = np.asarray(gh_width, dtype=NP_FLOAT)
        dx_f = float(dx)
        gh_alpha_f = float(gh_alpha)
        thickness_f = float(aq_thickness)
        if thickness_f <= 0.0:
            thickness_f = 1.0
    else:
        gh_mask_arr = None
        gh_width_arr = None
        dx_f = 1.0
        gh_alpha_f = 1.0
        thickness_f = 1.0

    ny, nx = T_field.shape
    tiny = np.float64(1.0e-12)

    T_pos = np.isfinite(T_field) & (T_field > NP_FLOAT(0.0))
    free = (active != 0) & (bc_mask == 0)

    sum_T = np.zeros((ny, nx), dtype=np.float64)

    # East/West connections
    if nx > 1:
        T_L = T_field[:, :-1].astype(np.float64, copy=False)
        T_R = T_field[:, 1:].astype(np.float64, copy=False)
        denom_E = T_L + T_R
        valid_E = (
            (active[:, :-1] != 0)
            & (active[:, 1:] != 0)
            & T_pos[:, :-1]
            & T_pos[:, 1:]
            & (denom_E > tiny)
        )
        cond_E = np.zeros_like(denom_E, dtype=np.float64)
        cond_E[valid_E] = 2.0 * T_L[valid_E] * T_R[valid_E] / denom_E[valid_E]

        sum_T[:, :-1] += cond_E  # T_e
        sum_T[:, 1:] += cond_E   # T_w

    # North/South connections
    if ny > 1:
        T_T = T_field[:-1, :].astype(np.float64, copy=False)
        T_B = T_field[1:, :].astype(np.float64, copy=False)
        denom_S = T_T + T_B
        valid_S = (
            (active[:-1, :] != 0)
            & (active[1:, :] != 0)
            & T_pos[:-1, :]
            & T_pos[1:, :]
            & (denom_S > tiny)
        )
        cond_S = np.zeros_like(denom_S, dtype=np.float64)
        cond_S[valid_S] = 2.0 * T_T[valid_S] * T_B[valid_S] / denom_S[valid_S]

        sum_T[:-1, :] += cond_S  # T_s
        sum_T[1:, :] += cond_S   # T_n

    # GHB diagonal contribution (same form as Warp kernels)
    if use_ghb:
        width_ok = np.isfinite(gh_width_arr) & (gh_width_arr > NP_FLOAT(0.0))
        gh_on = (gh_mask_arr != 0) & width_ok & T_pos
        if np.any(gh_on):
            C_gh = (
                np.float64(gh_alpha_f)
                * T_field.astype(np.float64, copy=False)
                / np.float64(thickness_f)
                * gh_width_arr.astype(np.float64, copy=False)
                * np.float64(dx_f)
            )
            sum_T[gh_on] += C_gh[gh_on]

    # Optional transient storage diagonal contribution
    if storage_diag is not None:
        Sdiag = np.asarray(storage_diag, dtype=np.float64)
        if Sdiag.shape != (ny, nx):
            raise ValueError(f"storage_diag shape {Sdiag.shape} expected {(ny, nx)}")
        if not np.all(np.isfinite(Sdiag)):
            raise ValueError("storage_diag contains non-finite values")
        if np.any(Sdiag < 0.0):
            raise ValueError("storage_diag must be >= 0")
        sum_T[free] += Sdiag[free]

    M_inv = np.ones((ny, nx), dtype=np.float64)
    valid_sum = np.isfinite(sum_T) & (sum_T > tiny)
    M_inv[valid_sum] = 1.0 / sum_T[valid_sum]

    # Enforce identity-like behavior for inactive/Dirichlet cells
    M_inv[~free] = 1.0

    return M_inv.astype(NP_FLOAT, copy=False)




@wp.kernel
def apply_A_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    h: wp.array(dtype=WP_FLOAT, ndim=2),
    Ah: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    aq_thickness: float,
):
    """
    Apply the same discrete operator A as solve_darcy_fd_2d_matrix.

    Interior active non Dirichlet cell:
        sum_T = T_e + T_w + T_n + T_s
        (A h)_C = sum_T*hC - T_e*hE - T_w*hW - T_n*hN - T_s*hS

    Inactive and Dirichlet cells:
        A h = h  (identity row)
    """
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    # Inactive and Dirichlet: identity row
    if active[j, i] == 0 or bc_mask[j, i] != 0:
        Ah[j, i] = h[j, i]
        return

    T_c = T_field[j, i]
    hC = h[j, i]

    T_e = WP_FLOAT(0.0)
    T_w = WP_FLOAT(0.0)
    T_n = WP_FLOAT(0.0)
    T_s = WP_FLOAT(0.0)

    tiny = WP_FLOAT(1.0e-12)

    # East neighbor
    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = T_field[j, i + 1]
        if T_c > WP_FLOAT(0.0) and T_nb > WP_FLOAT(0.0):
            T_e = WP_FLOAT(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    # West neighbor
    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = T_field[j, i - 1]
        if T_c > WP_FLOAT(0.0) and T_nb > WP_FLOAT(0.0):
            T_w = WP_FLOAT(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    # North neighbor
    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = T_field[j - 1, i]
        if T_c > WP_FLOAT(0.0) and T_nb > WP_FLOAT(0.0):
            T_n = WP_FLOAT(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    # South neighbor
    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = T_field[j + 1, i]
        if T_c > WP_FLOAT(0.0) and T_nb > WP_FLOAT(0.0):
            T_s = WP_FLOAT(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = WP_FLOAT(0.0)
    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        width = gh_width[j, i]
        if width > WP_FLOAT(0.0) and not wp.isnan(width):
            C_gh = (
                WP_FLOAT(gh_alpha)
                * T_c
                / WP_FLOAT(aq_thickness)
                * width
                * WP_FLOAT(dx)
            )

    Sdiag = wp.float64(storage_diag[j, i])
    if Sdiag < wp.float64(0.0):
        Sdiag = wp.float64(0.0)
    sum_T = wp.float64(T_e) + wp.float64(T_w) + wp.float64(T_n) + wp.float64(T_s) + wp.float64(C_gh) + Sdiag

    # Mirror FD behavior: if isolated, treat as identity with b = 0
    if sum_T < wp.float64(tiny):
        Ah[j, i] = hC
        return

    # A h = sum_T*hC - T_e*hE - T_w*hW - T_n*hN - T_s*hS
    val = sum_T * wp.float64(hC)

    if T_e > WP_FLOAT(0.0):
        val = val - wp.float64(T_e * h[j, i + 1])
    if T_w > WP_FLOAT(0.0):
        val = val - wp.float64(T_w * h[j, i - 1])
    if T_n > WP_FLOAT(0.0):
        val = val - wp.float64(T_n * h[j - 1, i])
    if T_s > WP_FLOAT(0.0):
        val = val - wp.float64(T_s * h[j + 1, i])

    Ah[j, i] = WP_FLOAT(val)


@wp.kernel
def apply_A_and_pAp_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    Ap: wp.array(dtype=WP_FLOAT, ndim=2),
    pAp_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    aq_thickness: float,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        Ap[j, i] = p[j, i]
        return

    tiny = wp.float64(1.0e-12)

    T_c = wp.float64(T_field[j, i])
    pC = wp.float64(p[j, i])

    pE = wp.float64(0.0)
    pW = wp.float64(0.0)
    pN = wp.float64(0.0)
    pS = wp.float64(0.0)

    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        pE = wp.float64(p[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        pW = wp.float64(p[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        pN = wp.float64(p[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        pS = wp.float64(p[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        width = wp.float64(gh_width[j, i])
        if width > wp.float64(0.0) and not wp.isnan(width):
            C_gh = (
                wp.float64(gh_alpha)
                * T_c
                / wp.float64(aq_thickness)
                * width
                * wp.float64(dx)
            )

    Sdiag = wp.float64(storage_diag[j, i])
    if Sdiag < wp.float64(0.0):
        Sdiag = wp.float64(0.0)
    sum_T = T_e + T_w + T_n + T_s + C_gh + Sdiag

    val64 = wp.float64(0.0)
    if sum_T < tiny:
        val64 = pC
    else:
        val64 = sum_T * pC
        if T_e > wp.float64(0.0):
            val64 = val64 - T_e * pE
        if T_w > wp.float64(0.0):
            val64 = val64 - T_w * pW
        if T_n > wp.float64(0.0):
            val64 = val64 - T_n * pN
        if T_s > wp.float64(0.0):
            val64 = val64 - T_s * pS

    Ap[j, i] = WP_FLOAT(val64)
    wp.atomic_add(pAp_buf, 0, pC * val64)




@wp.kernel
def init_pcg_kernel(
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    Ap: wp.array(dtype=WP_FLOAT, ndim=2),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    z: wp.array(dtype=WP_FLOAT, ndim=2),
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    rho_buf: wp.array(dtype=wp.float64, ndim=1),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = WP_FLOAT(0.0)
        z[j, i] = WP_FLOAT(0.0)
        p[j, i] = WP_FLOAT(0.0)
        return

    r_val = b[j, i] - Ap[j, i]
    z_val = M_inv[j, i] * r_val

    r[j, i] = r_val
    z[j, i] = z_val
    p[j, i] = z_val

    r64 = wp.float64(r_val)
    z64 = wp.float64(z_val)

    wp.atomic_add(rho_buf, 0, r64 * z64)
    wp.atomic_add(rTr_buf, 0, r64 * r64)


@wp.kernel
def init_pcg_with_A_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=2),
    Ap: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    z: wp.array(dtype=WP_FLOAT, ndim=2),
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    rho_buf: wp.array(dtype=wp.float64, ndim=1),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    dx: float,
    gh_alpha: float,
    aq_thickness: float,
):
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        Ap[j, i] = x[j, i]
        r[j, i] = WP_FLOAT(0.0)
        z[j, i] = WP_FLOAT(0.0)
        p[j, i] = WP_FLOAT(0.0)
        return

    tiny = wp.float64(1.0e-12)

    T_c = wp.float64(T_field[j, i])
    hC = wp.float64(x[j, i])

    hE = wp.float64(0.0)
    hW = wp.float64(0.0)
    hN = wp.float64(0.0)
    hS = wp.float64(0.0)

    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        hE = wp.float64(x[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        hW = wp.float64(x[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        hN = wp.float64(x[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        hS = wp.float64(x[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0 and aq_thickness > 0.0:
        width = wp.float64(gh_width[j, i])
        if width > wp.float64(0.0) and not wp.isnan(width):
            C_gh = (
                wp.float64(gh_alpha)
                * T_c
                / wp.float64(aq_thickness)
                * width
                * wp.float64(dx)
            )

    Sdiag = wp.float64(storage_diag[j, i])
    if Sdiag < wp.float64(0.0):
        Sdiag = wp.float64(0.0)
    sum_T = T_e + T_w + T_n + T_s + C_gh + Sdiag

    Ap64 = wp.float64(0.0)
    if sum_T < tiny:
        Ap64 = hC
    else:
        Ap64 = sum_T * hC
        if T_e > wp.float64(0.0):
            Ap64 = Ap64 - T_e * hE
        if T_w > wp.float64(0.0):
            Ap64 = Ap64 - T_w * hW
        if T_n > wp.float64(0.0):
            Ap64 = Ap64 - T_n * hN
        if T_s > wp.float64(0.0):
            Ap64 = Ap64 - T_s * hS

    Ap_val = WP_FLOAT(Ap64)
    Ap[j, i] = Ap_val

    r_val = b[j, i] - Ap_val
    z_val = M_inv[j, i] * r_val

    r[j, i] = r_val
    z[j, i] = z_val
    p[j, i] = z_val

    r64 = wp.float64(r_val)
    z64 = wp.float64(z_val)

    wp.atomic_add(rho_buf, 0, r64 * z64)
    wp.atomic_add(rTr_buf, 0, r64 * r64)


@wp.kernel
def update_x_r_z_rho_rTr_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    z: wp.array(dtype=WP_FLOAT, ndim=2),
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    Ap: wp.array(dtype=WP_FLOAT, ndim=2),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    alpha_buf: wp.array(dtype=wp.float64, ndim=1),
    rho_buf: wp.array(dtype=wp.float64, ndim=1),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = WP_FLOAT(0.0)
        z[j, i] = WP_FLOAT(0.0)
        return

    alpha = alpha_buf[0]

    x64 = wp.float64(x[j, i])
    r64 = wp.float64(r[j, i])
    p64 = wp.float64(p[j, i])
    Ap64 = wp.float64(Ap[j, i])

    x_new = x64 + alpha * p64
    r_new = r64 - alpha * Ap64

    z_new = wp.float64(M_inv[j, i]) * r_new

    x[j, i] = WP_FLOAT(x_new)
    r[j, i] = WP_FLOAT(r_new)
    z[j, i] = WP_FLOAT(z_new)

    wp.atomic_add(rho_buf, 0, r_new * z_new)
    wp.atomic_add(rTr_buf, 0, r_new * r_new)


@wp.kernel
def update_p_kernel(
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    z: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    beta_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        p[j, i] = WP_FLOAT(0.0)
        return

    beta = beta_buf[0]

    p64 = wp.float64(p[j, i])
    z64 = wp.float64(z[j, i])

    p[j, i] = WP_FLOAT(z64 + beta * p64)


@wp.kernel
def zero_scalar_kernel(
    buf: wp.array(dtype=wp.float64, ndim=1),
):
    """
    Zero a 1D Warp array (length >= 1).
    """
    k = wp.tid()
    if k >= buf.shape[0]:
        return
    buf[k] = wp.float64(0.0)


@wp.kernel
def reset_kcycle_check_buffers_kernel(
    dh2_buf: wp.array(dtype=wp.float64, ndim=1),
    dh_max_buf: wp.array(dtype=wp.float64, ndim=1),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    converged_flag: wp.array(dtype=wp.int32, ndim=1),
):
    if wp.tid() != 0:
        return
    dh2_buf[0] = wp.float64(0.0)
    dh_max_buf[0] = wp.float64(0.0)
    rTr_buf[0] = wp.float64(0.0)
    converged_flag[0] = 0


@wp.kernel
def compute_alpha_kernel(
    rho_buf: wp.array(dtype=wp.float64, ndim=1),
    pAp_buf: wp.array(dtype=wp.float64, ndim=1),
    alpha_buf: wp.array(dtype=wp.float64, ndim=1),
):
    k = wp.tid()
    if k > 0:
        return

    rho64 = rho_buf[0]
    pAp64 = pAp_buf[0]

    if pAp64 != wp.float64(0.0):
        alpha_buf[0] = rho64 / pAp64

    else:
        alpha_buf[0] = wp.float64(0.0)


@wp.kernel
def compute_beta_and_update_rho_kernel(
    rho_buf: wp.array(dtype=wp.float64, ndim=1),
    rho_new_buf: wp.array(dtype=wp.float64, ndim=1),
    beta_buf: wp.array(dtype=wp.float64, ndim=1),
):
    if wp.tid() != 0:
        return

    rho_old = rho_buf[0]
    rho_new = rho_new_buf[0]

    if rho_old > wp.float64(0.0):
        beta_buf[0] = rho_new / rho_old
    else:
        beta_buf[0] = wp.float64(0.0)

    rho_buf[0] = rho_new


@wp.kernel
def check_convergence_kernel(
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    n_active: int,
    tol_abs: float,
    converged_flag: wp.array(dtype=wp.int32, ndim=1),
):
    k = wp.tid()
    if k > 0:
        return

    if n_active <= 0:
        converged_flag[0] = 1
        return

    rTr64 = rTr_buf[0]
    if rTr64 < wp.float64(0.0):
        converged_flag[0] = 0
        return

    r_rms64 = wp.sqrt(rTr64 / wp.float64(n_active))

    if r_rms64 <= wp.float64(tol_abs):
        converged_flag[0] = 1
    else:
        converged_flag[0] = 0


@wp.kernel
def dot_active_kernel(
    a: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    out_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or bc_mask[j, i] != 0:
        return
    wp.atomic_add(out_buf, 0, wp.float64(a[j, i]) * wp.float64(b[j, i]))

@wp.kernel
def axpy_active_scalar_kernel(
    y: wp.array(dtype=WP_FLOAT, ndim=2),         # y[iy,ix]
    x: wp.array(dtype=WP_FLOAT, ndim=2),         # x[iy,ix]
    active: wp.array(dtype=wp.int32),            # active[idx] (1D)
    bc_mask: wp.array(dtype=wp.int32),           # bc_mask[idx] (1D)
    alpha_buf: wp.array(dtype=wp.float64),         # alpha_buf[0]
    nx: int,
    ny: int,
):
    iy, ix = wp.tid()
    if iy >= ny or ix >= nx:
        return

    idx = iy * nx + ix

    if active[idx] == 0:
        return
    if bc_mask[idx] != 0:
        return

    a = WP_FLOAT(alpha_buf[0])
    y[iy, ix] = y[iy, ix] + a * x[iy, ix]

@wp.kernel
def axpy_active_scalar_2dmask_kernel(
    y: wp.array(dtype=WP_FLOAT, ndim=2),
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    alpha_buf: wp.array(dtype=wp.float64),   # alpha_buf[0]
    nx: int,
    ny: int,
):
    iy, ix = wp.tid()
    if iy >= ny or ix >= nx:
        return

    if active[iy, ix] == 0:
        return
    if bc_mask[iy, ix] != 0:
        return

    a = WP_FLOAT(alpha_buf[0])
    y[iy, ix] = y[iy, ix] + a * x[iy, ix]


@wp.kernel
def compute_safe_alpha_kernel(
    num_buf: wp.array(dtype=wp.float64),
    den_buf: wp.array(dtype=wp.float64),
    alpha_buf: wp.array(dtype=wp.float64),
):
    if wp.tid() != 0:
        return

    den = den_buf[0]
    if den != WP_FLOAT(0.0):
        alpha_buf[0] = wp.float64(num_buf[0] / wp.float64(den))
    else:
        alpha_buf[0] = wp.float64(0.0)

@wp.kernel
def copy_field_kernel(
    src: wp.array(dtype=WP_FLOAT, ndim=2),
    dst: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    dst[j, i] = src[j, i]


@wp.kernel
def enforce_constraints_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    z: wp.array(dtype=WP_FLOAT, ndim=2),
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),  # physical heads, same shape
    head_scale: float,
):
    j, i = wp.tid()

    if active[j, i] == 0:
        x[j, i] = WP_FLOAT(0.0)
        r[j, i] = WP_FLOAT(0.0)
        z[j, i] = WP_FLOAT(0.0)
        p[j, i] = WP_FLOAT(0.0)
        return

    if bc_mask[j, i] != 0:
        x[j, i] = WP_FLOAT(wp.float64(bc_values[j, i]) / wp.float64(head_scale))
        r[j, i] = WP_FLOAT(0.0)
        z[j, i] = WP_FLOAT(0.0)
        p[j, i] = WP_FLOAT(0.0)
        return

@wp.kernel
def check_rtr_converged_kernel(
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    thr_rTr: wp.float64,
    converged_flag: wp.array(dtype=wp.int32, ndim=1),
):
    if wp.tid() == 0:
        converged_flag[0] = wp.int32(1) if rTr_buf[0] <= thr_rTr else wp.int32(0)

@wp.kernel
def head_update_rms_and_snapshot_kernel(
        x: wp.array(dtype=WP_FLOAT, ndim=2),
        x_prev: wp.array(dtype=WP_FLOAT, ndim=2),
        active: wp.array(dtype=wp.int32, ndim=2),
        bc_mask: wp.array(dtype=wp.int32, ndim=2),
        dh2_buf: wp.array(dtype=wp.float64, ndim=1),  # accumulates sum(dh^2)
        dh_max_buf: wp.array(dtype=wp.float64, ndim=1),  # accumulates max(|dh|)
        nx: int,
        ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    # Always update snapshot so masked cells do not accumulate stale diffs later
    x_new = wp.float64(x[j, i])
    x_old = wp.float64(x_prev[j, i])
    x_prev[j, i] = x[j, i]

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        return

    dh = x_new - x_old
    abs_dh = wp.abs(dh)

    wp.atomic_add(dh2_buf, 0, dh * dh)
    wp.atomic_max(dh_max_buf, 0, abs_dh)


@dataclass(slots=True)
class MGLevel:
    level_id: int
    nx: int
    ny: int
    dx: float
    n_active: int

    # Host fields (always 2D)
    T_host: np.ndarray
    R_host: np.ndarray
    active_host: np.ndarray
    bc_mask_host: np.ndarray
    bc_values_host: np.ndarray
    gh_mask_host: Optional[np.ndarray]
    gh_head_host: Optional[np.ndarray]
    gh_width_host: Optional[np.ndarray]

    # Device fields (always 2D)
    T_wp: wp.array
    R_wp: wp.array
    active_wp: wp.array
    bc_mask_wp: wp.array
    bc_values_wp: wp.array
    gh_mask_wp: Optional[wp.array]
    gh_head_wp: Optional[wp.array]
    gh_width_wp: Optional[wp.array]

    # Diagonal preconditioner for this level
    M_inv_wp: wp.array

    # Persistent work buffers (ready for V, W, K cycles)
    x_wp: wp.array
    b_wp: wp.array
    r_wp: wp.array
    Ax_wp: wp.array
    e_wp: wp.array

    # PCG buffers (K-cycle will often use inner Krylov on coarse levels)
    z_wp: wp.array
    p_wp: wp.array
    Ap_wp: wp.array

    # Scalar buffers
    rTr_buf: wp.array
    rho_buf: wp.array
    rho_new_buf: wp.array
    pAp_buf: wp.array
    alpha_buf: wp.array
    beta_buf: wp.array
    converged_flag: wp.array



class WarpDarcySolver:
    """
    GPU based solver for 2D steady Darcy flow using Warp.
    Supports PCG and a 2-level multigrid V cycle (Jacobi on fine, PCG on coarse).
    """

    # ----------------------------
    # Small internal container for level-specific operator data (2-level MG path)
    # ----------------------------
    class _GridLevel:
        __slots__ = (
            "T_wp",
            "active_wp",
            "bc_mask_wp",
            "gh_mask_wp",
            "gh_width_wp",
            "M_inv_wp",
            "nx",
            "ny",
            "dx",
        )

        def __init__(
            self,
            T_wp,
            active_wp,
            bc_mask_wp,
            gh_mask_wp,
            gh_width_wp,
            M_inv_wp,
            nx: int,
            ny: int,
            dx: float,
        ):
            self.T_wp = T_wp
            self.active_wp = active_wp
            self.bc_mask_wp = bc_mask_wp
            self.gh_mask_wp = gh_mask_wp
            self.gh_width_wp = gh_width_wp
            self.M_inv_wp = M_inv_wp
            self.nx = int(nx)
            self.ny = int(ny)
            self.dx = float(dx)


    # ----------------------------
    # Level container for build_hierarchy (ready for K cycle)
    # ----------------------------
    class _MGLevel:
        __slots__ = (
            "level_id",
            "nx",
            "ny",
            "dx",
            "n_active",
            # host fields
            "T_host",
            "R_host",
            "active_host",
            "bc_mask_host",
            "bc_values_host",
            "gh_mask_host",
            "gh_head_host",
            "gh_width_host",
            # device fields
            "T_wp",
            "R_wp",
            "active_wp",
            "bc_mask_wp",
            "bc_values_wp",
            "gh_mask_wp",
            "gh_head_wp",
            "gh_width_wp",
            "M_inv_wp",
            "storage_wp",
            # persistent work buffers (MG + PCG)
            "x_wp",
            "b_wp",
            "r_wp",
            "Ax_wp",
            "e_wp",
            "z_wp",
            "p_wp",
            "Ap_wp",
            "rTr_buf",
            "rho_buf",
            "rho_new_buf",
            "pAp_buf",
            "alpha_buf",
            "beta_buf",
            "converged_flag",
            "x_prev_wp",
            "dh_max_buf",
        )

        def __init__(
            self,
            level_id: int,
            nx: int,
            ny: int,
            dx: float,
            n_active: int,
            T_host,
            R_host,
            active_host,
            bc_mask_host,
            bc_values_host,
            gh_mask_host,
            gh_head_host,
            gh_width_host,
            T_wp,
            R_wp,
            active_wp,
            bc_mask_wp,
            bc_values_wp,
            gh_mask_wp,
            gh_head_wp,
            gh_width_wp,
            M_inv_wp,
            storage_wp,
            x_wp,
            b_wp,
            r_wp,
            Ax_wp,
            e_wp,
            z_wp,
            p_wp,
            Ap_wp,
            rTr_buf,
            rho_buf,
            rho_new_buf,
            pAp_buf,
            alpha_buf,
            beta_buf,
            converged_flag,
            x_prev_wp,
            dh_max_buf,
        ):
            self.level_id = int(level_id)
            self.nx = int(nx)
            self.ny = int(ny)
            self.dx = float(dx)
            self.n_active = int(n_active)

            self.T_host = T_host
            self.R_host = R_host
            self.active_host = active_host
            self.bc_mask_host = bc_mask_host
            self.bc_values_host = bc_values_host
            self.gh_mask_host = gh_mask_host
            self.gh_head_host = gh_head_host
            self.gh_width_host = gh_width_host

            self.T_wp = T_wp
            self.R_wp = R_wp
            self.active_wp = active_wp
            self.bc_mask_wp = bc_mask_wp
            self.bc_values_wp = bc_values_wp
            self.gh_mask_wp = gh_mask_wp
            self.gh_head_wp = gh_head_wp
            self.gh_width_wp = gh_width_wp
            self.M_inv_wp = M_inv_wp
            self.storage_wp = storage_wp

            self.x_wp = x_wp
            self.b_wp = b_wp
            self.r_wp = r_wp
            self.Ax_wp = Ax_wp
            self.e_wp = e_wp

            self.z_wp = z_wp
            self.p_wp = p_wp
            self.Ap_wp = Ap_wp

            self.rTr_buf = rTr_buf
            self.rho_buf = rho_buf
            self.rho_new_buf = rho_new_buf
            self.pAp_buf = pAp_buf
            self.alpha_buf = alpha_buf
            self.beta_buf = beta_buf
            self.converged_flag = converged_flag
            self.x_prev_wp = x_prev_wp
            self.dh_max_buf = dh_max_buf

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        device: str = "cuda:0",
        use_ghb: bool = False,
        solver_type: str = "pcg",
        head_scale: float = 1.0,
        aq_thickness: float = 1.0,
    ):
        """
        :param nx: number of columns
        :param ny: number of rows
        :param dx: cell size
        :param device: Warp device string, for example "cuda:0"
        :param use_ghb: if True, include GHB terms in operator and RHS assembly
        :param solver_type: "pcg" or "jacobi" (future)
        :param head_scale: characteristic head scale, h_scaled = h / head_scale
        :param aq_thickness: aquifer thickness used in GHB conductance scaling
        """
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.device_str = str(device)
        self.use_ghb = bool(use_ghb)
        self.solver_type = str(solver_type)

        if head_scale != 1.0:
            raise ValueError("head_scale has been removed. Use physical heads everywhere and set head_scale=1.0.")
        self.head_scale = 1.0


        if aq_thickness <= 0.0:
            raise ValueError("aq_thickness must be positive.")
        self.aq_thickness = float(aq_thickness)

        # Host side storage for fields
        self.T_field_host = None
        self.R_field_host = None
        self.active_host = None
        self.bc_mask_host = None
        self.bc_values_host = None
        self.gh_mask_host = None
        self.gh_head_host = None
        self.gh_width_host = None
        self.gh_alpha = 1.0

        # Device side Warp arrays (set in build_from_truth_inputs)
        self.T_wp = None
        self.R_wp = None
        self.active_wp = None
        self.bc_mask_wp = None
        self.bc_values_wp = None
        self.gh_mask_wp = None
        self.gh_head_wp = None
        self.gh_width_wp = None

        # Fine-level diagonal preconditioner
        self.M_inv_wp = None
        # Optional transient storage diagonal (fine level)
        self.storage_wp = None

        # Vectors for PCG
        self.b_wp = None
        self.x_wp = None
        self.r_wp = None
        self.z_wp = None
        self.p_wp = None
        self.Ap_wp = None

        # Scalar buffers (PCG)
        self.rho_buf = None
        self.rho_new_buf = None
        self.rTr_buf = None
        self.pAp_buf = None
        self.alpha_buf = None
        self.beta_buf = None
        self.converged_flag = None

        # Active cell count
        self.n_active = 0

        # ---------------- Multigrid cache (2-level path) ----------------
        self.mg_cache_built = False

        self.nx_c = None
        self.ny_c = None
        self.dx_c = None
        self.n_active_c = 0

        # Coarse host arrays (optional)
        self.T_c_host = None
        self.R_c_host = None
        self.active_c_host = None
        self.bc_mask_c_host = None
        self.bc_values_c_host = None
        self.gh_mask_c_host = None
        self.gh_head_c_host = None
        self.gh_width_c_host = None

        # Coarse device arrays
        self.T_c_wp = None
        self.R_c_wp = None
        self.active_c_wp = None
        self.bc_mask_c_wp = None
        self.bc_values_c_wp = None
        self.gh_mask_c_wp = None
        self.gh_head_c_wp = None
        self.gh_width_c_wp = None
        self.M_inv_c_wp = None

        # Internal level objects (built after uploads)
        self._fine_level = None
        self._coarse_level = None

        # Multigrid work buffers (allocated once per geometry)
        self._mg_work_built = False
        self._mg_work = {}

        # CPU staging buffers for in-place updates (avoid per-update wp.array allocations)
        self._stage_T0_host = None
        self._stage_M0_host = None

        self._stage_T0 = None
        self._stage_M0 = None
        self._stage_R0_host = None
        self._stage_R0 = None

        self._stage_Tc_2lvl = None
        self._stage_Mc_2lvl = None

        self._stage_T_levels = None
        self._stage_M_levels = None
        self._stage_ghw_levels = None
        self._stage_storage_levels = None

        self.T_field_host = None
        self._T_field_wp_host = None
        self.T_field_dev = None

        self._operator_dirty = True

        # Hierarchy storage (for K cycle later)
        self.mg_levels = None
        # Tracks whether the previous K-cycle solve used a spatial gh_alpha field.
        self._kcycle_spatial_alpha_active = False
        self._kcycle_transient_active = False
        self._kcycle_last_gh_width_eff_levels = None
        # ---------------- CUDA graph cache (K-cycle path) ----------------
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

    def _invalidate_kcycle_graph(self) -> None:
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

    # -------------------------------------------------------------------------
    # Hierarchy (ready for K cycle, not used by 2-level solve yet)
    # -------------------------------------------------------------------------
    def build_hierarchy(self, max_levels: int, min_coarse_n: int = 4) -> None:
        """
        Build a geometric multigrid hierarchy with 2:1 coarsening that works for any
        (nx, ny), including odd sizes, using ceil coarsening:
            nx_c = (nx_f + 1)//2
            ny_c = (ny_f + 1)//2

        Persistent per-level buffers are allocated so V cycle, W cycle, and K cycle
        can run without per-cycle allocations.

        Correction scheme assumption for levels > 0:
          - bc_values_host is set to 0.0 (homogeneous Dirichlet for error equation)
          - gh_head_host is set to 0.0 (homogeneous GHB head for error equation)

        If you later implement FAS, keep physical bc_values and gh_head on coarse levels
        instead of zeroing them here.

        :param max_levels: maximum number of levels including the finest
        :param min_coarse_n: stop if nx or ny would drop below this
        """
        if int(max_levels) < 1:
            raise ValueError("max_levels must be >= 1")

        if self.T_field_host is None or self.R_field_host is None:
            raise RuntimeError("build_from_truth_inputs must be called before build_hierarchy().")

        if self.T_wp is None or self.active_wp is None or self.bc_mask_wp is None:
            raise RuntimeError("Device arrays not initialized. Call build_from_truth_inputs() first.")

        # Invalidate any cached CUDA graph for K-cycle, because hierarchy buffers change.
        if hasattr(self, "_invalidate_kcycle_graph"):
            self._invalidate_kcycle_graph()
        else:
            self._kcycle_graph = None
            self._kcycle_graph_shape = None

        device = self.device_str

        levels = []

        lvl0 = self._mg_make_level_from_existing_fine(device=device)
        levels.append(lvl0)

        for level_id in range(1, int(max_levels)):
            fine = levels[level_id - 1]

            nx_c = (int(fine.nx) + 1) // 2
            ny_c = (int(fine.ny) + 1) // 2

            if nx_c < int(min_coarse_n) or ny_c < int(min_coarse_n):
                break

            dx_c = float(fine.dx) * 2.0

            (
                T_c,
                R_c,
                active_c,
                bc_mask_c,
                bc_values_c,
                gh_mask_c,
                gh_head_c,
                gh_width_c,
            ) = self._mg_coarsen_host_any(
                T_f=fine.T_host,
                R_f=fine.R_host,
                active_f=fine.active_host,
                bc_mask_f=fine.bc_mask_host,
                bc_values_f=fine.bc_values_host,
                gh_mask_f=fine.gh_mask_host,
                gh_head_f=fine.gh_head_host,
                gh_width_f=fine.gh_width_host,
                dx_c=float(dx_c),
            )

            # Homogeneous BCs on coarse levels for error equation (correction scheme).
            bc_values_c.fill(0.0)
            if gh_head_c is not None:
                gh_head_c.fill(0.0)

            n_active_c = int(np.count_nonzero(active_c))

            # Upload coarse fields
            T_c_wp = wp.array(T_c, dtype=WP_FLOAT, device=device)
            R_c_wp = wp.array(R_c, dtype=WP_FLOAT, device=device)
            active_c_wp = wp.array(active_c, dtype=wp.int32, device=device)
            bc_mask_c_wp = wp.array(bc_mask_c, dtype=wp.int32, device=device)
            bc_values_c_wp = wp.array(bc_values_c, dtype=WP_FLOAT, device=device)

            if self.use_ghb and gh_mask_c is not None:
                gh_mask_c_wp = wp.array(gh_mask_c, dtype=wp.int32, device=device)
                gh_head_c_wp = wp.array(gh_head_c, dtype=WP_FLOAT, device=device)
                gh_width_c_wp = wp.array(gh_width_c, dtype=WP_FLOAT, device=device)
            else:
                gh_mask_c_wp = None
                gh_head_c_wp = None
                gh_width_c_wp = None

            # Diagonal preconditioner on coarse level
            M_inv_c_host = build_diag_preconditioner(
                T_field=T_c,
                active=active_c,
                bc_mask=bc_mask_c,
                gh_mask=gh_mask_c if self.use_ghb else None,
                gh_width=gh_width_c if self.use_ghb else None,
                dx=float(dx_c) if self.use_ghb else None,
                gh_alpha=float(self.gh_alpha),
                aq_thickness=float(self.aq_thickness),
            )
            M_inv_c_wp = wp.array(M_inv_c_host, dtype=WP_FLOAT, device=device)

            # Persistent work buffers for this level
            shape_c = (int(ny_c), int(nx_c))
            storage_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            x_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            b_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            r_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            Ax_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            e_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)

            z_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            p_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            Ap_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)

            # Scalar buffers (always float64 for stable reductions)
            rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
            rho_buf = wp.zeros(1, dtype=wp.float64, device=device)
            rho_new_buf = wp.zeros(1, dtype=wp.float64, device=device)
            pAp_buf = wp.zeros(1, dtype=wp.float64, device=device)
            alpha_buf = wp.zeros(1, dtype=wp.float64, device=device)
            beta_buf = wp.zeros(1, dtype=wp.float64, device=device)
            converged_flag = wp.zeros(1, dtype=wp.int32, device=device)
            x_prev_wp = wp.zeros(shape_c, dtype=WP_FLOAT, device=device)
            dh_max_buf = wp.zeros(1, dtype=wp.float64, device=device)

            coarse = self._MGLevel(
                level_id=int(level_id),
                nx=int(nx_c),
                ny=int(ny_c),
                dx=float(dx_c),
                n_active=int(n_active_c),
                T_host=T_c,
                R_host=R_c,
                active_host=active_c,
                bc_mask_host=bc_mask_c,
                bc_values_host=bc_values_c,
                gh_mask_host=gh_mask_c if self.use_ghb else None,
                gh_head_host=gh_head_c if self.use_ghb else None,
                gh_width_host=gh_width_c if self.use_ghb else None,
                T_wp=T_c_wp,
                R_wp=R_c_wp,
                active_wp=active_c_wp,
                bc_mask_wp=bc_mask_c_wp,
                bc_values_wp=bc_values_c_wp,
                gh_mask_wp=gh_mask_c_wp,
                gh_head_wp=gh_head_c_wp,
                gh_width_wp=gh_width_c_wp,
                M_inv_wp=M_inv_c_wp,
                storage_wp=storage_wp,
                x_wp=x_wp,
                b_wp=b_wp,
                r_wp=r_wp,
                Ax_wp=Ax_wp,
                e_wp=e_wp,
                z_wp=z_wp,
                p_wp=p_wp,
                Ap_wp=Ap_wp,
                rTr_buf=rTr_buf,
                rho_buf=rho_buf,
                rho_new_buf=rho_new_buf,
                pAp_buf=pAp_buf,
                alpha_buf=alpha_buf,
                beta_buf=beta_buf,
                converged_flag=converged_flag,
                x_prev_wp=x_prev_wp,
                dh_max_buf=dh_max_buf,
            )

            levels.append(coarse)

        self.mg_levels = levels

    def _mg_make_level_from_existing_fine(self, device: str):
        nx = int(self.nx)
        ny = int(self.ny)

        # Use the exact host storage (no implicit casting to avoid creating
        # separate arrays). Host arrays were created with NP_FLOAT in
        # build_from_truth_inputs, so they already have the correct dtype.
        T0 = self.T_field_host
        R0 = self.R_field_host
        active0 = np.asarray(self.active_host, dtype=np.int32)
        bc_mask0 = np.asarray(self.bc_mask_host, dtype=np.int32)
        bc_values0 = np.asarray(self.bc_values_host, dtype=NP_FLOAT)

        if self.use_ghb:
            gh_mask0 = np.asarray(self.gh_mask_host, dtype=np.int32)
            gh_head0 = np.asarray(self.gh_head_host, dtype=NP_FLOAT)
            gh_width0 = np.asarray(self.gh_width_host, dtype=NP_FLOAT)
        else:
            gh_mask0 = None
            gh_head0 = None
            gh_width0 = None

        n_active0 = int(np.count_nonzero(active0))

        T0_wp = self.T_wp
        R0_wp = self.R_wp
        active0_wp = self.active_wp
        bc_mask0_wp = self.bc_mask_wp
        bc_values0_wp = self.bc_values_wp

        gh_mask0_wp = self.gh_mask_wp if self.use_ghb else None
        gh_head0_wp = self.gh_head_wp if self.use_ghb else None
        gh_width0_wp = self.gh_width_wp if self.use_ghb else None

        if self.M_inv_wp is None:
            M_inv0_host = build_diag_preconditioner(
                T_field=T0,
                active=active0,
                bc_mask=bc_mask0,
                gh_mask=gh_mask0 if self.use_ghb else None,
                gh_width=gh_width0 if self.use_ghb else None,
                dx=float(self.dx) if self.use_ghb else None,
                gh_alpha=float(self.gh_alpha),
                aq_thickness=float(self.aq_thickness),
            )
            M_inv0_wp = wp.array(M_inv0_host, dtype=WP_FLOAT, device=device)
            self.M_inv_wp = M_inv0_wp
        else:
            M_inv0_wp = self.M_inv_wp

        shape0 = (ny, nx)
        storage_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        x_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        b_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        r_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        Ax_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        e_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)

        z_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        p_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        Ap_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)

        rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
        rho_buf = wp.zeros(1, dtype=wp.float64, device=device)
        rho_new_buf = wp.zeros(1, dtype=wp.float64, device=device)
        pAp_buf = wp.zeros(1, dtype=wp.float64, device=device)
        alpha_buf = wp.zeros(1, dtype=wp.float64, device=device)
        beta_buf = wp.zeros(1, dtype=wp.float64, device=device)
        converged_flag = wp.zeros(1, dtype=wp.int32, device=device)
        x_prev_wp = wp.zeros(shape0, dtype=WP_FLOAT, device=device)
        dh_max_buf = wp.zeros(1, dtype=wp.float64, device=device)

        return self._MGLevel(
            level_id=0,
            nx=int(nx),
            ny=int(ny),
            dx=float(self.dx),
            n_active=int(n_active0),
            T_host=T0,
            R_host=R0,
            active_host=active0,
            bc_mask_host=bc_mask0,
            bc_values_host=bc_values0,
            gh_mask_host=gh_mask0,
            gh_head_host=gh_head0,
            gh_width_host=gh_width0,
            T_wp=T0_wp,
            R_wp=R0_wp,
            active_wp=active0_wp,
            bc_mask_wp=bc_mask0_wp,
            bc_values_wp=bc_values0_wp,
            gh_mask_wp=gh_mask0_wp,
            gh_head_wp=gh_head0_wp,
            gh_width_wp=gh_width0_wp,
            M_inv_wp=M_inv0_wp,
            storage_wp=storage_wp,
            x_wp=x_wp,
            b_wp=b_wp,
            r_wp=r_wp,
            Ax_wp=Ax_wp,
            e_wp=e_wp,
            z_wp=z_wp,
            p_wp=p_wp,
            Ap_wp=Ap_wp,
            rTr_buf=rTr_buf,
            rho_buf=rho_buf,
            rho_new_buf=rho_new_buf,
            pAp_buf=pAp_buf,
            alpha_buf=alpha_buf,
            beta_buf=beta_buf,
            converged_flag=converged_flag,
            x_prev_wp=x_prev_wp,
            dh_max_buf=dh_max_buf,
        )

    def _mg_coarsen_host_any(
        self,
        T_f: np.ndarray,
        R_f: np.ndarray,
        active_f: np.ndarray,
        bc_mask_f: np.ndarray,
        bc_values_f: np.ndarray,
        gh_mask_f,
        gh_head_f,
        gh_width_f,
        dx_c: float,
    ):
        """
        Odd-safe 2:1 coarsening via padding to even and block operations.
        Uses numpy vectorization, no lambdas.

        Returns host arrays for the coarse level.
        """
        ny_f, nx_f = T_f.shape
        pad_y = int(ny_f & 1)
        pad_x = int(nx_f & 1)

        pad_spec = ((0, pad_y), (0, pad_x))

        T_pad = np.pad(np.asarray(T_f, dtype=NP_FLOAT), pad_spec, mode="edge")
        R_pad = np.pad(np.asarray(R_f, dtype=NP_FLOAT), pad_spec, mode="edge")
        active_pad = np.pad(np.asarray(active_f, dtype=np.int32), pad_spec, mode="edge")
        bc_mask_pad = np.pad(np.asarray(bc_mask_f, dtype=np.int32), pad_spec, mode="edge")
        bc_values_pad = np.pad(np.asarray(bc_values_f, dtype=NP_FLOAT), pad_spec, mode="edge")

        ny_p, nx_p = T_pad.shape
        ny_c = ny_p // 2
        nx_c = nx_p // 2

        T_blk = T_pad.reshape(ny_c, 2, nx_c, 2)
        R_blk = R_pad.reshape(ny_c, 2, nx_c, 2)
        T_c = T_blk.mean(axis=(1, 3), dtype=np.float64).astype(NP_FLOAT, copy=False)
        R_c = R_blk.mean(axis=(1, 3), dtype=np.float64).astype(NP_FLOAT, copy=False)

        a_blk = active_pad.reshape(ny_c, 2, nx_c, 2)
        m_blk = bc_mask_pad.reshape(ny_c, 2, nx_c, 2)
        active_c = a_blk.max(axis=(1, 3)).astype(np.int32, copy=False)
        bc_mask_c = m_blk.max(axis=(1, 3)).astype(np.int32, copy=False)

        bc_values_c = np.zeros((ny_c, nx_c), dtype=NP_FLOAT)

        if self.use_ghb and (gh_mask_f is not None) and (gh_width_f is not None) and (gh_head_f is not None):
            gh_mask_pad = np.pad(np.asarray(gh_mask_f, dtype=np.int32), pad_spec, mode="edge")
            gh_width_pad = np.pad(np.asarray(gh_width_f, dtype=NP_FLOAT), pad_spec, mode="edge")
            gh_head_pad = np.pad(np.asarray(gh_head_f, dtype=NP_FLOAT), pad_spec, mode="edge")

            ghm_blk = gh_mask_pad.reshape(ny_c, 2, nx_c, 2)
            gh_mask_c = ghm_blk.max(axis=(1, 3)).astype(np.int32, copy=False)

            ghm_f = ghm_blk.astype(np.float64, copy=False)
            ghw_blk = gh_width_pad.reshape(ny_c, 2, nx_c, 2)

            wsum = (ghw_blk * ghm_f).sum(axis=(1, 3), dtype=np.float64)
            msum = ghm_f.sum(axis=(1, 3), dtype=np.float64)

            gh_width_c = np.full((ny_c, nx_c), NP_FLOAT(dx_c), dtype=NP_FLOAT)
            on = msum > 0.0
            gh_width_c[on] = (wsum[on] / msum[on]).astype(NP_FLOAT, copy=False)

            gh_head_c = np.zeros((ny_c, nx_c), dtype=NP_FLOAT)

            gh_mask_c, gh_width_c, gh_head_c = self._mg_sanitize_ghb_level_host(
                active=active_c,
                bc_mask=bc_mask_c,
                gh_mask=gh_mask_c,
                gh_width=gh_width_c,
                gh_head=gh_head_c,
                dx=float(dx_c),
            )
        else:
            gh_mask_c = None
            gh_head_c = None
            gh_width_c = None

        return T_c, R_c, active_c, bc_mask_c, bc_values_c, gh_mask_c, gh_head_c, gh_width_c

    def _mg_sanitize_ghb_level_host(
        self,
        active: np.ndarray,
        bc_mask: np.ndarray,
        gh_mask: np.ndarray,
        gh_width: np.ndarray,
        gh_head: np.ndarray,
        dx: float,
    ):
        """
        Same idea as _sanitize_ghb_host_fields, but operates on a provided level.
        """
        dx_f = float(dx)
        eps = NP_FLOAT(max(1.0e-8 * dx_f, 1.0e-12))

        gh_mask2 = np.asarray(gh_mask, dtype=np.int32).copy()
        gh_width2 = np.asarray(gh_width, dtype=NP_FLOAT).copy()
        gh_head2 = np.asarray(gh_head, dtype=NP_FLOAT).copy()

        gh_mask2[np.asarray(active, dtype=np.int32) == 0] = 0
        gh_mask2[np.asarray(bc_mask, dtype=np.int32) != 0] = 0

        bad_w = ~np.isfinite(gh_width2)
        if np.any(bad_w):
            gh_width2[bad_w] = NP_FLOAT(dx_f)

        gh_width2[gh_mask2 == 0] = NP_FLOAT(dx_f)

        on = gh_mask2 != 0
        if np.any(on):
            gh_width2[on] = np.maximum(gh_width2[on], eps)

        bad_h = ~np.isfinite(gh_head2)
        if np.any(bad_h):
            gh_head2[bad_h] = NP_FLOAT(0.0)

        return gh_mask2, gh_width2, gh_head2

    # -------------------------------------------------------------------------
    # GHB sanitization
    # -------------------------------------------------------------------------
    def _sanitize_ghb_host_fields(self) -> None:
        """
        Make GHB inputs numerically safe:
          - gh_mask is zeroed on inactive and Dirichlet cells
          - gh_width is finite and strictly positive where gh_mask != 0
          - gh_width is set to dx where gh_mask == 0
          - gh_head is finite
        """
        if self.gh_mask_host is None or self.gh_width_host is None:
            return

        dx = float(self.dx)
        eps = NP_FLOAT(max(1.0e-8 * dx, 1.0e-12))

        gh_mask = np.asarray(self.gh_mask_host, dtype=np.int32)
        gh_width = np.asarray(self.gh_width_host, dtype=NP_FLOAT)

        if gh_mask.shape != gh_width.shape:
            raise RuntimeError(f"GHB arrays shape mismatch: mask {gh_mask.shape} width {gh_width.shape}")

        if self.active_host is not None:
            gh_mask = gh_mask.copy()
            gh_mask[self.active_host == 0] = 0

        if self.bc_mask_host is not None:
            gh_mask = gh_mask.copy()
            gh_mask[self.bc_mask_host != 0] = 0

        gh_width = gh_width.copy()
        bad_w = ~np.isfinite(gh_width)
        if np.any(bad_w):
            gh_width[bad_w] = NP_FLOAT(dx)

        gh_width[gh_mask == 0] = NP_FLOAT(dx)

        gh_on = gh_mask != 0
        if np.any(gh_on):
            gh_width[gh_on] = np.maximum(gh_width[gh_on], eps)

        self.gh_mask_host = gh_mask
        self.gh_width_host = gh_width

        if self.gh_head_host is not None:
            gh_head = np.asarray(self.gh_head_host, dtype=NP_FLOAT).copy()
            bad_h = ~np.isfinite(gh_head)
            if np.any(bad_h):
                gh_head[bad_h] = NP_FLOAT(0.0)
            self.gh_head_host = gh_head


    def _prune_isolated_active_host_cells(self) -> None:
        """
        Mark active cells with zero coupling as inactive to avoid singular rows.
        """
        if self.active_host is None or self.T_field_host is None:
            return

        active = np.asarray(self.active_host, dtype=np.int32)
        bc_mask = (
            np.asarray(self.bc_mask_host, dtype=np.int32)
            if self.bc_mask_host is not None
            else np.zeros_like(active, dtype=np.int32)
        )
        T = np.asarray(self.T_field_host, dtype=np.float64)

        ny, nx = T.shape
        tiny = np.float64(1.0e-12)

        act = active != 0
        T_pos = np.isfinite(T) & (T > 0.0)

        sum_T = np.zeros((ny, nx), dtype=np.float64)

        if nx > 1:
            T_L = T[:, :-1].astype(np.float64, copy=False)
            T_R = T[:, 1:].astype(np.float64, copy=False)
            denom_E = T_L + T_R
            valid_E = act[:, :-1] & act[:, 1:] & T_pos[:, :-1] & T_pos[:, 1:] & (denom_E > tiny)
            cond_E = np.zeros_like(denom_E, dtype=np.float64)
            cond_E[valid_E] = 2.0 * T_L[valid_E] * T_R[valid_E] / denom_E[valid_E]
            sum_T[:, :-1] += cond_E
            sum_T[:, 1:] += cond_E

        if ny > 1:
            T_T = T[:-1, :].astype(np.float64, copy=False)
            T_B = T[1:, :].astype(np.float64, copy=False)
            denom_S = T_T + T_B
            valid_S = act[:-1, :] & act[1:, :] & T_pos[:-1, :] & T_pos[1:, :] & (denom_S > tiny)
            cond_S = np.zeros_like(denom_S, dtype=np.float64)
            cond_S[valid_S] = 2.0 * T_T[valid_S] * T_B[valid_S] / denom_S[valid_S]
            sum_T[:-1, :] += cond_S
            sum_T[1:, :] += cond_S

        if self.use_ghb and self.gh_mask_host is not None and self.gh_width_host is not None:
            gh_mask = np.asarray(self.gh_mask_host, dtype=np.int32)
            gh_width = np.asarray(self.gh_width_host, dtype=np.float64)
            width_ok = np.isfinite(gh_width) & (gh_width > 0.0)
            gh_on = (gh_mask != 0) & width_ok & T_pos & act
            if np.any(gh_on):
                C_gh = (
                    np.float64(self.gh_alpha)
                    * T
                    / np.float64(self.aq_thickness)
                    * gh_width
                    * np.float64(self.dx)
                )
                sum_T[gh_on] += C_gh[gh_on]

        isolated = act & (bc_mask == 0) & (sum_T <= tiny)
        if not np.any(isolated):
            return

        active = active.copy()
        active[isolated] = 0
        self.active_host = active

        if self.T_field_host is not None:
            T_host = np.asarray(self.T_field_host, dtype=NP_FLOAT).copy()
            T_host[isolated] = NP_FLOAT(0.0)
            self.T_field_host = T_host

        if self.R_field_host is not None:
            R_host = np.asarray(self.R_field_host, dtype=NP_FLOAT).copy()
            R_host[isolated] = NP_FLOAT(0.0)
            self.R_field_host = R_host

        if self.bc_values_host is not None:
            bc_vals = np.asarray(self.bc_values_host, dtype=NP_FLOAT).copy()
            bc_vals[isolated] = NP_FLOAT(0.0)
            self.bc_values_host = bc_vals

        if self.gh_mask_host is not None:
            gh_mask = np.asarray(self.gh_mask_host, dtype=np.int32).copy()
            gh_mask[isolated] = 0
            self.gh_mask_host = gh_mask

        if self.gh_head_host is not None:
            gh_head = np.asarray(self.gh_head_host, dtype=NP_FLOAT).copy()
            gh_head[isolated] = NP_FLOAT(0.0)
            self.gh_head_host = gh_head

        if self.gh_width_host is not None:
            gh_width = np.asarray(self.gh_width_host, dtype=NP_FLOAT).copy()
            gh_width[isolated] = NP_FLOAT(0.0)
            self.gh_width_host = gh_width

        self._n_isolated_pruned = int(np.count_nonzero(isolated))

    # -------------------------------------------------------------------------
    # Build and upload
    # -------------------------------------------------------------------------
    def build_from_truth_inputs(
        self,
        T_truth,
        R_truth,
        gh_alpha: float = 1.0,
        width: float = None,
    ):
        """
        Build FD style inpts from T_truth and R_truth and upload to device.

        :param T_truth: scalar or array transmissivity
        :param R_truth: scalar or array recharge
        :param gh_alpha: GHB scaling factor
        """
        if width is None:
            width = float(self.dx)
        self.gh_alpha = float(gh_alpha)

        (
            T_field,
            R_field,
            active,
            bc_mask,
            bc_values,
            gh_mask,
            gh_head,
            gh_width,
        ) = build_truth_inputs(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            T_truth=T_truth,
            R_truth=R_truth,
            use_ghb=self.use_ghb,
            width=width
        )

        # Keep host arrays in the package float dtype (NP_FLOAT) so single-precision
        # vs double-precision modes remain consistent.
        self.T_field_host = np.asarray(T_field, dtype=NP_FLOAT)
        self.R_field_host = np.asarray(R_field, dtype=NP_FLOAT)
        self.active_host = np.asarray(active, dtype=np.int32)
        self.bc_mask_host = np.asarray(bc_mask, dtype=np.int32)
        self.bc_values_host = np.asarray(bc_values, dtype=NP_FLOAT)
        self.gh_mask_host = np.asarray(gh_mask, dtype=np.int32)
        self.gh_head_host = np.asarray(gh_head, dtype=NP_FLOAT)
        self.gh_width_host = np.asarray(gh_width, dtype=NP_FLOAT)

        self._prune_isolated_active_host_cells()
        self.n_active = int(np.count_nonzero(self.active_host))

        device = self.device_str

        self._sanitize_ghb_host_fields()

        self.T_wp = wp.array(self.T_field_host, dtype=WP_FLOAT, device=device)
        self.R_wp = wp.array(self.R_field_host, dtype=WP_FLOAT, device=device)
        self.active_wp = wp.array(self.active_host, dtype=wp.int32, device=device)
        self.bc_mask_wp = wp.array(self.bc_mask_host, dtype=wp.int32, device=device)
        self.bc_values_wp = wp.array(self.bc_values_host, dtype=WP_FLOAT, device=device)

        # Always allocate GHB arrays and pass them to kernels (mask is zero if unused)
        self.gh_mask_wp = wp.array(self.gh_mask_host, dtype=wp.int32, device=device)
        self.gh_head_wp = wp.array(self.gh_head_host, dtype=WP_FLOAT, device=device)
        self.gh_width_wp = wp.array(self.gh_width_host, dtype=WP_FLOAT, device=device)

        # Create CPU-stage warp views that wrap the numpy host arrays. These
        # are reused by update_T_in_place to avoid allocating a new wp.array
        # on every update.
        self._stage_T0_host = self.T_field_host
        self._stage_T0 = wp.array(self._stage_T0_host, dtype=WP_FLOAT, device="cpu")
        self._stage_R0_host = self.R_field_host
        self._stage_R0 = wp.array(self._stage_R0_host, dtype=WP_FLOAT, device="cpu")

        # Mark operator dirty (because T changed)
        self._operator_dirty = True

        M_inv_host = build_diag_preconditioner(
            T_field=self.T_field_host,
            active=self.active_host,
            bc_mask=self.bc_mask_host,
            gh_mask=self.gh_mask_host if self.use_ghb else None,
            gh_width=self.gh_width_host if self.use_ghb else None,
            dx=float(self.dx) if self.use_ghb else None,
            gh_alpha=float(self.gh_alpha),
            aq_thickness=float(self.aq_thickness),
        )
        self.M_inv_wp = wp.array(M_inv_host, dtype=WP_FLOAT, device=device)
        self.storage_wp = wp.zeros((int(self.ny), int(self.nx)), dtype=WP_FLOAT, device=device)

        # CPU-stage for M_inv (wrap host memory for in-place copies later)
        self._stage_M0_host = M_inv_host
        self._stage_M0 = wp.array(self._stage_M0_host, dtype=WP_FLOAT, device="cpu")

        self._ensure_pcg_buffers_fine(device=device)

        self._build_two_level_cache()

        self._fine_level = self._GridLevel(
            T_wp=self.T_wp,
            active_wp=self.active_wp,
            bc_mask_wp=self.bc_mask_wp,
            gh_mask_wp=self.gh_mask_wp,
            gh_width_wp=self.gh_width_wp,
            M_inv_wp=self.M_inv_wp,
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
        )
        if self.mg_cache_built:
            self._coarse_level = self._GridLevel(
                T_wp=self.T_c_wp,
                active_wp=self.active_c_wp,
                bc_mask_wp=self.bc_mask_c_wp,
                gh_mask_wp=self.gh_mask_c_wp,
                gh_width_wp=self.gh_width_c_wp,
                M_inv_wp=self.M_inv_c_wp,
                nx=self.nx_c,
                ny=self.ny_c,
                dx=self.dx_c,
            )

        self._mg_work_built = False
        self._mg_work = {}
        self._ensure_mg_work_buffers(device=device)
        self._kcycle_spatial_alpha_active = False
        self._kcycle_transient_active = False
        self._kcycle_last_gh_width_eff_levels = None

    def build_from_fields(
        self,
        T_field: np.ndarray,
        R_field: np.ndarray,
        active: np.ndarray,
        bc_mask: np.ndarray,
        bc_values: np.ndarray,
        gh_mask: np.ndarray | None = None,
        gh_head: np.ndarray | None = None,
        gh_width: np.ndarray | None = None,
        gh_alpha: float = 1.0,
    ) -> None:
        """
        Build solver state from explicitly provided fields (no synthetic builder).
        """
        self.gh_alpha = float(gh_alpha)

        T_field = np.asarray(T_field, dtype=NP_FLOAT)
        R_field = np.asarray(R_field, dtype=NP_FLOAT)
        active = np.asarray(active, dtype=np.int32)
        bc_mask = np.asarray(bc_mask, dtype=np.int32)
        bc_values = np.asarray(bc_values, dtype=NP_FLOAT)

        if T_field.shape != (self.ny, self.nx):
            raise ValueError(f"T_field shape {T_field.shape} expected {(self.ny, self.nx)}")
        if R_field.shape != (self.ny, self.nx):
            raise ValueError(f"R_field shape {R_field.shape} expected {(self.ny, self.nx)}")
        if active.shape != (self.ny, self.nx):
            raise ValueError(f"active shape {active.shape} expected {(self.ny, self.nx)}")
        if bc_mask.shape != (self.ny, self.nx):
            raise ValueError(f"bc_mask shape {bc_mask.shape} expected {(self.ny, self.nx)}")
        if bc_values.shape != (self.ny, self.nx):
            raise ValueError(f"bc_values shape {bc_values.shape} expected {(self.ny, self.nx)}")

        if self.use_ghb:
            if gh_mask is None or gh_head is None or gh_width is None:
                raise ValueError("gh_mask, gh_head, gh_width are required when use_ghb=True.")
            gh_mask = np.asarray(gh_mask, dtype=np.int32)
            gh_head = np.asarray(gh_head, dtype=NP_FLOAT)
            gh_width = np.asarray(gh_width, dtype=NP_FLOAT)
            if gh_mask.shape != (self.ny, self.nx):
                raise ValueError(f"gh_mask shape {gh_mask.shape} expected {(self.ny, self.nx)}")
            if gh_head.shape != (self.ny, self.nx):
                raise ValueError(f"gh_head shape {gh_head.shape} expected {(self.ny, self.nx)}")
            if gh_width.shape != (self.ny, self.nx):
                raise ValueError(f"gh_width shape {gh_width.shape} expected {(self.ny, self.nx)}")
        else:
            gh_mask = np.zeros((self.ny, self.nx), dtype=np.int32)
            gh_head = np.zeros((self.ny, self.nx), dtype=NP_FLOAT)
            gh_width = np.zeros((self.ny, self.nx), dtype=NP_FLOAT)

        self.T_field_host = T_field
        self.R_field_host = R_field
        self.active_host = active
        self.bc_mask_host = bc_mask
        self.bc_values_host = bc_values
        self.gh_mask_host = gh_mask
        self.gh_head_host = gh_head
        self.gh_width_host = gh_width

        self._prune_isolated_active_host_cells()
        self.n_active = int(np.count_nonzero(self.active_host))

        device = self.device_str

        self._sanitize_ghb_host_fields()

        self.T_wp = wp.array(self.T_field_host, dtype=WP_FLOAT, device=device)
        self.R_wp = wp.array(self.R_field_host, dtype=WP_FLOAT, device=device)
        self.active_wp = wp.array(self.active_host, dtype=wp.int32, device=device)
        self.bc_mask_wp = wp.array(self.bc_mask_host, dtype=wp.int32, device=device)
        self.bc_values_wp = wp.array(self.bc_values_host, dtype=WP_FLOAT, device=device)

        self.gh_mask_wp = wp.array(self.gh_mask_host, dtype=wp.int32, device=device)
        self.gh_head_wp = wp.array(self.gh_head_host, dtype=WP_FLOAT, device=device)
        self.gh_width_wp = wp.array(self.gh_width_host, dtype=WP_FLOAT, device=device)

        self._stage_T0_host = self.T_field_host
        self._stage_T0 = wp.array(self._stage_T0_host, dtype=WP_FLOAT, device="cpu")

        self._stage_R0_host = self.R_field_host
        self._stage_R0 = wp.array(self._stage_R0_host, dtype=WP_FLOAT, device="cpu")

        self._operator_dirty = True

        M_inv_host = build_diag_preconditioner(
            T_field=self.T_field_host,
            active=self.active_host,
            bc_mask=self.bc_mask_host,
            gh_mask=self.gh_mask_host if self.use_ghb else None,
            gh_width=self.gh_width_host if self.use_ghb else None,
            dx=float(self.dx) if self.use_ghb else None,
            gh_alpha=float(self.gh_alpha),
            aq_thickness=float(self.aq_thickness),
        )
        self.M_inv_wp = wp.array(M_inv_host, dtype=WP_FLOAT, device=device)
        self.storage_wp = wp.zeros((int(self.ny), int(self.nx)), dtype=WP_FLOAT, device=device)

        self._stage_M0_host = M_inv_host
        self._stage_M0 = wp.array(self._stage_M0_host, dtype=WP_FLOAT, device="cpu")

        self._ensure_pcg_buffers_fine(device=device)

        self._build_two_level_cache()

        self._fine_level = self._GridLevel(
            T_wp=self.T_wp,
            active_wp=self.active_wp,
            bc_mask_wp=self.bc_mask_wp,
            gh_mask_wp=self.gh_mask_wp,
            gh_width_wp=self.gh_width_wp,
            M_inv_wp=self.M_inv_wp,
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
        )
        if self.mg_cache_built:
            self._coarse_level = self._GridLevel(
                T_wp=self.T_c_wp,
                active_wp=self.active_c_wp,
                bc_mask_wp=self.bc_mask_c_wp,
                gh_mask_wp=self.gh_mask_c_wp,
                gh_width_wp=self.gh_width_c_wp,
                M_inv_wp=self.M_inv_c_wp,
                nx=self.nx_c,
                ny=self.ny_c,
                dx=self.dx_c,
            )

        self._mg_work_built = False
        self._mg_work = {}
        self._ensure_mg_work_buffers(device=device)
        self._kcycle_spatial_alpha_active = False
        self._kcycle_transient_active = False
        self._kcycle_last_gh_width_eff_levels = None

    def _ensure_pcg_buffers_fine(self, device: str) -> None:
        """
        Ensure fine-level PCG buffers exist and match current fine grid size.
        """
        shape = (int(self.ny), int(self.nx))

        need = False
        if self.x_wp is None:
            need = True
        else:
            try:
                if tuple(self.x_wp.shape) != tuple(shape):
                    need = True
            except Exception:
                need = True

        if need:
            self.b_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
            self.x_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
            self.r_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
            self.z_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
            self.p_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
            self.Ap_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)

            self.rho_buf = wp.zeros(1, dtype=wp.float64, device=device)
            self.rho_new_buf = wp.zeros(1, dtype=wp.float64, device=device)
            self.rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
            self.pAp_buf = wp.zeros(1, dtype=wp.float64, device=device)
            self.alpha_buf = wp.zeros(1, dtype=wp.float64, device=device)
            self.beta_buf = wp.zeros(1, dtype=wp.float64, device=device)
            self.converged_flag = wp.zeros(1, dtype=wp.int32, device=device)

        if self.storage_wp is None:
            self.storage_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
        else:
            try:
                if tuple(self.storage_wp.shape) != tuple(shape):
                    self.storage_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
            except Exception:
                self.storage_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)

    # -------------------------------------------------------------------------
    # Coarse cache build (2-level)
    # -------------------------------------------------------------------------
    def _build_two_level_cache(self):
        """
        Build and cache a 2:1 coarse grid for 2-level multigrid.
        Must be called after build_from_truth_inputs has set fine level host arrays.
        """
        if self.T_field_host is None:
            raise RuntimeError("_build_two_level_cache called before build_from_truth_inputs.")

        device = self.device_str
        self._sanitize_ghb_host_fields()

        (
            T_c_host,
            R_c_host,
            active_c_host,
            bc_mask_c_host,
            bc_values_c_host,
            gh_mask_c_host,
            gh_head_c_host,
            gh_width_c_host,
        ) = build_coarse_level_from_fine(
            T_f=self.T_field_host,
            R_f=self.R_field_host,
            active_f=self.active_host,
            bc_mask_f=self.bc_mask_host,
            bc_values_f=self.bc_values_host,
            gh_mask_f=self.gh_mask_host,
            gh_head_f=self.gh_head_host,
            gh_width_f=self.gh_width_host,
        )

        bc_values_c_host[...] = 0.0
        gh_head_c_host[...] = 0.0

        ny_c, nx_c = T_c_host.shape

        self.nx_c = int(nx_c)
        self.ny_c = int(ny_c)
        self.dx_c = 2.0 * float(self.dx)
        self.n_active_c = int(np.count_nonzero(active_c_host))

        self.T_c_host = T_c_host
        self.R_c_host = R_c_host
        self.active_c_host = active_c_host
        self.bc_mask_c_host = bc_mask_c_host
        self.bc_values_c_host = bc_values_c_host
        self.gh_mask_c_host = gh_mask_c_host
        self.gh_head_c_host = gh_head_c_host
        self.gh_width_c_host = gh_width_c_host

        self.T_c_wp = wp.array(T_c_host, dtype=WP_FLOAT, device=device)
        self.R_c_wp = wp.array(R_c_host, dtype=WP_FLOAT, device=device)
        self.active_c_wp = wp.array(active_c_host, dtype=wp.int32, device=device)
        self.bc_mask_c_wp = wp.array(bc_mask_c_host, dtype=wp.int32, device=device)
        self.bc_values_c_wp = wp.array(bc_values_c_host, dtype=WP_FLOAT, device=device)
        self.gh_mask_c_wp = wp.array(gh_mask_c_host, dtype=wp.int32, device=device)
        self.gh_head_c_wp = wp.array(gh_head_c_host, dtype=WP_FLOAT, device=device)
        self.gh_width_c_wp = wp.array(gh_width_c_host, dtype=WP_FLOAT, device=device)

        M_inv_c_host = build_diag_preconditioner(
            T_field=T_c_host,
            active=active_c_host,
            bc_mask=bc_mask_c_host,
            gh_mask=gh_mask_c_host if self.use_ghb else None,
            gh_width=gh_width_c_host if self.use_ghb else None,
            dx=float(self.dx_c) if self.use_ghb else None,
            gh_alpha=float(self.gh_alpha),
            aq_thickness=float(self.aq_thickness),
        )
        self.M_inv_c_wp = wp.array(M_inv_c_host, dtype=WP_FLOAT, device=device)

        self.mg_cache_built = True



    # -------------------------------------------------------------------------
    # PCG internals
    # -------------------------------------------------------------------------
    def _select_rhs_backend(self) -> str:
        """
        Select RHS assembly backend.

        Environment controls:
          - DARCY_RHS_MODE: "auto" (default), "host", or "device"
          - DARCY_RHS_DEVICE_MIN_CELLS: threshold used when mode="auto"
        """
        mode = str(os.environ.get("DARCY_RHS_MODE", "auto")).strip().lower()
        if mode in {"host", "device"}:
            return mode
        if mode not in {"auto", "host", "device"}:
            mode = "auto"

        raw_thr = str(os.environ.get("DARCY_RHS_DEVICE_MIN_CELLS", "500000")).strip()
        try:
            min_cells = max(1, int(raw_thr))
        except Exception:
            min_cells = 500000

        n_cells = int(self.nx) * int(self.ny)
        return "device" if n_cells >= min_cells else "host"

    def _build_rhs_fine_host(self, b_out_wp, aq_thickness: float) -> None:
        """
        Assemble fine-grid RHS on host and upload via reusable CPU staging.
        """
        if (
            self.T_field_host is None
            or self.R_field_host is None
            or self.active_host is None
            or self.bc_mask_host is None
            or self.bc_values_host is None
        ):
            raise RuntimeError("Host field arrays are not initialized. Call build_from_truth_inputs() first.")

        ny = int(self.ny)
        nx = int(self.nx)
        if b_out_wp is None or tuple(b_out_wp.shape) != (ny, nx):
            raise RuntimeError("RHS destination buffer has wrong shape or is missing.")

        b_host = build_rhs_fd_like(
            T_field=self.T_field_host,
            R_field=self.R_field_host,
            active=self.active_host,
            bc_mask=self.bc_mask_host,
            bc_values=self.bc_values_host,
            dx=float(self.dx),
            gh_mask=self.gh_mask_host,
            gh_head=self.gh_head_host,
            gh_width=self.gh_width_host,
            gh_alpha=float(self.gh_alpha),
            head_scale=self.head_scale,
            aq_thickness=float(aq_thickness),
        ).reshape(ny, nx).astype(NP_FLOAT, copy=False)

        if (
            not hasattr(self, "_kcycle_stage_b")
            or self._kcycle_stage_b is None
            or tuple(self._kcycle_stage_b.shape) != (ny, nx)
        ):
            self._kcycle_stage_b = wp.zeros((ny, nx), dtype=WP_FLOAT, device="cpu")

        stage_b_np = self._kcycle_stage_b.numpy()
        stage_b_np[...] = b_host
        wp.copy(b_out_wp, self._kcycle_stage_b)

    def _build_rhs_fine_device(self, b_out_wp, aq_thickness: float) -> None:
        """
        Assemble fine-grid RHS directly on device, avoiding host staging.
        """
        if (
            self.T_wp is None
            or self.R_wp is None
            or self.active_wp is None
            or self.bc_mask_wp is None
            or self.bc_values_wp is None
            or self.gh_mask_wp is None
            or self.gh_head_wp is None
            or self.gh_width_wp is None
        ):
            raise RuntimeError("Field/device arrays are not initialized. Call build_from_truth_inputs() first.")

        ny = int(self.ny)
        nx = int(self.nx)
        if b_out_wp is None or tuple(b_out_wp.shape) != (ny, nx):
            raise RuntimeError("RHS destination buffer has wrong shape or is missing.")

        wp.launch(
            kernel=build_rhs_kernel,
            dim=(ny, nx),
            inputs=[
                self.T_wp,
                self.R_wp,
                self.active_wp,
                self.bc_mask_wp,
                self.bc_values_wp,
                self.gh_mask_wp,
                self.gh_head_wp,
                self.gh_width_wp,
                nx,
                ny,
                float(self.dx),
                float(self.gh_alpha),
                float(self.head_scale),
                float(aq_thickness),
                b_out_wp,
            ],
            device=self.device_str,
        )

    def _build_rhs_fine(self, b_out_wp, aq_thickness: float) -> None:
        """
        Assemble fine-grid RHS using configured backend.
        """
        backend = self._select_rhs_backend()
        if backend == "device":
            self._build_rhs_fine_device(b_out_wp, aq_thickness=float(aq_thickness))
        else:
            self._build_rhs_fine_host(b_out_wp, aq_thickness=float(aq_thickness))

    def _pcg_build_rhs_and_upload(self, aq_thickness: float) -> None:
        """
        Build RHS for PCG backend.
        """
        self._ensure_pcg_buffers_fine(device=self.device_str)
        self._build_rhs_fine(self.b_wp, aq_thickness=float(aq_thickness))

    def _pcg_initialize_guess_and_upload(self, initial_head: np.ndarray | None) -> None:
        """
        Initialize scaled x0 and upload to device for PCG backend.
        """
        device = self.device_str
        nx = int(self.nx)
        ny = int(self.ny)

        head_scale = float(self.head_scale)

        if initial_head is not None:
            x0_phys = np.asarray(initial_head, dtype=np.float64).copy()
            x0 = x0_phys / head_scale
        else:
            x0 = np.zeros((ny, nx), dtype=np.float64)

        bc_idx = self.bc_mask_host != 0
        x0[bc_idx] = self.bc_values_host[bc_idx] / head_scale
        x0[self.active_host == 0] = np.float64(0.0)

        self.x_wp = wp.array(x0, dtype=WP_FLOAT, device=device)

    def _pcg_reset_work_vectors(self) -> None:
        """
        Reset fine-level PCG vectors and scalars on device.
        """
        self.r_wp.fill_(0.0)
        self.z_wp.fill_(0.0)
        self.p_wp.fill_(0.0)
        self.Ap_wp.fill_(0.0)

        self.rho_buf.fill_(0.0)
        self.rho_new_buf.fill_(0.0)
        self.rTr_buf.fill_(0.0)
        self.pAp_buf.fill_(0.0)
        self.alpha_buf.fill_(0.0)
        self.beta_buf.fill_(0.0)
        self.converged_flag.fill_(0)

    def _solve_pcg_device_loop(
        self,
        max_iter: int,
        rel_tol: float,
        abs_tol_min: float,
        initial_head: np.ndarray | None,
        aq_thickness: float,
    ):
        """
        Internal PCG solve. Residuals and norms computed on GPU.
        CPU orchestrates kernel launches and reads convergence flag.
        """
        if self.T_field_host is None:
            raise RuntimeError("build_from_truth_inputs must be called before solve().")

        device = self.device_str
        nx = int(self.nx)
        ny = int(self.ny)
        dx = float(self.dx)

        self._pcg_build_rhs_and_upload(aq_thickness=aq_thickness)
        self._pcg_initialize_guess_and_upload(initial_head=initial_head)
        self._pcg_reset_work_vectors()

        dim = (ny, nx)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.rho_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.rTr_buf], device=device)

        wp.launch(
            kernel=init_pcg_with_A_kernel,
            dim=dim,
            inputs=[
                self.x_wp,
                self.b_wp,
                self.T_wp,
                self.active_wp,
                self.bc_mask_wp,
                self.gh_mask_wp,
                self.gh_width_wp,
                self.storage_wp,
                self.M_inv_wp,
                self.Ap_wp,
                self.r_wp,
                self.z_wp,
                self.p_wp,
                self.rho_buf,
                self.rTr_buf,
                nx,
                ny,
                float(dx),
                float(self.gh_alpha),
                float(aq_thickness),
            ],
            device=device,
        )

        wp.launch(
            enforce_constraints_kernel,
            dim=dim,
            inputs=[
                self.x_wp, self.r_wp, self.z_wp, self.p_wp,
                self.active_wp, self.bc_mask_wp, self.bc_values_wp,
                float(self.head_scale),
            ],
            device=self.device_str,
        )

        rTr0 = float(self.rTr_buf.numpy()[0])
        if self.n_active > 0 and rTr0 > 0.0:
            r_rms0_scaled = float(np.sqrt(rTr0 / float(self.n_active)))
        else:
            r_rms0_scaled = 0.0

        r_rms0_phys = r_rms0_scaled * float(self.head_scale)

        abs_tol_scaled = float(abs_tol_min) / float(self.head_scale)
        tol_abs_scaled = max(abs_tol_scaled, float(rel_tol) * r_rms0_scaled)

        n_iter_used = 0
        converged = False

        for it in range(int(max_iter)):
            n_iter_used = it + 1

            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.pAp_buf], device=device)
            wp.launch(
                kernel=apply_A_and_pAp_kernel,
                dim=dim,
                inputs=[
                    self.T_wp,
                    self.active_wp,
                    self.bc_mask_wp,
                    self.gh_mask_wp,
                    self.gh_width_wp,
                    self.storage_wp,
                    self.p_wp,
                    self.Ap_wp,
                    self.pAp_buf,
                    nx,
                    ny,
                    float(dx),
                    float(self.gh_alpha),
                    float(aq_thickness),
                ],
                device=device,
            )

            wp.launch(
                kernel=compute_alpha_kernel,
                dim=1,
                inputs=[self.rho_buf, self.pAp_buf, self.alpha_buf],
                device=device,
            )

            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.rho_new_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.rTr_buf], device=device)

            wp.launch(
                kernel=update_x_r_z_rho_rTr_kernel,
                dim=dim,
                inputs=[
                    self.x_wp,
                    self.r_wp,
                    self.z_wp,
                    self.p_wp,
                    self.Ap_wp,
                    self.M_inv_wp,
                    self.active_wp,
                    self.bc_mask_wp,
                    self.alpha_buf,
                    self.rho_new_buf,
                    self.rTr_buf,
                    nx,
                    ny,
                ],
                device=device,
            )

            wp.launch(
                kernel=check_convergence_kernel,
                dim=1,
                inputs=[
                    self.rTr_buf,
                    int(self.n_active),
                    float(tol_abs_scaled),
                    self.converged_flag,
                ],
                device=device,
            )

            if int(self.converged_flag.numpy()[0]) == 1:
                converged = True
                break

            wp.launch(
                kernel=compute_beta_and_update_rho_kernel,
                dim=1,
                inputs=[self.rho_buf, self.rho_new_buf, self.beta_buf],
                device=device,
            )

            wp.launch(
                kernel=update_p_kernel,
                dim=dim,
                inputs=[
                    self.p_wp,
                    self.z_wp,
                    self.active_wp,
                    self.bc_mask_wp,
                    self.beta_buf,
                    nx,
                    ny,
                ],
                device=device,
            )

            # Constraints are preserved by masked updates in PCG kernels.

        head_scaled = self.x_wp.numpy()
        head = head_scaled * float(self.head_scale)

        rTr_final = float(self.rTr_buf.numpy()[0]) if self.n_active > 0 else 0.0
        if self.n_active > 0 and rTr_final >= 0.0:
            r_rms_final_scaled = float(np.sqrt(rTr_final / float(self.n_active)))
        else:
            r_rms_final_scaled = 0.0

        r_rms_final_phys = r_rms_final_scaled * float(self.head_scale)
        tol_abs_phys = float(tol_abs_scaled) * float(self.head_scale)

        info = {
            "solver_type": "pcg",
            "nx": int(self.nx),
            "ny": int(self.ny),
            "n_cells_total": int(self.nx * self.ny),
            "n_iter_used": int(n_iter_used),
            "max_iter": int(max_iter),
            "converged": bool(converged),
            "rel_tol": float(rel_tol),
            "abs_tol_min_phys": float(abs_tol_min),
            "tol_abs_phys": float(tol_abs_phys),
            "head_scale": float(self.head_scale),
            "rms_res_initial_phys": float(r_rms0_phys),
            "rms_res_final_phys": float(r_rms_final_phys),
        }

        return head, info

    def _solve_amg_host_backend(
        self,
        max_cycles: int,
        rel_tol: float,
        abs_tol_min: float,
        initial_head: np.ndarray | None,
        aq_thickness: float,
        amg_max_iter: int | None,
        amg_cycle: str,
    ):
        """
        Optional host-side AMG solve using pyamg, assembled from the same
        discrete operator as the Warp kernels.
        """
        if self.T_field_host is None:
            raise RuntimeError("build_from_truth_inputs must be called before solve().")

        try:
            import pyamg
        except Exception as exc:
            raise RuntimeError(
                "AMG backend requested but pyamg is unavailable. "
                "Install pyamg (e.g., `pip install pyamg`) or use linear_backend='kcycle'."
            ) from exc

        ny = int(self.ny)
        nx = int(self.nx)
        n_cells = int(nx * ny)

        x0 = np.zeros((ny, nx), dtype=np.float64)
        if initial_head is not None:
            init_arr = np.asarray(initial_head, dtype=np.float64)
            if init_arr.shape != (ny, nx):
                raise ValueError(f"initial_head must have shape ({ny}, {nx}), got {init_arr.shape}")
            x0[:, :] = init_arr

        bc_idx = np.asarray(self.bc_mask_host, dtype=np.int32) != 0
        x0[bc_idx] = np.asarray(self.bc_values_host, dtype=np.float64)[bc_idx]
        x0[np.asarray(self.active_host, dtype=np.int32) == 0] = 0.0

        A, b, free_mask_flat = build_sparse_system_fd_like(
            T_field=self.T_field_host,
            R_field=self.R_field_host,
            active=self.active_host,
            bc_mask=self.bc_mask_host,
            bc_values=self.bc_values_host,
            dx=float(self.dx),
            gh_mask=self.gh_mask_host if self.use_ghb else None,
            gh_head=self.gh_head_host if self.use_ghb else None,
            gh_width=self.gh_width_host if self.use_ghb else None,
            gh_alpha=float(self.gh_alpha),
            aq_thickness=float(aq_thickness),
        )

        if A.shape != (n_cells, n_cells):
            raise RuntimeError("AMG assembly produced unexpected matrix size.")

        n_free = int(np.count_nonzero(free_mask_flat))
        if n_free <= 0:
            head_out = x0.astype(NP_FLOAT, copy=False)
            info = {"solver_type": "amg", "n_cycles_used": 0, "converged": True}
            return head_out, info

        x0_flat = x0.reshape(-1)

        r0 = b - A.dot(x0_flat)
        r0_free = r0[free_mask_flat]
        r_rms0 = float(np.sqrt(np.dot(r0_free, r0_free) / float(n_free)))
        tol_abs = float(max(abs_tol_min, rel_tol * r_rms0))

        max_iter = int(max_cycles if amg_max_iter is None else amg_max_iter)
        max_iter = max(1, max_iter)

        target_res_norm = float(tol_abs * np.sqrt(float(n_free)))
        b_norm = float(np.linalg.norm(b[free_mask_flat]))
        if b_norm <= 0.0:
            solve_tol = 1.0e-12
        else:
            solve_tol = float(np.clip(target_res_norm / b_norm, 1.0e-12, 0.5))

        residuals: list[float] = []
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric")

        try:
            x_flat = ml.solve(
                b,
                x0=x0_flat,
                tol=solve_tol,
                maxiter=max_iter,
                cycle=str(amg_cycle),
                accel="cg",
                residuals=residuals,
            )
        except TypeError:
            x_flat = ml.solve(
                b,
                x0=x0_flat,
                tol=solve_tol,
                maxiter=max_iter,
                cycle=str(amg_cycle),
                residuals=residuals,
            )

        x_flat = np.asarray(x_flat, dtype=np.float64).reshape(-1)

        r_end = b - A.dot(x_flat)
        r_end_free = r_end[free_mask_flat]
        r_rms_end = float(np.sqrt(np.dot(r_end_free, r_end_free) / float(n_free)))
        converged = bool(r_rms_end <= tol_abs)

        if len(residuals) >= 2:
            n_iter_used = int(len(residuals) - 1)
        elif len(residuals) == 1:
            n_iter_used = 1
        else:
            n_iter_used = int(max_iter)

        head_out = x_flat.reshape(ny, nx).astype(NP_FLOAT, copy=False)
        head_out[bc_idx] = np.asarray(self.bc_values_host, dtype=NP_FLOAT)[bc_idx]
        head_out[np.asarray(self.active_host, dtype=np.int32) == 0] = NP_FLOAT(0.0)

        # Keep device state coherent for downstream calls that may use Warp paths.
        try:
            if (
                self.mg_levels is not None
                and len(self.mg_levels) > 0
                and getattr(self.mg_levels[0], "x_wp", None) is not None
                and tuple(self.mg_levels[0].x_wp.shape) == (ny, nx)
            ):
                if (
                    not hasattr(self, "_kcycle_stage_x")
                    or self._kcycle_stage_x is None
                    or tuple(self._kcycle_stage_x.shape) != (ny, nx)
                ):
                    self._kcycle_stage_x = wp.zeros((ny, nx), dtype=WP_FLOAT, device="cpu")
                self._kcycle_stage_x.numpy()[:, :] = np.asarray(head_out, dtype=NP_FLOAT, order="C")
                wp.copy(self.mg_levels[0].x_wp, self._kcycle_stage_x)
        except Exception:
            pass

        info = {
            "solver_type": "amg",
            "linear_backend": "amg",
            "amg_method": "pyamg_smoothed_aggregation",
            "amg_cycle": str(amg_cycle),
            "n_levels": int(getattr(ml, "levels", []) and len(ml.levels) or 0),
            "max_cycles": int(max_cycles),
            "n_cycles_used": int(n_iter_used),
            "converged": bool(converged),
            "rel_tol": float(rel_tol),
            "abs_tol_min": float(abs_tol_min),
            "tol_abs": float(tol_abs),
            "amg_tol_used": float(solve_tol),
            "r_rms0": float(r_rms0),
            "r_rms_end": float(r_rms_end),
            "h_rms_end": float("nan"),
            "aq_thickness": float(aq_thickness),
            "use_ghb": bool(self.use_ghb),
            "cuda_graph_reused": False,
            "cuda_graph_built_this_call": False,
            "check_every": 0,
        }

        return head_out, info

    # -------------------------------------------------------------------------
    # Multigrid helpers (2-level path)
    # -------------------------------------------------------------------------
    def _ensure_two_level_ready(self) -> None:
        """
        Ensure coarse cache and level objects exist.
        """
        if self.T_field_host is None:
            raise RuntimeError("build_from_truth_inputs must be called first.")

        if (not self.mg_cache_built) or (self.T_c_wp is None):
            self._build_two_level_cache()

        if self._fine_level is None:
            self._fine_level = self._GridLevel(
                T_wp=self.T_wp,
                active_wp=self.active_wp,
                bc_mask_wp=self.bc_mask_wp,
                gh_mask_wp=self.gh_mask_wp,
                gh_width_wp=self.gh_width_wp,
                M_inv_wp=self.M_inv_wp,
                nx=self.nx,
                ny=self.ny,
                dx=self.dx,
            )

        if self._coarse_level is None:
            self._coarse_level = self._GridLevel(
                T_wp=self.T_c_wp,
                active_wp=self.active_c_wp,
                bc_mask_wp=self.bc_mask_c_wp,
                gh_mask_wp=self.gh_mask_c_wp,
                gh_width_wp=self.gh_width_c_wp,
                M_inv_wp=self.M_inv_c_wp,
                nx=self.nx_c,
                ny=self.ny_c,
                dx=self.dx_c,
            )

    def _ensure_mg_work_buffers(self, device: str) -> None:
        """
        Allocate multigrid work arrays once per geometry.
        """
        if self._mg_work_built:
            return

        self._ensure_two_level_ready()

        ny_f = int(self._fine_level.ny)
        nx_f = int(self._fine_level.nx)
        ny_c = int(self._coarse_level.ny)
        nx_c = int(self._coarse_level.nx)

        self._mg_work["Ax_f_wp"] = wp.zeros((ny_f, nx_f), dtype=WP_FLOAT, device=device)
        self._mg_work["r_f_wp"] = wp.zeros((ny_f, nx_f), dtype=WP_FLOAT, device=device)
        self._mg_work["e_f_wp"] = wp.zeros((ny_f, nx_f), dtype=WP_FLOAT, device=device)
        self._mg_work["rTr_f_buf"] = wp.zeros(1, dtype=wp.float64, device=device)

        self._mg_work["x_c_wp"] = wp.zeros((ny_c, nx_c), dtype=WP_FLOAT, device=device)
        self._mg_work["Ax_c_wp"] = wp.zeros((ny_c, nx_c), dtype=WP_FLOAT, device=device)
        self._mg_work["b_c_wp"] = wp.zeros((ny_c, nx_c), dtype=WP_FLOAT, device=device)

        self._mg_work["r_c_wp"] = wp.zeros((ny_c, nx_c), dtype=WP_FLOAT, device=device)
        self._mg_work["z_c_wp"] = wp.zeros((ny_c, nx_c), dtype=WP_FLOAT, device=device)
        self._mg_work["p_c_wp"] = wp.zeros((ny_c, nx_c), dtype=WP_FLOAT, device=device)

        self._mg_work["rho_c_buf"] = wp.zeros(1, dtype=wp.float64, device=device)
        self._mg_work["rho_new_c_buf"] = wp.zeros(1, dtype=wp.float64, device=device)
        self._mg_work["rTr_c_buf"] = wp.zeros(1, dtype=wp.float64, device=device)
        self._mg_work["pAp_c_buf"] = wp.zeros(1, dtype=wp.float64, device=device)
        self._mg_work["alpha_c_buf"] = wp.zeros(1, dtype=wp.float64, device=device)
        self._mg_work["beta_c_buf"] = wp.zeros(1, dtype=wp.float64, device=device)
        self._mg_work["converged_c_flag"] = wp.zeros(1, dtype=wp.int32, device=device)

        self._mg_work_built = True

    def _ensure_kcycle_level_stage_buffers(self, levels) -> None:
        """
        Ensure reusable CPU staging buffers exist for per-level gh_width and M_inv uploads.
        """
        nL = int(len(levels))

        need_ghw = self._stage_ghw_levels is None or len(self._stage_ghw_levels) != nL
        if not need_ghw:
            for lid, lvl in enumerate(levels):
                if tuple(self._stage_ghw_levels[lid].shape) != (int(lvl.ny), int(lvl.nx)):
                    need_ghw = True
                    break
        if need_ghw:
            self._stage_ghw_levels = [
                wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu")
                for lvl in levels
            ]

        need_m = self._stage_M_levels is None or len(self._stage_M_levels) != nL
        if not need_m:
            for lid, lvl in enumerate(levels):
                if tuple(self._stage_M_levels[lid].shape) != (int(lvl.ny), int(lvl.nx)):
                    need_m = True
                    break
        if need_m:
            self._stage_M_levels = [
                wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu")
                for lvl in levels
            ]

    def _ensure_kcycle_storage_stage_buffers(self, levels) -> None:
        """
        Ensure reusable CPU staging buffers exist for per-level transient storage uploads.
        """
        nL = int(len(levels))
        need_s = self._stage_storage_levels is None or len(self._stage_storage_levels) != nL
        if not need_s:
            for lid, lvl in enumerate(levels):
                if tuple(self._stage_storage_levels[lid].shape) != (int(lvl.ny), int(lvl.nx)):
                    need_s = True
                    break
        if need_s:
            self._stage_storage_levels = [
                wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu")
                for lvl in levels
            ]

    def _rebuild_kcycle_level_diagonals(
        self,
        levels,
        aq_thickness: float,
        storage_diag_levels: list[np.ndarray] | None = None,
    ) -> None:
        """
        Rebuild level-wise diagonal preconditioners using current T/ghb and optional storage diag.
        """
        self._ensure_kcycle_level_stage_buffers(levels)

        gh_width_eff_levels = getattr(self, "_kcycle_last_gh_width_eff_levels", None)
        nL = int(len(levels))

        for lid, lvl in enumerate(levels):
            shapeL = (int(lvl.ny), int(lvl.nx))

            gh_mask_l = lvl.gh_mask_host if self.use_ghb else None
            if self.use_ghb:
                if (
                    gh_width_eff_levels is not None
                    and len(gh_width_eff_levels) == nL
                    and tuple(np.asarray(gh_width_eff_levels[lid]).shape) == shapeL
                ):
                    gh_width_l = np.asarray(gh_width_eff_levels[lid], dtype=NP_FLOAT)
                elif lvl.gh_width_host is not None:
                    gh_width_l = np.asarray(lvl.gh_width_host, dtype=NP_FLOAT)
                else:
                    gh_width_l = np.zeros(shapeL, dtype=NP_FLOAT)
            else:
                gh_width_l = None

            if storage_diag_levels is not None:
                if len(storage_diag_levels) != nL:
                    raise RuntimeError("storage_diag_levels length mismatch")
                sdiag_l = np.asarray(storage_diag_levels[lid], dtype=np.float64)
                if tuple(sdiag_l.shape) != shapeL:
                    raise RuntimeError(f"storage_diag level {lid} shape mismatch: {sdiag_l.shape} vs {shapeL}")
            else:
                sdiag_l = None

            M_inv_host = build_diag_preconditioner(
                T_field=lvl.T_host,
                active=lvl.active_host,
                bc_mask=lvl.bc_mask_host,
                gh_mask=gh_mask_l,
                gh_width=gh_width_l,
                storage_diag=sdiag_l,
                dx=float(lvl.dx) if self.use_ghb else None,
                gh_alpha=float(self.gh_alpha),
                aq_thickness=float(aq_thickness),
            ).astype(NP_FLOAT, copy=False)

            stage_m = self._stage_M_levels[lid].numpy()
            stage_m[:, :] = M_inv_host
            wp.copy(lvl.M_inv_wp, self._stage_M_levels[lid])

    def _configure_kcycle_confined_transient_storage(
        self,
        levels,
        storage_coeff,
        dt: float,
        head_prev: np.ndarray | None,
        initial_head: np.ndarray | None,
        aq_thickness: float,
        refresh_diag_with_transient_storage: bool,
    ) -> np.ndarray:
        """
        Configure per-level storage diagonals for confined transient K-cycle.
        Returns fine-level `head_prev` array used in RHS storage term.
        """
        dt_f = float(dt)
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            raise ValueError("Transient confined mode requires dt > 0.")

        lvl0 = levels[0]
        ny0 = int(lvl0.ny)
        nx0 = int(lvl0.nx)
        shape0 = (ny0, nx0)

        if storage_coeff is None:
            raise ValueError("Transient confined mode requires storage_coeff.")

        s_in = np.asarray(storage_coeff, dtype=NP_FLOAT)
        if s_in.shape == ():
            S0 = np.full(shape0, NP_FLOAT(s_in.reshape(()).item()), dtype=NP_FLOAT)
        else:
            if tuple(s_in.shape) != shape0:
                raise ValueError(f"storage_coeff shape {s_in.shape} expected {shape0}")
            S0 = np.asarray(s_in, dtype=NP_FLOAT)

        if not np.all(np.isfinite(S0)):
            raise ValueError("storage_coeff must contain finite values.")
        if np.any(S0 < NP_FLOAT(0.0)):
            raise ValueError("storage_coeff must be >= 0.")

        S_levels = [S0]
        for lid in range(1, int(len(levels))):
            S_levels.append(_coarsen_mean_edge_2x2(S_levels[-1]))
            expected = (int(levels[lid].ny), int(levels[lid].nx))
            if tuple(S_levels[-1].shape) != expected:
                raise RuntimeError(
                    f"Coarsened storage_coeff shape {S_levels[-1].shape} does not match level {lid} shape {expected}."
                )

        storage_diag_levels: list[np.ndarray] = []
        for lid, lvl in enumerate(levels):
            dxL = float(lvl.dx)
            diag_l = (
                S_levels[lid].astype(np.float64, copy=False)
                * np.float64(dxL * dxL / dt_f)
            ).astype(NP_FLOAT, copy=False)

            act_l = np.asarray(lvl.active_host, dtype=np.int32) != 0
            bc_l = np.asarray(lvl.bc_mask_host, dtype=np.int32) != 0
            diag_l = np.where(act_l & (~bc_l), diag_l, NP_FLOAT(0.0)).astype(NP_FLOAT, copy=False)
            storage_diag_levels.append(diag_l)

        self._ensure_kcycle_storage_stage_buffers(levels)
        for lid, lvl in enumerate(levels):
            stage_s = self._stage_storage_levels[lid].numpy()
            stage_s[:, :] = storage_diag_levels[lid]
            wp.copy(lvl.storage_wp, self._stage_storage_levels[lid])

        if refresh_diag_with_transient_storage:
            self._rebuild_kcycle_level_diagonals(
                levels=levels,
                aq_thickness=float(aq_thickness),
                storage_diag_levels=storage_diag_levels,
            )

        active0 = np.asarray(lvl0.active_host, dtype=np.int32) != 0
        bc0 = np.asarray(lvl0.bc_mask_host, dtype=np.int32) != 0
        bc_vals0 = np.asarray(lvl0.bc_values_host, dtype=NP_FLOAT)

        if head_prev is not None:
            h_prev = np.asarray(head_prev, dtype=NP_FLOAT).copy()
            if tuple(h_prev.shape) != shape0:
                raise ValueError(f"head_prev shape {h_prev.shape} expected {shape0}")
        elif initial_head is not None:
            h_prev = np.asarray(initial_head, dtype=NP_FLOAT).copy()
            if tuple(h_prev.shape) != shape0:
                raise ValueError(f"initial_head shape {h_prev.shape} expected {shape0}")
        else:
            h_prev = np.asarray(lvl0.x_wp.numpy(), dtype=NP_FLOAT, order="C")

        h_prev[bc0] = bc_vals0[bc0]
        h_prev[~active0] = NP_FLOAT(0.0)
        if not np.all(np.isfinite(h_prev)):
            raise ValueError("head_prev contains non-finite values.")

        self._kcycle_transient_active = True
        return h_prev

    def _sync_kcycle_ghb_for_spatial_alpha(
        self,
        levels,
        gh_alpha_field: np.ndarray | None,
        aq_thickness: float,
        refresh_diag_with_spatial_alpha: bool,
    ) -> bool:
        """
        Prepare per-level gh_width (and optional M_inv) for K-cycle.

        If `gh_alpha_field` is provided, applies spatial scaling:
            gh_width_eff = gh_width * gh_alpha_field
        with coarse alpha created by repeated 2:1 mean coarsening.

        Returns True when spatial alpha is active for this solve.
        """
        if not self.use_ghb:
            if gh_alpha_field is not None:
                raise ValueError("gh_alpha_field requires use_ghb=True.")
            self._kcycle_spatial_alpha_active = False
            self._kcycle_last_gh_width_eff_levels = None
            return False

        was_spatial = bool(getattr(self, "_kcycle_spatial_alpha_active", False))
        spatial_active = gh_alpha_field is not None

        if not spatial_active and not was_spatial:
            self._kcycle_last_gh_width_eff_levels = None
            return False

        alpha_levels = None
        if spatial_active:
            alpha0 = np.asarray(gh_alpha_field, dtype=NP_FLOAT)
            if alpha0.shape != (int(self.ny), int(self.nx)):
                raise ValueError(
                    f"gh_alpha_field shape {alpha0.shape} expected {(int(self.ny), int(self.nx))}"
                )
            if not np.all(np.isfinite(alpha0)):
                raise ValueError("gh_alpha_field must contain only finite values.")
            if np.any(alpha0 < NP_FLOAT(0.0)):
                raise ValueError("gh_alpha_field must be >= 0 everywhere.")

            alpha_levels = [alpha0]
            for lid in range(1, int(len(levels))):
                alpha_levels.append(_coarsen_mean_edge_2x2(alpha_levels[-1]))
                expected = (int(levels[lid].ny), int(levels[lid].nx))
                if tuple(alpha_levels[-1].shape) != expected:
                    raise RuntimeError(
                        f"Coarsened gh_alpha_field shape {alpha_levels[-1].shape} "
                        f"does not match level {lid} shape {expected}."
                    )

        self._ensure_kcycle_level_stage_buffers(levels)
        gh_width_eff_levels: list[np.ndarray] = []

        for lid, lvl in enumerate(levels):
            nyL = int(lvl.ny)
            nxL = int(lvl.nx)
            shapeL = (nyL, nxL)

            if lvl.gh_width_wp is None:
                continue

            if lvl.gh_width_host is not None:
                gh_width_base = np.asarray(lvl.gh_width_host, dtype=NP_FLOAT)
            else:
                gh_width_base = np.zeros(shapeL, dtype=NP_FLOAT)

            if tuple(gh_width_base.shape) != shapeL:
                raise RuntimeError(
                    f"Level {lid} gh_width_host shape mismatch: {gh_width_base.shape} vs {shapeL}"
                )

            if spatial_active:
                alpha_l = alpha_levels[lid]
                gh_width_eff = (
                    gh_width_base.astype(np.float64, copy=False)
                    * alpha_l.astype(np.float64, copy=False)
                ).astype(NP_FLOAT, copy=False)
            else:
                gh_width_eff = gh_width_base

            if lvl.gh_mask_host is not None:
                gh_mask_l = np.asarray(lvl.gh_mask_host, dtype=np.int32)
                if tuple(gh_mask_l.shape) != shapeL:
                    raise RuntimeError(
                        f"Level {lid} gh_mask_host shape mismatch: {gh_mask_l.shape} vs {shapeL}"
                    )
                if spatial_active:
                    gh_width_eff = np.where(gh_mask_l != 0, gh_width_eff, gh_width_base).astype(
                        NP_FLOAT, copy=False
                    )

            if not np.all(np.isfinite(gh_width_eff)):
                raise ValueError(f"Level {lid} effective gh_width has non-finite values.")
            if np.any(gh_width_eff < NP_FLOAT(0.0)):
                raise ValueError(f"Level {lid} effective gh_width must be >= 0.")

            stage_w = self._stage_ghw_levels[lid].numpy()
            stage_w[:, :] = gh_width_eff
            wp.copy(lvl.gh_width_wp, self._stage_ghw_levels[lid])
            gh_width_eff_levels.append(np.asarray(gh_width_eff, dtype=NP_FLOAT).copy())

            if refresh_diag_with_spatial_alpha and (spatial_active or was_spatial):
                M_inv_host = build_diag_preconditioner(
                    T_field=lvl.T_host,
                    active=lvl.active_host,
                    bc_mask=lvl.bc_mask_host,
                    gh_mask=lvl.gh_mask_host if self.use_ghb else None,
                    gh_width=gh_width_eff if self.use_ghb else None,
                    dx=float(lvl.dx) if self.use_ghb else None,
                    gh_alpha=float(self.gh_alpha),
                    aq_thickness=float(aq_thickness),
                ).astype(NP_FLOAT, copy=False)
                stage_m = self._stage_M_levels[lid].numpy()
                stage_m[:, :] = M_inv_host
                wp.copy(lvl.M_inv_wp, self._stage_M_levels[lid])

        self._kcycle_spatial_alpha_active = bool(spatial_active)
        self._kcycle_last_gh_width_eff_levels = gh_width_eff_levels
        return bool(spatial_active)

    def _update_fine_T_and_upload(self, T_truth) -> None:
        """
        Update fine-level transmissivity on host and upload to device without reallocations.
        """
        if self.T_field_host is None or self.T_wp is None:
            raise RuntimeError("Call build_from_truth_inputs() once before updating T.")

        T_arr = np.asarray(T_truth, dtype=NP_FLOAT, order="C")
        if T_arr.shape != tuple(self.T_field_host.shape):
            if T_arr.size == 1:
                self.T_field_host[:, :] = T_arr.reshape(1)[0]
            else:
                raise ValueError(f"T_truth shape {T_arr.shape} expected {self.T_field_host.shape}")
        else:
            np.copyto(self.T_field_host, T_arr)

        ny0 = int(self.ny)
        nx0 = int(self.nx)

        if self._stage_T0 is None or tuple(self._stage_T0.shape) != (ny0, nx0):
            self._stage_T0 = wp.zeros((ny0, nx0), dtype=WP_FLOAT, device="cpu")

        self._stage_T0.numpy()[:, :] = self.T_field_host
        wp.copy(self.T_wp, self._stage_T0)

        if self._fine_level is not None:
            self._fine_level.T_wp = self.T_wp

    def _update_fine_diag_preconditioner(self) -> None:
        """
        Rebuild fine-level diagonal preconditioner and upload in place.
        """
        if self.T_field_host is None:
            raise RuntimeError("Call build_from_truth_inputs() once before updating M_inv.")

        device = self.device_str
        ny0 = int(self.ny)
        nx0 = int(self.nx)

        M_inv_host = build_diag_preconditioner(
            T_field=self.T_field_host,
            active=self.active_host,
            bc_mask=self.bc_mask_host,
            gh_mask=self.gh_mask_host if self.use_ghb else None,
            gh_width=self.gh_width_host if self.use_ghb else None,
            dx=float(self.dx) if self.use_ghb else None,
            gh_alpha=float(self.gh_alpha),
            aq_thickness=float(self.aq_thickness),
        ).astype(NP_FLOAT, copy=False)

        if self.M_inv_wp is None:
            self.M_inv_wp = wp.empty((ny0, nx0), dtype=WP_FLOAT, device=device)

        if self._stage_M0 is None or tuple(self._stage_M0.shape) != (ny0, nx0):
            self._stage_M0 = wp.zeros((ny0, nx0), dtype=WP_FLOAT, device="cpu")

        self._stage_M0.numpy()[:, :] = M_inv_host
        wp.copy(self.M_inv_wp, self._stage_M0)

        if self._fine_level is not None:
            self._fine_level.M_inv_wp = self.M_inv_wp

    def update_T_in_place(self, T_truth) -> None:
        """
        Update transmissivity in place and refresh diagonal preconditioners.
        Keeps Warp arrays allocated so CUDA graphs remain valid.

        :param T_truth: transmissivity field, shape (ny, nx) or broadcastable scalar
        :return: None
        """
        if self.T_field_host is None or self.T_wp is None:
            raise RuntimeError("Call build_from_truth_inputs() once before update_T_in_place().")

        device = self.device_str

        # -------- fine host update + upload --------
        self._update_fine_T_and_upload(T_truth)

        # -------- rebuild fine diagonal preconditioner --------
        self._update_fine_diag_preconditioner()

        # -------- update 2-level cache (if built) --------
        if self.mg_cache_built and (self.T_c_host is not None) and (self.T_c_wp is not None):
            (
                T_c_new,
                R_c_new,
                active_c_new,
                bc_mask_c_new,
                bc_values_c_new,
                gh_mask_c_new,
                gh_head_c_new,
                gh_width_c_new,
            ) = build_coarse_level_from_fine(
                T_f=self.T_field_host,
                R_f=self.R_field_host,
                active_f=self.active_host,
                bc_mask_f=self.bc_mask_host,
                bc_values_f=self.bc_values_host,
                gh_mask_f=self.gh_mask_host,
                gh_head_f=self.gh_head_host,
                gh_width_f=self.gh_width_host,
            )

            # correction scheme conventions
            bc_values_c_new[...] = NP_FLOAT(0.0)
            if gh_head_c_new is not None:
                gh_head_c_new[...] = NP_FLOAT(0.0)

            # copy into existing coarse host arrays (no realloc)
            np.copyto(self.T_c_host, np.asarray(T_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.R_c_host, np.asarray(R_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.active_c_host, np.asarray(active_c_new, dtype=np.int32, order="C"))
            np.copyto(self.bc_mask_c_host, np.asarray(bc_mask_c_new, dtype=np.int32, order="C"))
            np.copyto(self.bc_values_c_host, np.asarray(bc_values_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.gh_mask_c_host, np.asarray(gh_mask_c_new, dtype=np.int32, order="C"))
            np.copyto(self.gh_width_c_host, np.asarray(gh_width_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.gh_head_c_host, np.asarray(gh_head_c_new, dtype=NP_FLOAT, order="C"))

            nyc = int(self.ny_c)
            nxc = int(self.nx_c)

            if self._stage_Tc_2lvl is None or tuple(self._stage_Tc_2lvl.shape) != (nyc, nxc):
                self._stage_Tc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")
            if self._stage_Mc_2lvl is None or tuple(self._stage_Mc_2lvl.shape) != (nyc, nxc):
                self._stage_Mc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")

            self._stage_Tc_2lvl.numpy()[:, :] = self.T_c_host
            wp.copy(self.T_c_wp, self._stage_Tc_2lvl)

            M_inv_c_host = build_diag_preconditioner(
                T_field=self.T_c_host,
                active=self.active_c_host,
                bc_mask=self.bc_mask_c_host,
                gh_mask=self.gh_mask_c_host if self.use_ghb else None,
                gh_width=self.gh_width_c_host if self.use_ghb else None,
                dx=float(self.dx_c) if self.use_ghb else None,
                gh_alpha=float(self.gh_alpha),
                aq_thickness=float(self.aq_thickness),
            ).astype(NP_FLOAT, copy=False)

            self._stage_Mc_2lvl.numpy()[:, :] = M_inv_c_host
            wp.copy(self.M_inv_c_wp, self._stage_Mc_2lvl)

            if self._coarse_level is not None:
                self._coarse_level.T_wp = self.T_c_wp
                self._coarse_level.M_inv_wp = self.M_inv_c_wp

        # -------- update full MG hierarchy (K-cycle) if it exists --------
        if self.mg_levels is not None:
            levels = self.mg_levels
            nL = int(len(levels))

            if self._stage_T_levels is None or self._stage_M_levels is None or len(self._stage_T_levels) != nL:
                self._stage_T_levels = []
                self._stage_M_levels = []
                for lvl in levels:
                    self._stage_T_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))
                    self._stage_M_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))

            # Level 0: make sure level 0 host matches solver host
            lvl0 = levels[0]
            if tuple(lvl0.T_host.shape) == tuple(self.T_field_host.shape):
                np.copyto(lvl0.T_host, self.T_field_host)
            else:
                raise RuntimeError("Level 0 host shape mismatch. Rebuild hierarchy.")

            self._stage_T_levels[0].numpy()[:, :] = lvl0.T_host
            wp.copy(lvl0.T_wp, self._stage_T_levels[0])

            M0 = build_diag_preconditioner(
                T_field=lvl0.T_host,
                active=lvl0.active_host,
                bc_mask=lvl0.bc_mask_host,
                gh_mask=lvl0.gh_mask_host if self.use_ghb else None,
                gh_width=lvl0.gh_width_host if self.use_ghb else None,
                dx=float(lvl0.dx) if self.use_ghb else None,
                gh_alpha=float(self.gh_alpha),
                aq_thickness=float(self.aq_thickness),
            ).astype(NP_FLOAT, copy=False)

            self._stage_M_levels[0].numpy()[:, :] = M0
            wp.copy(lvl0.M_inv_wp, self._stage_M_levels[0])

            # Coarse levels: re-coarsen from previous level and update T + M_inv
            for lid in range(1, nL):
                fine = levels[lid - 1]
                coarse = levels[lid]

                (
                    T_c,
                    R_c,
                    active_c,
                    bc_mask_c,
                    bc_values_c,
                    gh_mask_c,
                    gh_head_c,
                    gh_width_c,
                ) = self._mg_coarsen_host_any(
                    T_f=fine.T_host,
                    R_f=fine.R_host,
                    active_f=fine.active_host,
                    bc_mask_f=fine.bc_mask_host,
                    bc_values_f=fine.bc_values_host,
                    gh_mask_f=fine.gh_mask_host,
                    gh_head_f=fine.gh_head_host,
                    gh_width_f=fine.gh_width_host,
                    dx_c=float(coarse.dx),
                )

                bc_values_c.fill(NP_FLOAT(0.0))
                if gh_head_c is not None:
                    gh_head_c.fill(NP_FLOAT(0.0))

                if T_c.shape != coarse.T_host.shape:
                    raise RuntimeError(f"Level {lid} shape mismatch. Rebuild hierarchy.")

                np.copyto(coarse.T_host, T_c)

                self._stage_T_levels[lid].numpy()[:, :] = coarse.T_host
                wp.copy(coarse.T_wp, self._stage_T_levels[lid])

                Mc = build_diag_preconditioner(
                    T_field=coarse.T_host,
                    active=coarse.active_host,
                    bc_mask=coarse.bc_mask_host,
                    gh_mask=coarse.gh_mask_host if self.use_ghb else None,
                    gh_width=coarse.gh_width_host if self.use_ghb else None,
                    dx=float(coarse.dx) if self.use_ghb else None,
                    gh_alpha=float(self.gh_alpha),
                    aq_thickness=float(self.aq_thickness),
                ).astype(NP_FLOAT, copy=False)

                self._stage_M_levels[lid].numpy()[:, :] = Mc
                wp.copy(coarse.M_inv_wp, self._stage_M_levels[lid])

        # Operator changed
        self._operator_dirty = True


    def update_R_in_place(self, R_truth) -> None:
        """
        Update recharge in place (host + device) without reallocations.
        """
        if self.R_field_host is None or self.R_wp is None:
            raise RuntimeError("Call build_from_truth_inputs() once before update_R_in_place().")

        R_arr = np.asarray(R_truth, dtype=NP_FLOAT, order="C")
        if R_arr.shape != tuple(self.R_field_host.shape):
            if R_arr.size == 1:
                self.R_field_host[:, :] = R_arr.reshape(1)[0]
            else:
                raise ValueError(f"R_truth shape {R_arr.shape} expected {self.R_field_host.shape}")
        else:
            np.copyto(self.R_field_host, R_arr)

        ny0 = int(self.ny)
        nx0 = int(self.nx)

        if self._stage_R0 is None or tuple(self._stage_R0.shape) != (ny0, nx0):
            self._stage_R0 = wp.zeros((ny0, nx0), dtype=WP_FLOAT, device="cpu")

        self._stage_R0.numpy()[:, :] = self.R_field_host
        wp.copy(self.R_wp, self._stage_R0)

    def update_T_in_place_fast(self, T_truth, update_diag_preconditioner: bool = False) -> None:
        """
        Fast update: fine-level T upload only. Optionally refresh fine M_inv.
        Skips coarse cache and multigrid hierarchy rebuilds.
        """
        if self.T_field_host is None or self.T_wp is None:
            raise RuntimeError("Call build_from_truth_inputs() once before update_T_in_place_fast().")

        self._update_fine_T_and_upload(T_truth)

        if update_diag_preconditioner:
            self._update_fine_diag_preconditioner()

        self._operator_dirty = True


    def update_T_in_place_ultrafast(self, T_truth, update_diag_preconditioner: bool = False) -> None:
        """
        Ultrafast update: try device-to-device copy when T_truth is a Warp array.
        Falls back to fast host staging otherwise. Coarse levels are not rebuilt.
        """
        if self.T_field_host is None or self.T_wp is None:
            raise RuntimeError("Call build_from_truth_inputs() once before update_T_in_place_ultrafast().")

        # Prefer device-to-device copy if possible
        used_device_copy = False
        if hasattr(T_truth, "device") and hasattr(T_truth, "shape"):
            try:
                if str(T_truth.device) == str(self.device_str) and tuple(T_truth.shape) == tuple(self.T_wp.shape):
                    wp.copy(self.T_wp, T_truth)
                    used_device_copy = True
            except Exception:
                used_device_copy = False

        # Keep host copy coherent for host-side RHS/coarsening paths.
        if used_device_copy:
            try:
                T_dev_host = np.asarray(T_truth.numpy(), dtype=NP_FLOAT, order="C")
                np.copyto(self.T_field_host, T_dev_host)
            except Exception:
                np.copyto(self.T_field_host, np.asarray(self.T_wp.numpy(), dtype=NP_FLOAT, order="C"))
        else:
            T_arr = np.asarray(T_truth, dtype=NP_FLOAT, order="C")
            if T_arr.shape != tuple(self.T_field_host.shape):
                if T_arr.size == 1:
                    self.T_field_host[:, :] = T_arr.reshape(1)[0]
                else:
                    raise ValueError(f"T_truth shape {T_arr.shape} expected {self.T_field_host.shape}")
            else:
                np.copyto(self.T_field_host, T_arr)

            ny0 = int(self.ny)
            nx0 = int(self.nx)
            if self._stage_T0 is None or tuple(self._stage_T0.shape) != (ny0, nx0):
                self._stage_T0 = wp.zeros((ny0, nx0), dtype=WP_FLOAT, device="cpu")
            self._stage_T0.numpy()[:, :] = self.T_field_host
            wp.copy(self.T_wp, self._stage_T0)

        if self._fine_level is not None:
            self._fine_level.T_wp = self.T_wp

        if update_diag_preconditioner:
            self._update_fine_diag_preconditioner()

        self._operator_dirty = True



    def solve_multigrid_kcycle(
            self,
            max_cycles: int = 20,
            nu_pre: int = 2,
            nu_post: int = 2,
            nu_coarse: int = 30,
            omega: float = 0.8,
            rel_tol: float = 5.0e-7,
            abs_tol_min: float = 5.0e-7,
            initial_head: np.ndarray | None = None,
            aq_thickness: float | None = None,
            max_levels: int = 5,
            return_info: bool = True,
            check_every_no: int = 10,
            dh_rms_tol: float | None  = 1.0e-4,
            dh_max_tol: float | None = None,
            dh_max_factor: float = 5.0,
            linear_backend: str = "kcycle",
            stencil: str = "5-point",
            amg_max_iter: int | None = None,
            amg_cycle: str = "V",
            smoother: str = "chebyshev",
            cheby_lambda_min: float = 0.05,
            cheby_lambda_max: float = 1.95,
            gh_alpha_field: np.ndarray | None = None,
            refresh_diag_with_spatial_alpha: bool = True,
            unconfined: bool = False,
            K_field: np.ndarray | None = None,
            zbot_field: np.ndarray | None = None,
            unconfined_min_sat: float = 0.1,
            unconfined_max_picard_iter: int = 8,
            unconfined_relax: float = 0.7,
            unconfined_head_tol: float = 1.0e-3,
            transient: bool = False,
            storage_coeff: np.ndarray | float | None = None,
            dt: float | None = None,
            head_prev: np.ndarray | None = None,
            refresh_diag_with_transient_storage: bool = True,
    ):
        """
        K-cycle multigrid using your existing hierarchy (self.mg_levels).

        Uses correction scheme (coarse RHS is restricted residual) and 2-term Krylov accel:
            z1 = B(b)
            r1 = b - A z1
            z2 = B(r1)
            alpha = (r1^T z2) / (z2^T A z2)
            e = z1 + alpha z2

        Optional linear backends:
          - "kcycle": Warp GPU multigrid K-cycle (default)
          - "amg": host sparse assembly + pyamg smoothed aggregation

        Stencil options:
          - "5-point": current 2D path used by this K-cycle implementation
          - "7-point": reserved for upcoming multilayer integration in this method
            (use solve_chebyshev_7point_3d or solve_multigrid_kcycle_7point_3d
             for current prototype streams)

        Smoother options:
          - "chebyshev": Chebyshev-weighted Jacobi semi-iteration (default in this clone)
          - "jacobi": legacy fixed-omega Jacobi

        Optional spatial GHB scaling (K-cycle only):
          - gh_alpha_field: per-cell multiplier field (fine-grid shape ny x nx)
            used in conductance as: C_gh ~ gh_alpha * gh_alpha_field * width.
          - refresh_diag_with_spatial_alpha: if True, rebuild level diagonal
            preconditioners to match the active spatial-alpha conductance.

        Optional unconfined MVP (Picard outer loop, K-cycle/Chebyshev inner solve):
          - unconfined: enable transmissivity update T(h) = K * max(h - zbot, min_sat)
          - K_field: hydraulic conductivity [L/T], fine-grid shape (ny, nx)
          - zbot_field: aquifer bottom elevation [L], fine-grid shape (ny, nx)
          - unconfined_min_sat: minimum saturated thickness floor [L]
          - unconfined_max_picard_iter: max nonlinear (Picard) iterations
          - unconfined_relax: Picard damping factor in (0, 1]
          - unconfined_head_tol: RMS head-change tolerance for Picard convergence [L]

        Optional transient streams (Chebyshev + K-cycle):
          - transient=False: steady-state (default)
          - transient=True, unconfined=False: confined transient stream
              storage term: S * dx^2 / dt
          - transient=True, unconfined=True: unconfined transient stream placeholder
              (declared for future use; currently raises NotImplementedError)

          Controls:
          - storage_coeff: scalar/field storage coefficient S (required for confined transient)
          - dt: time step size (required for confined transient)
          - head_prev: previous time-step head h^n (defaults to initial_head, else last solved head)
          - refresh_diag_with_transient_storage: rebuild diagonal preconditioners with storage term
        """

        # Normalize tolerances (treat None as disabled)
        dh_rms_tol_f = None if dh_rms_tol is None else float(dh_rms_tol)

        if dh_max_tol is None:
            dh_max_tol = None if dh_rms_tol_f is None else float(dh_max_factor) * dh_rms_tol_f
        else:
            dh_max_tol = float(dh_max_tol)

        max_cycles_i = int(max_cycles)
        if max_cycles_i < 1:
            raise ValueError("max_cycles must be >= 1")

        check_every = int(check_every_no)
        if check_every < 1:
            raise ValueError("check_every_no must be >= 1")

        if float(self.head_scale) != 1.0:
            raise ValueError(
                "K-cycle runs in physical head units only. "
                "Set head_scale=1.0 for K-cycle, or use PCG / 2-level MG if you want scaling."
            )

        if aq_thickness is None:
            aq_thickness_f = float(self.aq_thickness)
        else:
            aq_thickness_f = float(aq_thickness)

        backend_mode = str(linear_backend).strip().lower()
        if backend_mode not in {"kcycle", "amg"}:
            raise ValueError("linear_backend must be 'kcycle' or 'amg'")

        stencil_mode = _normalize_stencil_mode(stencil)
        if stencil_mode == "7-point":
            raise NotImplementedError(
                "solve_multigrid_kcycle currently runs the 2D 5-point path only. "
                "Use solve_chebyshev_7point_3d(...) or solve_multigrid_kcycle_7point_3d(...) "
                "for the current 7-point prototype streams."
            )

        transient_mode = bool(transient)
        if transient_mode:
            if backend_mode != "kcycle":
                raise ValueError("Transient streams are currently supported only with linear_backend='kcycle'.")
            if str(smoother).strip().lower() != "chebyshev":
                raise ValueError("Transient streams are currently supported only with smoother='chebyshev'.")
            if bool(unconfined):
                raise NotImplementedError(
                    "Transient unconfined stream is scaffolded but not implemented yet. "
                    "Use transient confined stream for now."
                )

        if bool(unconfined):
            if backend_mode != "kcycle":
                raise ValueError("unconfined mode is currently supported only with linear_backend='kcycle'.")

            smoother_mode_unconf = str(smoother).strip().lower()
            if smoother_mode_unconf != "chebyshev":
                raise ValueError("unconfined mode is currently supported only with smoother='chebyshev'.")

            if self.active_host is None or self.bc_mask_host is None or self.bc_values_host is None:
                raise RuntimeError("build_from_truth_inputs or build_from_fields must be called before solve().")

            if K_field is None or zbot_field is None:
                raise ValueError("unconfined mode requires both K_field and zbot_field.")

            ny0 = int(self.ny)
            nx0 = int(self.nx)
            shape0 = (ny0, nx0)

            K_arr = np.asarray(K_field, dtype=NP_FLOAT)
            zbot_arr = np.asarray(zbot_field, dtype=NP_FLOAT)
            if K_arr.shape != shape0:
                raise ValueError(f"K_field shape {K_arr.shape} expected {shape0}")
            if zbot_arr.shape != shape0:
                raise ValueError(f"zbot_field shape {zbot_arr.shape} expected {shape0}")
            if not np.all(np.isfinite(K_arr)):
                raise ValueError("K_field must contain finite values.")
            if np.any(K_arr < NP_FLOAT(0.0)):
                raise ValueError("K_field must be >= 0.")
            if not np.all(np.isfinite(zbot_arr)):
                raise ValueError("zbot_field must contain finite values.")

            min_sat = float(unconfined_min_sat)
            if min_sat <= 0.0:
                raise ValueError("unconfined_min_sat must be positive.")

            n_picard = int(unconfined_max_picard_iter)
            if n_picard < 1:
                raise ValueError("unconfined_max_picard_iter must be >= 1.")

            picard_relax = float(unconfined_relax)
            if picard_relax <= 0.0 or picard_relax > 1.0:
                raise ValueError("unconfined_relax must be in (0, 1].")

            picard_tol = float(unconfined_head_tol)
            if picard_tol < 0.0:
                raise ValueError("unconfined_head_tol must be >= 0.")

            active_mask = np.asarray(self.active_host, dtype=np.int32) != 0
            bc_mask0 = np.asarray(self.bc_mask_host, dtype=np.int32) != 0
            free_mask0 = active_mask & (~bc_mask0)
            bc_values0 = np.asarray(self.bc_values_host, dtype=NP_FLOAT)

            if initial_head is None:
                h_iter = (zbot_arr.astype(np.float64, copy=False) + float(min_sat)).astype(NP_FLOAT, copy=False)
            else:
                h_iter = np.asarray(initial_head, dtype=NP_FLOAT).copy()
                if h_iter.shape != shape0:
                    raise ValueError(f"initial_head must have shape {shape0}, got {h_iter.shape}")

            h_iter = np.asarray(h_iter, dtype=NP_FLOAT).copy()
            h_iter[bc_mask0] = bc_values0[bc_mask0]
            h_iter[~active_mask] = NP_FLOAT(0.0)
            if not np.all(np.isfinite(h_iter)):
                raise ValueError("initial head for unconfined solve must be finite.")

            nonlinear_converged = False
            picard_used = 0
            dh_rms_last = float("nan")
            dh_max_last = float("nan")
            last_linear_info = {}

            for pic_it in range(n_picard):
                picard_used = pic_it + 1

                sat = np.maximum(
                    h_iter.astype(np.float64, copy=False) - zbot_arr.astype(np.float64, copy=False),
                    float(min_sat),
                ).astype(NP_FLOAT, copy=False)

                T_pic = (
                    K_arr.astype(np.float64, copy=False) * sat.astype(np.float64, copy=False)
                ).astype(NP_FLOAT, copy=False)
                T_pic[~active_mask] = NP_FLOAT(0.0)

                self.update_T_in_place(T_pic)

                head_lin, info_lin = self.solve_multigrid_kcycle(
                    max_cycles=int(max_cycles),
                    nu_pre=int(nu_pre),
                    nu_post=int(nu_post),
                    nu_coarse=int(nu_coarse),
                    omega=float(omega),
                    rel_tol=float(rel_tol),
                    abs_tol_min=float(abs_tol_min),
                    initial_head=h_iter,
                    aq_thickness=aq_thickness,
                    max_levels=int(max_levels),
                    return_info=True,
                    check_every_no=int(check_every_no),
                    dh_rms_tol=dh_rms_tol,
                    dh_max_tol=dh_max_tol,
                    dh_max_factor=float(dh_max_factor),
                    linear_backend="kcycle",
                    stencil=str(stencil_mode),
                    amg_max_iter=amg_max_iter,
                    amg_cycle=str(amg_cycle),
                    smoother="chebyshev",
                    cheby_lambda_min=float(cheby_lambda_min),
                    cheby_lambda_max=float(cheby_lambda_max),
                    gh_alpha_field=gh_alpha_field,
                    refresh_diag_with_spatial_alpha=bool(refresh_diag_with_spatial_alpha),
                    unconfined=False,
                    K_field=None,
                    zbot_field=None,
                    transient=False,
                    storage_coeff=None,
                    dt=None,
                    head_prev=None,
                )

                last_linear_info = info_lin if isinstance(info_lin, dict) else {}
                h_lin = np.asarray(head_lin, dtype=NP_FLOAT)
                if h_lin.shape != shape0:
                    raise RuntimeError(f"Inner linear solve returned shape {h_lin.shape}, expected {shape0}.")

                h_next = (
                    h_iter.astype(np.float64, copy=False)
                    + picard_relax * (h_lin.astype(np.float64, copy=False) - h_iter.astype(np.float64, copy=False))
                ).astype(NP_FLOAT, copy=False)

                h_next[bc_mask0] = bc_values0[bc_mask0]
                h_next[~active_mask] = NP_FLOAT(0.0)

                if np.any(free_mask0):
                    dh = (
                        h_next.astype(np.float64, copy=False) - h_iter.astype(np.float64, copy=False)
                    )[free_mask0]
                    dh_rms_last = float(np.sqrt(np.mean(dh * dh)))
                    dh_max_last = float(np.max(np.abs(dh)))
                else:
                    dh_rms_last = 0.0
                    dh_max_last = 0.0

                h_iter = h_next

                if dh_rms_last <= picard_tol:
                    nonlinear_converged = True
                    break

            info_out = dict(last_linear_info) if isinstance(last_linear_info, dict) else {}
            info_out["solver_type"] = "kcycle_unconfined_picard"
            info_out["linear_solver_type"] = str(last_linear_info.get("solver_type", "kcycle"))
            info_out["unconfined"] = True
            info_out["picard_converged"] = bool(nonlinear_converged)
            info_out["picard_n_iter_used"] = int(picard_used)
            info_out["picard_max_iter"] = int(n_picard)
            info_out["picard_relax"] = float(picard_relax)
            info_out["picard_head_tol"] = float(picard_tol)
            info_out["picard_dh_rms_end"] = float(dh_rms_last)
            info_out["picard_dh_max_end"] = float(dh_max_last)
            info_out["unconfined_min_sat"] = float(min_sat)

            return (h_iter, info_out) if return_info else h_iter

        if gh_alpha_field is not None and backend_mode != "kcycle":
            raise ValueError("gh_alpha_field is currently supported only with linear_backend='kcycle'.")

        if backend_mode == "amg":
            head_amg, info_amg = self._solve_amg_host_backend(
                max_cycles=int(max_cycles),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                initial_head=initial_head,
                aq_thickness=float(aq_thickness_f),
                amg_max_iter=amg_max_iter,
                amg_cycle=str(amg_cycle),
            )
            return (head_amg, info_amg) if return_info else head_amg

        smoother_mode = str(smoother).strip().lower()
        if smoother_mode not in {"chebyshev", "jacobi"}:
            raise ValueError("smoother must be 'chebyshev' or 'jacobi'")

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

        if not hasattr(self, "_kcycle_graph"):
            self._kcycle_graph = None
            self._kcycle_graph_shape = None

        if self.mg_levels is None:
            self.build_hierarchy(max_levels=int(max_levels), min_coarse_n=4)

        levels = self.mg_levels
        if levels is None or len(levels) < 1:
            raise RuntimeError("No multigrid levels available. build_hierarchy() failed.")

        device = self.device_str

        # Ensure every level has gh_mask_wp and gh_width_wp (allocate once if missing).
        for lvl in levels:
            shape = (int(lvl.ny), int(lvl.nx))
            if getattr(lvl, "gh_mask_wp", None) is None:
                lvl.gh_mask_wp = wp.zeros(shape, dtype=wp.int32, device=device)
            if getattr(lvl, "gh_width_wp", None) is None:
                lvl.gh_width_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)

        spatial_alpha_active = self._sync_kcycle_ghb_for_spatial_alpha(
            levels=levels,
            gh_alpha_field=gh_alpha_field,
            aq_thickness=float(aq_thickness_f),
            refresh_diag_with_spatial_alpha=bool(refresh_diag_with_spatial_alpha),
        )

        lvl0 = levels[0]
        ny0 = int(lvl0.ny)
        nx0 = int(lvl0.nx)
        dim0 = (ny0, nx0)

        h_prev_transient = None
        if transient_mode:
            h_prev_transient = self._configure_kcycle_confined_transient_storage(
                levels=levels,
                storage_coeff=storage_coeff,
                dt=float(dt) if dt is not None else float("nan"),
                head_prev=head_prev,
                initial_head=initial_head,
                aq_thickness=float(aq_thickness_f),
                refresh_diag_with_transient_storage=bool(refresh_diag_with_transient_storage),
            )
        else:
            for lvl in levels:
                if getattr(lvl, "storage_wp", None) is None:
                    lvl.storage_wp = wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device=device)
                else:
                    lvl.storage_wp.fill_(WP_FLOAT(0.0))

            if self._kcycle_transient_active:
                if bool(refresh_diag_with_transient_storage):
                    self._rebuild_kcycle_level_diagonals(
                        levels=levels,
                        aq_thickness=float(aq_thickness_f),
                        storage_diag_levels=None,
                    )
                self._kcycle_transient_active = False

        # No allocations in solve: require hierarchy to have buffers.
        required = (
            "b_wp",
            "x_wp",
            "r_wp",
            "Ax_wp",
            "e_wp",
            "storage_wp",
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
        if spatial_alpha_active or transient_mode:
            wp.launch(
                kernel=build_rhs_kernel,
                dim=dim0,
                inputs=[
                    lvl0.T_wp,
                    lvl0.R_wp,
                    lvl0.active_wp,
                    lvl0.bc_mask_wp,
                    lvl0.bc_values_wp,
                    lvl0.gh_mask_wp,
                    lvl0.gh_head_wp,
                    lvl0.gh_width_wp,
                    nx0,
                    ny0,
                    float(lvl0.dx),
                    float(self.gh_alpha),
                    float(self.head_scale),
                    float(aq_thickness_f),
                    lvl0.b_wp,
                ],
                device=device,
            )
        else:
            self._build_rhs_fine(lvl0.b_wp, aq_thickness=float(aq_thickness_f))

        if transient_mode:
            stage_x_np = self._kcycle_stage_x.numpy()
            stage_x_np[:, :] = np.asarray(h_prev_transient, dtype=NP_FLOAT, order="C")
            wp.copy(lvl0.e_wp, self._kcycle_stage_x)
            wp.launch(
                kernel=add_storage_rhs_kernel,
                dim=dim0,
                inputs=[
                    lvl0.b_wp,
                    lvl0.storage_wp,
                    lvl0.e_wp,
                    lvl0.active_wp,
                    lvl0.bc_mask_wp,
                    nx0,
                    ny0,
                ],
                device=device,
            )

        # Initial guess (host), then copy into persistent lvl0.x_wp
        x0 = np.zeros((ny0, nx0), dtype=NP_FLOAT)

        if initial_head is not None:
            init_arr = np.asarray(initial_head, dtype=NP_FLOAT)
            if init_arr.shape != (ny0, nx0):
                raise ValueError(f"initial_head must have shape ({ny0}, {nx0}), got {init_arr.shape}")
            x0[:, :] = init_arr
        elif transient_mode and h_prev_transient is not None:
            x0[:, :] = np.asarray(h_prev_transient, dtype=NP_FLOAT)

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
            info = {
                "solver_type": "kcycle",
                "stencil": str(stencil_mode),
                "n_cycles_used": 0,
                "converged": True,
            }
            return (head_out, info) if return_info else head_out

        # Initial residual for tol computation (one scalar readback per solve)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
        wp.launch(
            kernel=compute_residual_kernel,
            dim=dim0,
            inputs=[
                lvl0.x_wp,
                lvl0.b_wp,
                lvl0.T_wp,
                lvl0.active_wp,
                lvl0.bc_mask_wp,
                lvl0.gh_mask_wp,
                lvl0.gh_width_wp,
                lvl0.storage_wp,
                lvl0.r_wp,
                lvl0.rTr_buf,
                nx0,
                ny0,
                float(lvl0.dx),
                float(self.gh_alpha),
                float(aq_thickness_f),
            ],
            device=device,
        )
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

            wp.launch(
                kernel=init_pcg_with_A_kernel,
                dim=dimL,
                inputs=[
                    level.x_wp,
                    level.b_wp,
                    level.T_wp,
                    level.active_wp,
                    level.bc_mask_wp,
                    level.gh_mask_wp,
                    level.gh_width_wp,
                    level.storage_wp,
                    level.M_inv_wp,
                    level.Ap_wp,
                    level.r_wp,
                    level.z_wp,
                    level.p_wp,
                    level.rho_buf,
                    level.rTr_buf,
                    nxL,
                    nyL,
                    float(level.dx),
                    float(self.gh_alpha),
                    float(aq_thickness_f),
                ],
                device=device,
            )

            for _ in range(int(max_iter_level)):
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.pAp_buf], device=device)
                wp.launch(
                    kernel=apply_A_and_pAp_kernel,
                    dim=dimL,
                    inputs=[
                        level.T_wp,
                        level.active_wp,
                        level.bc_mask_wp,
                        level.gh_mask_wp,
                        level.gh_width_wp,
                        level.storage_wp,
                        level.p_wp,
                        level.Ap_wp,
                        level.pAp_buf,
                        nxL,
                        nyL,
                        float(level.dx),
                        float(self.gh_alpha),
                        float(aq_thickness_f),
                    ],
                    device=device,
                )

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

            dxL = float(level.dx)
            gh_alpha_f = float(self.gh_alpha)
            aq_thick_f = float(aq_thickness_f)

            x_tmp_wp = level.Ax_wp
            x_in = level.x_wp
            x_out = x_tmp_wp

            for omega_step in pre_omegas:
                wp.launch(
                    kernel=jacobi_applyA_fused_kernel,
                    dim=dimL,
                    inputs=[
                        level.T_wp,
                        level.active_wp,
                        level.bc_mask_wp,
                        level.gh_mask_wp,
                        level.gh_width_wp,
                        level.storage_wp,
                        level.b_wp,
                        x_in,
                        level.M_inv_wp,
                        level.bc_values_wp,
                        float(omega_step),
                        nxL,
                        nyL,
                        dxL,
                        gh_alpha_f,
                        aq_thick_f,
                        x_out,
                    ],
                    device=device,
                )
                tmp = x_in
                x_in = x_out
                x_out = tmp

            if x_in is not level.x_wp:
                wp.launch(kernel=copy_field_kernel, dim=dimL, inputs=[x_in, level.x_wp, nxL, nyL], device=device)

            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[level.rTr_buf], device=device)
            wp.launch(
                kernel=compute_residual_kernel,
                dim=dimL,
                inputs=[
                    level.x_wp,
                    level.b_wp,
                    level.T_wp,
                    level.active_wp,
                    level.bc_mask_wp,
                    level.gh_mask_wp,
                    level.gh_width_wp,
                    level.storage_wp,
                    level.r_wp,
                    level.rTr_buf,
                    nxL,
                    nyL,
                    dxL,
                    gh_alpha_f,
                    aq_thick_f,
                ],
                device=device,
            )

            if level_id == (len(levels) - 1):
                pcg_solve_level(level=level, max_iter_level=int(nu_coarse))
                return

            coarse = levels[level_id + 1]
            nxC = int(coarse.nx)
            nyC = int(coarse.ny)
            dimC = (nyC, nxC)
            dxC = float(coarse.dx)

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
            wp.launch(
                kernel=compute_residual_kernel,
                dim=dimC,
                inputs=[
                    z1_wp,
                    coarse.b_wp,
                    coarse.T_wp,
                    coarse.active_wp,
                    coarse.bc_mask_wp,
                    coarse.gh_mask_wp,
                    coarse.gh_width_wp,
                    coarse.storage_wp,
                    coarse.r_wp,
                    coarse.rTr_buf,
                    nxC,
                    nyC,
                    dxC,
                    gh_alpha_f,
                    aq_thick_f,
                ],
                device=device,
            )

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
            wp.launch(
                kernel=apply_A_and_pAp_kernel,
                dim=dimC,
                inputs=[
                    coarse.T_wp,
                    coarse.active_wp,
                    coarse.bc_mask_wp,
                    coarse.gh_mask_wp,
                    coarse.gh_width_wp,
                    coarse.storage_wp,
                    coarse.x_wp,
                    coarse.Ax_wp,
                    coarse.pAp_buf,
                    nxC,
                    nyC,
                    dxC,
                    gh_alpha_f,
                    aq_thick_f,
                ],
                device=device,
            )

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
                wp.launch(
                    kernel=jacobi_applyA_fused_kernel,
                    dim=dimL,
                    inputs=[
                        level.T_wp,
                        level.active_wp,
                        level.bc_mask_wp,
                        level.gh_mask_wp,
                        level.gh_width_wp,
                        level.storage_wp,
                        level.b_wp,
                        x_in,
                        level.M_inv_wp,
                        level.bc_values_wp,
                        float(omega_step),
                        nxL,
                        nyL,
                        dxL,
                        gh_alpha_f,
                        aq_thick_f,
                        x_out,
                    ],
                    device=device,
                )
                tmp = x_in
                x_in = x_out
                x_out = tmp

            if x_in is not level.x_wp:
                wp.launch(kernel=copy_field_kernel, dim=dimL, inputs=[x_in, level.x_wp, nxL, nyL], device=device)

        # Outer cycles
        n_cycles_used = 0
        converged = False

        graph_key = (
            "kcycle",
            str(stencil_mode),
            int(len(levels)),
            tuple((int(l.ny), int(l.nx)) for l in levels),
            int(nu_pre),
            int(nu_post),
            int(nu_coarse),
            str(smoother_mode),
            tuple(float(v) for v in pre_omegas),
            tuple(float(v) for v in post_omegas),
            float(omega),
            float(self.gh_alpha),
            float(aq_thickness_f),
        )

        graph_built_this_call = False

        dh_rms_lastcheck = float("nan")
        dh_max_lastcheck = float("nan")

        for cyc in range(max_cycles_i):
            n_cycles_used = cyc + 1

            if self._kcycle_graph is None or self._kcycle_graph_shape != graph_key:
                with wp.ScopedCapture() as cap:
                    kcycle(0)
                self._kcycle_graph = cap.graph
                self._kcycle_graph_shape = graph_key
                graph_built_this_call = True
            else:
                wp.capture_launch(self._kcycle_graph)

            should_check = ((cyc % check_every) == (check_every - 1)) or (cyc == (max_cycles_i - 1))
            if not should_check:
                continue

            # (A) dvclose-like diagnostics and (B) flux residual check in one pass.
            wp.launch(
                kernel=reset_kcycle_check_buffers_kernel,
                dim=1,
                inputs=[lvl0.rho_buf, lvl0.dh_max_buf, lvl0.rTr_buf, lvl0.converged_flag],
                device=device,
            )
            wp.launch(
                kernel=kcycle_check_dh_and_residual_kernel,
                dim=dim0,
                inputs=[
                    lvl0.x_wp,
                    lvl0.x_prev_wp,
                    lvl0.b_wp,
                    lvl0.T_wp,
                    lvl0.active_wp,
                    lvl0.bc_mask_wp,
                    lvl0.gh_mask_wp,
                    lvl0.gh_width_wp,
                    lvl0.storage_wp,
                    lvl0.rho_buf,  # dh2_buf
                    lvl0.dh_max_buf,
                    lvl0.rTr_buf,  # residual norm
                    nx0,
                    ny0,
                    float(lvl0.dx),
                    float(self.gh_alpha),
                    float(aq_thickness_f),
                ],
                device=device,
            )
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
            if dh_rms_tol_f is not None:
                dh_ok = dh_ok and (dh_rms_lastcheck <= float(dh_rms_tol_f))
            if dh_max_tol is not None:
                dh_ok = dh_ok and (dh_max_lastcheck <= float(dh_max_tol))

            res_ok = int(lvl0.converged_flag.numpy()[0]) != 0

            if res_ok and dh_ok:
                converged = True
                break

        # Final head pullback
        head_out = lvl0.x_wp.numpy()

        # Final flux residual RMS for reporting
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
        wp.launch(
            kernel=compute_residual_kernel,
            dim=dim0,
            inputs=[
                lvl0.x_wp,
                lvl0.b_wp,
                lvl0.T_wp,
                lvl0.active_wp,
                lvl0.bc_mask_wp,
                lvl0.gh_mask_wp,
                lvl0.gh_width_wp,
                lvl0.storage_wp,
                lvl0.r_wp,
                lvl0.rTr_buf,
                nx0,
                ny0,
                float(lvl0.dx),
                float(self.gh_alpha),
                float(aq_thickness_f),
            ],
            device=device,
        )
        rTr_end = float(lvl0.rTr_buf.numpy()[0])
        r_rms_end = float(np.sqrt(max(rTr_end, 0.0) / float(n_free0)))

        # Head-equivalent residual RMS for reporting
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
        wp.launch(
            kernel=compute_head_residual_kernel,
            dim=dim0,
            inputs=[
                lvl0.x_wp,
                lvl0.b_wp,
                lvl0.T_wp,
                lvl0.active_wp,
                lvl0.bc_mask_wp,
                lvl0.gh_mask_wp,
                lvl0.gh_width_wp,
                lvl0.storage_wp,
                lvl0.r_wp,  # stores r_h [m]
                lvl0.rTr_buf,  # sum(r_h^2)
                nx0,
                ny0,
                float(lvl0.dx),
                float(self.gh_alpha),
                float(aq_thickness_f),
            ],
            device=device,
        )
        hrTr_end = float(lvl0.rTr_buf.numpy()[0])
        h_rms_end = float(np.sqrt(max(hrTr_end, 0.0) / float(n_free0)))

        # For dvclose-like metrics: the last check is the meaningful "end" value
        dh_rms_end = float(dh_rms_lastcheck)
        dh_max_end = float(dh_max_lastcheck)

        info = {
            "solver_type": "kcycle",
            "linear_backend": "kcycle",
            "stencil": str(stencil_mode),
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
            "aq_thickness": float(aq_thickness_f),
            "use_ghb": bool(self.use_ghb),
            "spatial_gh_alpha": bool(spatial_alpha_active),
            "transient": bool(transient_mode),
            "transient_formulation": "confined" if transient_mode else "steady",
            "dt": float(dt) if transient_mode else float("nan"),
            "cuda_graph_reused": bool((not graph_built_this_call) and (self._kcycle_graph is not None)),
            "cuda_graph_built_this_call": bool(graph_built_this_call),
            "check_every": int(check_every),
        }

        return (head_out, info) if return_info else head_out

    def solve(
        self,
        formulation: str = "confined",
        solver: str | None = None,
        smoother: str = "chebyshev",
        linear_backend: str = "kcycle",
        stencil: str = "5-point",
        initial_head: np.ndarray | None = None,
        aq_thickness: float | None = None,
        rel_tol: float = 5.0e-7,
        abs_tol_min: float = 5.0e-7,
        return_info: bool = True,
        # PCG controls
        pcg_max_iter: int = 250,
        # K-cycle controls
        max_cycles: int = 20,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 30,
        omega: float = 0.8,
        max_levels: int = 5,
        check_every_no: int = 10,
        dh_rms_tol: float | None = 1.0e-4,
        dh_max_tol: float | None = None,
        dh_max_factor: float = 5.0,
        amg_max_iter: int | None = None,
        amg_cycle: str = "V",
        cheby_lambda_min: float = 0.05,
        cheby_lambda_max: float = 1.95,
        gh_alpha_field: np.ndarray | None = None,
        refresh_diag_with_spatial_alpha: bool = True,
        # Unconfined controls
        K_field: np.ndarray | None = None,
        zbot_field: np.ndarray | None = None,
        unconfined_min_sat: float = 0.1,
        unconfined_max_picard_iter: int = 8,
        unconfined_relax: float = 0.7,
        unconfined_head_tol: float = 1.0e-3,
        # Transient controls (Chebyshev/K-cycle stream)
        transient: bool = False,
        storage_coeff: np.ndarray | float | None = None,
        dt: float | None = None,
        head_prev: np.ndarray | None = None,
        refresh_diag_with_transient_storage: bool = True,
    ):
        """
        Unified solver wrapper with explicit model/solver switches.

        Parameters
        ----------
        formulation : {"confined", "unconfined"}
            Flow formulation selector. Unconfined uses the Picard outer loop implemented
            in `solve_multigrid_kcycle`.
        solver : {"pcg", "kcycle", "amg"} or None
            High-level solver selector. If None, uses `self.solver_type`.
            For `formulation="unconfined"` or `transient=True`, defaults to "kcycle".
            - "pcg": device PCG (`_solve_pcg_device_loop`)
            - "kcycle": multigrid K-cycle (`solve_multigrid_kcycle` with linear_backend="kcycle")
            - "amg": host pyamg path (`solve_multigrid_kcycle` with linear_backend="amg")
        smoother : {"chebyshev", "jacobi"}
            Smoother for K-cycle backend.
        linear_backend : {"kcycle", "amg"}
            Linear backend for K-cycle path. If `solver="amg"`, this is forced to "amg".
            If `solver="pcg"`, this must be "kcycle".
        stencil : {"5-point", "7-point"}
            Discretization stencil selector. Wrapper-level K-cycle currently supports
            "5-point"; 7-point prototype is exposed via solve_chebyshev_7point_3d(...).

        All remaining controls are forwarded to the selected backend.
        """
        form_mode = str(formulation).strip().lower()
        if form_mode not in {"confined", "unconfined"}:
            raise ValueError("formulation must be 'confined' or 'unconfined'.")

        if solver is None and (bool(transient) or form_mode == "unconfined"):
            solver_mode_raw = "kcycle"
        else:
            solver_mode_raw = self.solver_type if solver is None else solver
        solver_mode = str(solver_mode_raw).strip().lower()
        if solver_mode in {"multigrid", "mg"}:
            solver_mode = "kcycle"
        if solver_mode not in {"pcg", "kcycle", "amg"}:
            raise ValueError("solver must be one of: 'pcg', 'kcycle', 'amg'.")

        backend_mode = str(linear_backend).strip().lower()
        if backend_mode not in {"kcycle", "amg"}:
            raise ValueError("linear_backend must be 'kcycle' or 'amg'.")

        stencil_mode = _normalize_stencil_mode(stencil)

        if solver_mode == "amg":
            backend_mode = "amg"

        if solver_mode == "pcg":
            if form_mode != "confined":
                raise ValueError("PCG path currently supports only formulation='confined'.")
            if backend_mode != "kcycle":
                raise ValueError("linear_backend='amg' is not compatible with solver='pcg'.")
            if bool(transient):
                raise ValueError("Transient stream currently routes through Chebyshev/K-cycle, not solver='pcg'.")
            if gh_alpha_field is not None:
                raise ValueError("gh_alpha_field is not supported with solver='pcg'.")
            if str(smoother).strip().lower() != "chebyshev":
                raise ValueError("smoother switch applies only to K-cycle; use smoother='chebyshev' with PCG.")
            if stencil_mode != "5-point":
                raise ValueError("solver='pcg' currently supports only stencil='5-point'.")

            head_out, info = self._solve_pcg_device_loop(
                max_iter=int(pcg_max_iter),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                initial_head=initial_head,
                aq_thickness=float(self.aq_thickness if aq_thickness is None else aq_thickness),
            )
            if return_info:
                info_out = dict(info) if isinstance(info, dict) else {}
                info_out["formulation"] = "confined"
                info_out["solve_wrapper_solver"] = "pcg"
                info_out["stencil"] = str(stencil_mode)
                return head_out, info_out
            return head_out

        # K-cycle / AMG dispatch (including unconfined Picard wrapper path).
        return self.solve_multigrid_kcycle(
            max_cycles=int(max_cycles),
            nu_pre=int(nu_pre),
            nu_post=int(nu_post),
            nu_coarse=int(nu_coarse),
            omega=float(omega),
            rel_tol=float(rel_tol),
            abs_tol_min=float(abs_tol_min),
            initial_head=initial_head,
            aq_thickness=aq_thickness,
            max_levels=int(max_levels),
            return_info=bool(return_info),
            check_every_no=int(check_every_no),
            dh_rms_tol=dh_rms_tol,
            dh_max_tol=dh_max_tol,
            dh_max_factor=float(dh_max_factor),
            linear_backend=str(backend_mode),
            stencil=str(stencil_mode),
            amg_max_iter=amg_max_iter,
            amg_cycle=str(amg_cycle),
            smoother=str(smoother),
            cheby_lambda_min=float(cheby_lambda_min),
            cheby_lambda_max=float(cheby_lambda_max),
            gh_alpha_field=gh_alpha_field,
            refresh_diag_with_spatial_alpha=bool(refresh_diag_with_spatial_alpha),
            unconfined=(form_mode == "unconfined"),
            K_field=K_field,
            zbot_field=zbot_field,
            unconfined_min_sat=float(unconfined_min_sat),
            unconfined_max_picard_iter=int(unconfined_max_picard_iter),
            unconfined_relax=float(unconfined_relax),
            unconfined_head_tol=float(unconfined_head_tol),
            transient=bool(transient),
            storage_coeff=storage_coeff,
            dt=dt,
            head_prev=head_prev,
            refresh_diag_with_transient_storage=bool(refresh_diag_with_transient_storage),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always release references deterministically at scope exit.
        self.close()
        return False

    def close(self) -> None:
        """
        Release all Warp device arrays and graph objects held by this solver.

        Note: this does not guarantee VRAM is returned to the OS driver, because Warp uses a
        CUDA mempool allocator. However, it *does* drop Python references so the pool can
        reuse the memory for later allocations. Warp distinguishes pool "used" memory from
        "reserved" memory. :contentReference[oaicite:2]{index=2}
        """
        # 1) Break graph references first, because graphs can keep arrays alive.
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

        # 2) Drop CPU staging buffers you created for kcycle (they are Warp arrays on CPU).
        if hasattr(self, "_kcycle_stage_b"):
            self._kcycle_stage_b = None
        if hasattr(self, "_kcycle_stage_x"):
            self._kcycle_stage_x = None
        if hasattr(self, "_stage_R0"):
            self._stage_R0 = None
        if hasattr(self, "_stage_R0_host"):
            self._stage_R0_host = None

        # 3) Drop multigrid hierarchy objects (these contain many Warp arrays).
        if self.mg_levels is not None:
            for lvl in self.mg_levels:
                try:
                    for name in lvl.__slots__:
                        setattr(lvl, name, None)
                except Exception:
                    pass
            self.mg_levels = None

        # 4) Drop 2-level cache and level containers
        self._fine_level = None
        self._coarse_level = None

        # 5) Drop MG work dict buffers
        if self._mg_work is not None:
            try:
                self._mg_work.clear()
            except Exception:
                pass
        self._mg_work = None
        self._mg_work_built = False
        self._stage_ghw_levels = None
        self._stage_storage_levels = None
        self._kcycle_spatial_alpha_active = False
        self._kcycle_transient_active = False
        self._kcycle_last_gh_width_eff_levels = None

        # 6) Drop all device arrays on the solver itself
        self.T_wp = None
        self.R_wp = None
        self.active_wp = None
        self.bc_mask_wp = None
        self.bc_values_wp = None
        self.gh_mask_wp = None
        self.gh_head_wp = None
        self.gh_width_wp = None

        self.M_inv_wp = None
        self.storage_wp = None

        self.b_wp = None
        self.x_wp = None
        self.r_wp = None
        self.z_wp = None
        self.p_wp = None
        self.Ap_wp = None

        self.rho_buf = None
        self.rho_new_buf = None
        self.rTr_buf = None
        self.pAp_buf = None
        self.alpha_buf = None
        self.beta_buf = None
        self.converged_flag = None

        self.T_c_wp = None
        self.R_c_wp = None
        self.active_c_wp = None
        self.bc_mask_c_wp = None
        self.bc_values_c_wp = None
        self.gh_mask_c_wp = None
        self.gh_head_c_wp = None
        self.gh_width_c_wp = None
        self.M_inv_c_wp = None

        # 7) Optionally keep host arrays for reuse, but if you want to drop everything:
        # self.T_field_host = None
        # self.R_field_host = None
        # self.active_host = None
        # self.bc_mask_host = None
        # self.bc_values_host = None
        # self.gh_mask_host = None
        # self.gh_head_host = None
        # self.gh_width_host = None

        # 8) Force Python GC so __del__ for Warp arrays runs promptly
        gc.collect()

        # 9) Ensure all pending work is completed on the device
        try:
            wp.synchronize_device(self.device_str)
        except Exception:
            pass