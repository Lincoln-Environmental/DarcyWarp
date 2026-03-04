from __future__ import annotations
import warp as wp
from dataclasses import dataclass
from typing import Optional
import gc
import numpy as np
import os

from DARCY_WARP_PACKAGE.model_builder import (
    _build_domain,
    _build_dirichlet_boundary_mask,
    _build_dem,
    _model_bottom,
    build_base_fields,
    build_truth_inputs,
)

import ctypes.util

cuda_device_found = False

lib_path = ctypes.util.find_library("cuda")
if lib_path is None:
    lib_path = "libcuda.so.1"  # common on Linux

try:
    libcuda = ctypes.CDLL(lib_path)

    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int

    libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDeviceGetCount.restype = ctypes.c_int

    res = libcuda.cuInit(0)
    if res == 0:
        count = ctypes.c_int(0)
        res2 = libcuda.cuDeviceGetCount(ctypes.byref(count))
        if res2 == 0 and count.value > 0:
            cuda_device_found = True
            print(f"✅ CUDA driver reports {count.value} device(s).")
        else:
            print(f"❌ CUDA driver present, but no devices (res2={res2}, count={count.value}).")
    else:
        print(f"❌ cuInit failed (res={res}).")

except OSError as e:
    print(f"❌ Could not load CUDA driver library ({lib_path}): {e}, WARP must have CUDA!!!")

print(f"cuda_device_found = {cuda_device_found}")

# Optional GHB helper: use if available
try:
    from legacy_code.model_builder import _build_ghb_boundary_mask
except ImportError:
    _build_ghb_boundary_mask = None




_float_env = os.environ.get("DARCY_FLOAT", "float32")

if _float_env == "float64":
    WP_FLOAT = wp.float64
    NP_FLOAT = np.float64
elif _float_env == "float32":
    WP_FLOAT = wp.float32
    NP_FLOAT = np.float32
else:
    raise ValueError("DARCY_FLOAT must be 'float32' or 'float64'")

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
    T_c = (T_sum / count_safe).astype(np.float32, copy=False)
    R_c = (R_sum / count_safe).astype(np.float32, copy=False)

    # Zero out inactive coarse cells (matches loop behavior)
    inactive = active_c == 0
    if np.any(inactive):
        T_c[inactive] = np.float32(0.0)
        R_c[inactive] = np.float32(0.0)

    # GHB width: mean over block if any GHB present, else 0
    gh_width_sum = (
        _pad(gh_width_f, fill_value=0.0).astype(np.float64, copy=False).reshape(ny_c, 2, nx_c, 2).sum(axis=(1, 3))
    )
    gh_width_c = (gh_width_sum / count_safe).astype(np.float32, copy=False)
    gh_width_c[gh_mask_c == 0] = np.float32(0.0)

    bc_values_c = np.zeros((ny_c, nx_c), dtype=np.float32)
    gh_head_c = np.zeros((ny_c, nx_c), dtype=np.float32)

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

@wp.kernel
def jacobi_applyA_fused_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_width: wp.array(dtype=WP_FLOAT, ndim=2),
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

    sum_T = T_e + T_w + T_n + T_s +C_gh

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

    diagA = T_e + T_w + T_n + T_s + C_gh

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

    sum_T = T_e + T_w + T_n + T_s + C_gh

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

    sum_T = T_e + T_w + T_n + T_s + C_gh

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


def build_diag_preconditioner(
    T_field: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    gh_mask: np.ndarray | None = None,
    gh_width: np.ndarray | None = None,
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

    sum_T = wp.float64(T_e) + wp.float64(T_w) + wp.float64(T_n) + wp.float64(T_s) + wp.float64(C_gh)

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

    sum_T = T_e + T_w + T_n + T_s + C_gh

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

    sum_T = T_e + T_w + T_n + T_s + C_gh

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

        self.T_field_host = None
        self._T_field_wp_host = None
        self.T_field_dev = None

        self._operator_dirty = True

        # Hierarchy storage (for K cycle later)
        self.mg_levels = None
        # ---------------- CUDA graph cache (K-cycle path) ----------------
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

    def _invalidate_kcycle_graph(self) -> None:
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

    # -------------------------------------------------------------------------
    # Hierarchy (ready for K cycle, not used by 2-level solve yet)
    # -------------------------------------------------------------------------
    def build_hierarchy(
        self,
        max_levels: int,
        min_coarse_n: int = 4,
        min_coarse_cells: int | None = 500,
    ) -> None:
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
        :param min_coarse_cells: optional stop if nx*ny would drop below this
        """
        if int(max_levels) < 1:
            raise ValueError("max_levels must be >= 1")
        if min_coarse_cells is not None and int(min_coarse_cells) < 1:
            raise ValueError("min_coarse_cells must be >= 1 when provided")

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
            if min_coarse_cells is not None and (nx_c * ny_c) < int(min_coarse_cells):
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
        bc_values0 = np.asarray(self.bc_values_host, dtype=np.float32)

        if self.use_ghb:
            gh_mask0 = np.asarray(self.gh_mask_host, dtype=np.int32)
            gh_head0 = np.asarray(self.gh_head_host, dtype=np.float32)
            gh_width0 = np.asarray(self.gh_width_host, dtype=np.float32)
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

        T_pad = np.pad(np.asarray(T_f, dtype=np.float32), pad_spec, mode="edge")
        R_pad = np.pad(np.asarray(R_f, dtype=np.float32), pad_spec, mode="edge")
        active_pad = np.pad(np.asarray(active_f, dtype=np.int32), pad_spec, mode="edge")
        bc_mask_pad = np.pad(np.asarray(bc_mask_f, dtype=np.int32), pad_spec, mode="edge")
        bc_values_pad = np.pad(np.asarray(bc_values_f, dtype=np.float32), pad_spec, mode="edge")

        ny_p, nx_p = T_pad.shape
        ny_c = ny_p // 2
        nx_c = nx_p // 2

        T_blk = T_pad.reshape(ny_c, 2, nx_c, 2)
        R_blk = R_pad.reshape(ny_c, 2, nx_c, 2)
        T_c = T_blk.mean(axis=(1, 3), dtype=np.float64).astype(np.float32, copy=False)
        R_c = R_blk.mean(axis=(1, 3), dtype=np.float64).astype(np.float32, copy=False)

        a_blk = active_pad.reshape(ny_c, 2, nx_c, 2)
        m_blk = bc_mask_pad.reshape(ny_c, 2, nx_c, 2)
        active_c = a_blk.max(axis=(1, 3)).astype(np.int32, copy=False)
        bc_mask_c = m_blk.max(axis=(1, 3)).astype(np.int32, copy=False)

        bc_values_c = np.zeros((ny_c, nx_c), dtype=np.float32)

        if self.use_ghb and (gh_mask_f is not None) and (gh_width_f is not None) and (gh_head_f is not None):
            gh_mask_pad = np.pad(np.asarray(gh_mask_f, dtype=np.int32), pad_spec, mode="edge")
            gh_width_pad = np.pad(np.asarray(gh_width_f, dtype=np.float32), pad_spec, mode="edge")
            gh_head_pad = np.pad(np.asarray(gh_head_f, dtype=np.float32), pad_spec, mode="edge")

            ghm_blk = gh_mask_pad.reshape(ny_c, 2, nx_c, 2)
            gh_mask_c = ghm_blk.max(axis=(1, 3)).astype(np.int32, copy=False)

            ghm_f = ghm_blk.astype(np.float32, copy=False)
            ghw_blk = gh_width_pad.reshape(ny_c, 2, nx_c, 2)

            wsum = (ghw_blk * ghm_f).sum(axis=(1, 3), dtype=np.float64)
            msum = ghm_f.sum(axis=(1, 3), dtype=np.float64)

            gh_width_c = np.full((ny_c, nx_c), np.float32(dx_c), dtype=np.float32)
            on = msum > 0.0
            gh_width_c[on] = (wsum[on] / msum[on]).astype(np.float32, copy=False)

            gh_head_c = np.zeros((ny_c, nx_c), dtype=np.float32)

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
        eps = np.float32(max(1.0e-8 * dx_f, 1.0e-12))

        gh_mask2 = np.asarray(gh_mask, dtype=np.int32).copy()
        gh_width2 = np.asarray(gh_width, dtype=np.float32).copy()
        gh_head2 = np.asarray(gh_head, dtype=np.float32).copy()

        gh_mask2[np.asarray(active, dtype=np.int32) == 0] = 0
        gh_mask2[np.asarray(bc_mask, dtype=np.int32) != 0] = 0

        bad_w = ~np.isfinite(gh_width2)
        if np.any(bad_w):
            gh_width2[bad_w] = np.float32(dx_f)

        gh_width2[gh_mask2 == 0] = np.float32(dx_f)

        on = gh_mask2 != 0
        if np.any(on):
            gh_width2[on] = np.maximum(gh_width2[on], eps)

        bad_h = ~np.isfinite(gh_head2)
        if np.any(bad_h):
            gh_head2[bad_h] = np.float32(0.0)

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
        eps = np.float32(max(1.0e-8 * dx, 1.0e-12))

        gh_mask = np.asarray(self.gh_mask_host, dtype=np.int32)
        gh_width = np.asarray(self.gh_width_host, dtype=np.float32)

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
            gh_width[bad_w] = np.float32(dx)

        gh_width[gh_mask == 0] = np.float32(dx)

        gh_on = gh_mask != 0
        if np.any(gh_on):
            gh_width[gh_on] = np.maximum(gh_width[gh_on], eps)

        self.gh_mask_host = gh_mask
        self.gh_width_host = gh_width

        if self.gh_head_host is not None:
            gh_head = np.asarray(self.gh_head_host, dtype=np.float32).copy()
            bad_h = ~np.isfinite(gh_head)
            if np.any(bad_h):
                gh_head[bad_h] = np.float32(0.0)
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

        # Always update host copy for RHS consistency
        T_arr = np.asarray(T_truth, dtype=NP_FLOAT, order="C")
        if T_arr.shape != tuple(self.T_field_host.shape):
            if T_arr.size == 1:
                self.T_field_host[:, :] = T_arr.reshape(1)[0]
            else:
                raise ValueError(f"T_truth shape {T_arr.shape} expected {self.T_field_host.shape}")
        else:
            np.copyto(self.T_field_host, T_arr)

        # Prefer device-to-device copy if possible
        used_device_copy = False
        if hasattr(T_truth, "device") and hasattr(T_truth, "shape") and hasattr(T_truth, "numpy"):
            try:
                if str(T_truth.device) == str(self.device_str) and tuple(T_truth.shape) == tuple(self.T_wp.shape):
                    wp.copy(self.T_wp, T_truth)
                    used_device_copy = True
            except Exception:
                used_device_copy = False

        if not used_device_copy:
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
            min_coarse_cells: int | None = 500,
    ):
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
        """

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

        if aq_thickness is None:
            aq_thickness_f = float(self.aq_thickness)
        else:
            aq_thickness_f = float(aq_thickness)

        if not hasattr(self, "_kcycle_graph"):
            self._kcycle_graph = None
            self._kcycle_graph_shape = None

        if self.mg_levels is None:
            self.build_hierarchy(
                max_levels=int(max_levels),
                min_coarse_n=4,
                min_coarse_cells=min_coarse_cells,
            )

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
        self._build_rhs_fine(lvl0.b_wp, aq_thickness=float(aq_thickness_f))

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
            omega_f = float(omega)

            x_tmp_wp = level.Ax_wp
            x_in = level.x_wp
            x_out = x_tmp_wp

            for _ in range(int(nu_pre)):
                wp.launch(
                    kernel=jacobi_applyA_fused_kernel,
                    dim=dimL,
                    inputs=[
                        level.T_wp,
                        level.active_wp,
                        level.bc_mask_wp,
                        level.gh_mask_wp,
                        level.gh_width_wp,
                        level.b_wp,
                        x_in,
                        level.M_inv_wp,
                        level.bc_values_wp,
                        omega_f,
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

            for _ in range(int(nu_post)):
                wp.launch(
                    kernel=jacobi_applyA_fused_kernel,
                    dim=dimL,
                    inputs=[
                        level.T_wp,
                        level.active_wp,
                        level.bc_mask_wp,
                        level.gh_mask_wp,
                        level.gh_width_wp,
                        level.b_wp,
                        x_in,
                        level.M_inv_wp,
                        level.bc_values_wp,
                        omega_f,
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

        check_every = check_every_no  # reduce sync frequency; set to 1 for debugging

        graph_key = (
            "kcycle",
            int(len(levels)),
            tuple((int(l.ny), int(l.nx)) for l in levels),
            int(nu_pre),
            int(nu_post),
            int(nu_coarse),
            float(omega),
            float(self.gh_alpha),
            float(aq_thickness_f),
        )

        graph_built_this_call = False

        dh_rms_lastcheck = float("nan")
        dh_max_lastcheck = float("nan")

        for cyc in range(int(max_cycles)):
            n_cycles_used = cyc + 1

            if self._kcycle_graph is None or self._kcycle_graph_shape != graph_key:
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
            if dh_max_tol is not None and dh_rms_tol is not None:
                dh_ok = dh_max_lastcheck <= float(dh_max_tol) and dh_rms_lastcheck <= float(dh_rms_tol)

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
            "n_levels": int(len(levels)),
            "max_cycles": int(max_cycles),
            "n_cycles_used": int(n_cycles_used),
            "nu_pre": int(nu_pre),
            "nu_post": int(nu_post),
            "nu_coarse": int(nu_coarse),
            "omega": float(omega),
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
            "cuda_graph_reused": bool((not graph_built_this_call) and (self._kcycle_graph is not None)),
            "cuda_graph_built_this_call": bool(graph_built_this_call),
            "check_every": int(check_every),
            "min_coarse_cells": None if min_coarse_cells is None else int(min_coarse_cells),
        }

        return (head_out, info) if return_info else head_out

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
