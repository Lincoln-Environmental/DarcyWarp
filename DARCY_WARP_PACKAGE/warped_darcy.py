from __future__ import annotations
import warp as wp
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import gc
import numpy as np
import os
import time
import warnings

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




_float_env = os.environ.get("DARCY_FLOAT", "float64")

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


def _normalize_scalar_or_grid_to_shape(
    value: float | np.ndarray,
    *,
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, str]:
    """
    Normalize scalar-or-grid input to a (ny, nx) float64 array.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        val = float(arr)
        if not np.isfinite(val):
            raise ValueError(f"{name} scalar must be finite.")
        return np.full(shape, val, dtype=np.float64), "scalar"

    if arr.ndim == 2 and arr.shape == shape:
        return arr.astype(np.float64, copy=False), "grid"
    if arr.ndim == 3 and arr.shape[0] == 1 and tuple(arr.shape[1:]) == shape:
        return np.asarray(arr[0], dtype=np.float64), "grid"

    raise ValueError(f"{name} must be a scalar or shape {shape}. Got {arr.shape}.")


def specific_storage_potential(
    *,
    head: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_storage: float,
) -> np.ndarray:
    """
    Specific-storage potential per unit plan area for a convertible layer.
    """
    head_arr = np.asarray(head, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    rel = head_arr - bottom_arr
    ss = float(specific_storage)

    phi = np.zeros(np.broadcast_shapes(head_arr.shape, bottom_arr.shape, top_arr.shape), dtype=np.float64)
    rel_b = np.broadcast_to(rel, phi.shape)
    thickness_b = np.broadcast_to(thickness, phi.shape)

    partial = (rel_b > 0.0) & (rel_b < thickness_b)
    full = rel_b >= thickness_b
    phi[partial] = 0.5 * ss * rel_b[partial] * rel_b[partial]
    phi[full] = (
        0.5 * ss * thickness_b[full] * thickness_b[full]
        + ss * thickness_b[full] * (rel_b[full] - thickness_b[full])
    )
    return phi


def secant_specific_yield_coeff(
    *,
    head_ref: np.ndarray,
    head_old: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_yield: float,
    secant_eps: float = 1.0e-12,
) -> np.ndarray:
    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    head_old_arr = np.asarray(head_old, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    sat_ref = np.clip(head_ref_arr - bottom_arr, 0.0, thickness)
    sat_old = np.clip(head_old_arr - bottom_arr, 0.0, thickness)
    dh = head_ref_arr - head_old_arr

    coeff = np.zeros_like(np.broadcast_to(dh, np.broadcast_shapes(dh.shape, thickness.shape)), dtype=np.float64)
    moving = np.abs(dh) > float(secant_eps)
    coeff[moving] = float(specific_yield) * ((sat_ref[moving] - sat_old[moving]) / dh[moving])
    fallback = (~moving) & (head_ref_arr > bottom_arr) & (head_ref_arr < top_arr)
    coeff[fallback] = float(specific_yield)
    return np.clip(coeff, 0.0, float(specific_yield))


def secant_specific_storage_coeff(
    *,
    head_ref: np.ndarray,
    head_old: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_storage: float,
    secant_eps: float = 1.0e-12,
) -> np.ndarray:
    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    head_old_arr = np.asarray(head_old, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    dh = head_ref_arr - head_old_arr
    phi_ref = specific_storage_potential(
        head=head_ref_arr,
        bottom=bottom_arr,
        top=top_arr,
        specific_storage=float(specific_storage),
    )
    phi_old = specific_storage_potential(
        head=head_old_arr,
        bottom=bottom_arr,
        top=top_arr,
        specific_storage=float(specific_storage),
    )
    coeff = np.zeros_like(phi_ref, dtype=np.float64)
    moving = np.abs(dh) > float(secant_eps)
    coeff[moving] = (phi_ref[moving] - phi_old[moving]) / dh[moving]
    fallback = ~moving
    if np.any(fallback):
        saturated_thickness = np.clip(head_ref_arr - bottom_arr, 0.0, thickness)
        coeff[fallback] = float(specific_storage) * saturated_thickness[fallback]
    return np.maximum(coeff, 0.0)


def exact_unconfined_storage_terms(
    *,
    head_new: np.ndarray,
    head_old: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_yield: float,
    specific_storage: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return storage, Sy, and Ss terms per unit plan area per time.

    Positive values mean water has entered storage; mass-balance release is the
    negative of the returned total term.
    """
    head_new_arr = np.asarray(head_new, dtype=np.float64)
    head_old_arr = np.asarray(head_old, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    sat_new = np.clip(head_new_arr - bottom_arr, 0.0, thickness)
    sat_old = np.clip(head_old_arr - bottom_arr, 0.0, thickness)
    sy_term = float(specific_yield) * (sat_new - sat_old) / float(dt)
    ss_term = (
        specific_storage_potential(
            head=head_new_arr,
            bottom=bottom_arr,
            top=top_arr,
            specific_storage=float(specific_storage),
        )
        - specific_storage_potential(
            head=head_old_arr,
            bottom=bottom_arr,
            top=top_arr,
            specific_storage=float(specific_storage),
        )
    ) / float(dt)
    return sy_term + ss_term, sy_term, ss_term


def _chebyshev_update_weights(
    order: int,
    lambda_min_fraction: float,
) -> tuple[float, ...]:
    """
    Build bounded Chebyshev-style weights for nonlinear Picard update damping.
    """
    m = int(order)
    if m <= 0:
        return tuple()

    lam_hi = 1.0
    lam_lo = max(1.0e-12, min(float(lambda_min_fraction), 0.999999 * lam_hi))
    c = 0.5 * (lam_hi + lam_lo)
    d = 0.5 * (lam_hi - lam_lo)

    out: list[float] = []
    for k in range(1, m + 1):
        theta_k = np.pi * (2.0 * float(k) - 1.0) / (2.0 * float(m))
        denom = c - d * float(np.cos(theta_k))
        if denom <= 1.0e-12:
            denom = 1.0e-12
        out.append(float(1.0 / denom))
    return tuple(out)


def _chebyshev_relaxation_sequence(
    order: int,
    lambda_min: float,
    lambda_max: float,
) -> tuple[float, ...]:
    """
    Build Chebyshev semi-iteration relaxation factors for weighted Jacobi updates.
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


def _top_switch_above_mask(
    *,
    head_ref: np.ndarray,
    top: np.ndarray,
    threshold_mode: str,
    free_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the raw above-top mask used by the MF6-like top-switch storativity.

    :param head_ref: Reference head field.
    :param top: Model-top elevation field.
    :param threshold_mode: ``"ge"`` or ``"gt"``.
    :param free_mask: Optional active non-Dirichlet mask.
    :return: Boolean above-top mask.
    """
    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    mode = str(threshold_mode).strip().lower()
    if mode == "ge":
        mask = head_ref_arr >= top_arr
    elif mode == "gt":
        mask = head_ref_arr > top_arr
    else:
        raise ValueError("threshold_mode must be 'ge' or 'gt'.")
    if free_mask is not None:
        mask = mask & np.asarray(free_mask, dtype=bool)
    return mask


def _apply_top_switch_hysteresis(
    *,
    raw_above_top: np.ndarray,
    head_ref: np.ndarray,
    top: np.ndarray,
    previous_above_top: np.ndarray | None,
    hysteresis_eps: float,
    free_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply a symmetric top-surface hysteresis band to the above-top mask.

    :param raw_above_top: Raw top-switch mask from the current reference head.
    :param head_ref: Current reference head field.
    :param top: Model-top elevation field.
    :param previous_above_top: Previous effective above-top mask.
    :param hysteresis_eps: Hysteresis half-band around ``top``.
    :param free_mask: Active non-Dirichlet mask.
    :return: Stabilised above-top mask.
    """
    raw_mask = np.asarray(raw_above_top, dtype=bool)
    free = np.asarray(free_mask, dtype=bool)
    if previous_above_top is None:
        out = raw_mask.copy()
        out[~free] = False
        return out

    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    prev_mask = np.asarray(previous_above_top, dtype=bool)
    eps = max(float(hysteresis_eps), 0.0)

    stay_above = prev_mask & (head_ref_arr >= (top_arr - eps))
    stay_below = (~prev_mask) & (head_ref_arr < (top_arr + eps))

    out = raw_mask.copy()
    out[stay_above] = True
    out[stay_below] = False
    out[~free] = False
    return out


def _storage_active_set_change_metrics(
    *,
    current_mask: np.ndarray | None,
    previous_mask: np.ndarray | None,
    free_mask: np.ndarray,
) -> tuple[int, float]:
    """
    Compute active-set change count and fraction over active non-Dirichlet cells.

    :param current_mask: Current effective above-top mask.
    :param previous_mask: Previous effective above-top mask.
    :param free_mask: Active non-Dirichlet mask.
    :return: ``(changed_count, changed_fraction)``.
    """
    free = np.asarray(free_mask, dtype=bool)
    denom = int(np.count_nonzero(free))
    if current_mask is None or previous_mask is None or denom == 0:
        return 0, 0.0
    changed = np.asarray(current_mask, dtype=bool) ^ np.asarray(previous_mask, dtype=bool)
    changed[~free] = False
    count = int(np.count_nonzero(changed))
    return count, float(count / denom)



def _prepare_5point_transient_terms(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, dict]:
    """
    Prepare RHS and storage diagonal for an optional 2D transient step.

    The transient term is backward Euler on active non-boundary cells:
    ``storage_diag = storage_coeff * dx**2 / dt`` and
    ``rhs += storage_diag * head_prev``. Fixed-head boundary cells keep their
    boundary value in ``head_prev`` and receive zero storage contribution.
    Inactive cells also receive zero storage contribution.

    Parameters are accepted as host NumPy arrays or scalar storage input. The
    function returns copies and never mutates the caller's RHS or storage array.

    Returns
    -------
    tuple
        ``(b_eff, storage_diag, head_prev_used, dt_used, info)`` where
        ``info`` records whether storage was scalar/grid input and where the
        previous-time head came from.
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

    info = {
        "transient": bool(transient),
        "dt": float(dt) if transient and dt is not None else float("nan"),
        "storage_coeff_mode": "none",
        "head_prev_source": "none"
    }

    if not bool(transient):
        return b, sdiag, None, float("nan"), info

    dt_f = float(dt) if dt is not None else float("nan")
    if not np.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError("transient=True requires dt > 0.")
    if storage_coeff is None:
        raise ValueError("transient=True requires storage_coeff.")

    dx_f = float(dx)
    if dx_f <= 0.0:
        raise ValueError("dx must be positive for transient terms.")

    vol = np.float64(dx_f * dx_f)

    s_in = np.asarray(storage_coeff, dtype=NP_FLOAT)
    if s_in.ndim == 0:
        Scoeff = np.full(shape, NP_FLOAT(s_in.reshape(()).item()), dtype=NP_FLOAT)
        info["storage_coeff_mode"] = "scalar"
    else:
        if s_in.shape != shape:
            raise ValueError(f"storage_coeff shape {s_in.shape} expected {shape}")
        Scoeff = np.asarray(s_in, dtype=NP_FLOAT)
        info["storage_coeff_mode"] = "grid"

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
        info["head_prev_source"] = "head_prev"
        if h_prev.shape != shape:
            raise ValueError(f"head_prev shape {h_prev.shape} expected {shape}")
    elif initial_head is not None:
        h_prev = np.asarray(initial_head, dtype=NP_FLOAT).copy()
        info["head_prev_source"] = "initial_head"
        if h_prev.shape != shape:
            raise ValueError(f"initial_head shape {h_prev.shape} expected {shape}")
    else:
        h_prev = np.zeros(shape, dtype=NP_FLOAT)
        info["head_prev_source"] = "zeros"

    h_prev[bcm != 0] = bcv[bcm != 0]
    h_prev[act == 0] = NP_FLOAT(0.0)
    if not np.all(np.isfinite(h_prev)):
        raise ValueError("head_prev contains non-finite values.")

    b[free] = (
        b[free].astype(np.float64, copy=False)
        + sdiag_add[free].astype(np.float64, copy=False) * h_prev[free].astype(np.float64, copy=False)
    ).astype(NP_FLOAT, copy=False)
    sdiag[free] = sdiag_add[free]

    return b, sdiag, h_prev, float(dt_f), info

def _compute_ghb_factor_from_raw_fields(
    *,
    gh_mask: np.ndarray,
    gh_width: np.ndarray,
    gh_alpha: float | np.ndarray,
    aq_thickness: float | np.ndarray,
    dx: float,
    active: np.ndarray | None = None,
    bc_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """
    Compute ghb_factor = gh_alpha * gh_width * dx / aq_thickness on valid GHB cells.
    Returns (ghb_factor, gh_alpha_grid, aq_thickness_grid, mode_summary).
    """
    gh_mask_i = np.asarray(gh_mask, dtype=np.int32)
    gh_width_f = np.asarray(gh_width, dtype=np.float64)
    shape = tuple(gh_mask_i.shape)
    if gh_width_f.shape != shape:
        raise ValueError(f"gh_width shape {gh_width_f.shape} must match gh_mask shape {shape}.")

    gh_alpha_grid, gh_alpha_mode = _normalize_scalar_or_grid_to_shape(
        gh_alpha,
        shape=shape,
        name="gh_alpha",
    )
    aq_thickness_grid, aq_mode = _normalize_scalar_or_grid_to_shape(
        aq_thickness,
        shape=shape,
        name="aq_thickness",
    )

    gh_on = gh_mask_i != 0
    if active is not None:
        gh_on &= np.asarray(active, dtype=np.int32) != 0
    if bc_mask is not None:
        gh_on &= np.asarray(bc_mask, dtype=np.int32) == 0
    gh_on &= np.isfinite(gh_width_f) & (gh_width_f > 0.0)

    if np.any(gh_on):
        thick_used = np.asarray(aq_thickness_grid[gh_on], dtype=np.float64)
        alpha_used = np.asarray(gh_alpha_grid[gh_on], dtype=np.float64)
        bad_thick = (~np.isfinite(thick_used)) | (thick_used <= 0.0)
        # `gh_alpha` may be either a direct scaling factor or an "effective alpha"
        # derived from a blend coefficient. The effective form can exceed 1.0, so
        # only finite positive values are valid here.
        bad_alpha = (~np.isfinite(alpha_used)) | (alpha_used <= 0.0)
        if np.any(bad_thick):
            raise ValueError("aq_thickness must be finite and > 0 on active GHB cells.")
        if np.any(bad_alpha):
            raise ValueError("gh_alpha must be finite and > 0 on active GHB cells.")

    ghb_factor = np.zeros(shape, dtype=np.float64)
    if np.any(gh_on):
        ghb_factor[gh_on] = (
            gh_alpha_grid[gh_on]
            * gh_width_f[gh_on]
            * float(dx)
            / aq_thickness_grid[gh_on]
        )

    return (
        ghb_factor,
        gh_alpha_grid,
        aq_thickness_grid,
        {"gh_alpha": gh_alpha_mode, "aq_thickness": aq_mode},
    )


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

    if (gh_mask is not None) and (gh_head is not None) and ((gh_width is not None) or (ghb_factor is not None)):
        ghm = np.asarray(gh_mask, dtype=np.int32) != 0
        ghe = np.asarray(gh_head, dtype=np.float64)
        if ghe.shape != (ny, nx):
            raise ValueError("gh_head shape mismatch")

        if ghb_factor is not None:
            ghbf = np.asarray(ghb_factor, dtype=np.float64)
            if ghbf.shape != (ny, nx):
                raise ValueError("ghb_factor shape mismatch")
            Cgh = (T * ghbf).astype(np.float64, copy=False)
            gh_ok = np.isfinite(ghbf) & (ghbf > 0.0)
        else:
            ghw = np.asarray(gh_width, dtype=np.float64)
            if ghw.shape != (ny, nx):
                raise ValueError("gh_width shape mismatch")
            if float(aq_thickness) <= 0.0:
                raise ValueError("aq_thickness must be positive")
            Cgh = (float(gh_alpha) * T / float(aq_thickness) * ghw * dx_f).astype(np.float64)
            gh_ok = np.isfinite(ghw) & (ghw > 0.0)

        mask_gh = (
            ghm
            & cell_is_interior
            & gh_ok
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
    ghb_factor_f: np.ndarray | None = None,
    storage_diag_f: np.ndarray | None = None,
):
    return _coarsen_level_host_2x2(
        T_f=T_f,
        R_f=R_f,
        active_f=active_f,
        bc_mask_f=bc_mask_f,
        bc_values_f=bc_values_f,
        gh_mask_f=gh_mask_f,
        gh_head_f=gh_head_f,
        gh_width_f=gh_width_f,
        ghb_factor_f=ghb_factor_f,
        storage_diag_f=storage_diag_f,
    )


def _pad_to_coarse_block_shape(
    arr: np.ndarray,
    *,
    pad_y: int,
    pad_x: int,
    dtype,
    fill_value,
) -> np.ndarray:
    return np.pad(
        np.asarray(arr, dtype=dtype),
        ((0, int(pad_y)), (0, int(pad_x))),
        mode="constant",
        constant_values=fill_value,
    )


def _harmonic_mean_2x2_coarsen(T_pad: np.ndarray, active_pad: np.ndarray) -> np.ndarray:
    ny_c = int(T_pad.shape[0] // 2)
    nx_c = int(T_pad.shape[1] // 2)

    T_blk = np.asarray(T_pad, dtype=np.float64).reshape(ny_c, 2, nx_c, 2)
    a_blk = np.asarray(active_pad, dtype=np.int32).reshape(ny_c, 2, nx_c, 2)

    use = (a_blk != 0) & np.isfinite(T_blk) & (T_blk > 0.0)
    inv_blk = np.zeros_like(T_blk, dtype=np.float64)
    inv_blk[use] = 1.0 / T_blk[use]

    denom = inv_blk.sum(axis=(1, 3), dtype=np.float64)
    count = use.sum(axis=(1, 3), dtype=np.int32)

    T_c = np.zeros((ny_c, nx_c), dtype=np.float64)
    ok = (count > 0) & (denom > 0.0)
    T_c[ok] = count[ok] / denom[ok]
    return T_c.astype(NP_FLOAT, copy=False)


def _mean_2x2_with_mask(values_pad: np.ndarray, mask_pad: np.ndarray) -> np.ndarray:
    ny_c = int(values_pad.shape[0] // 2)
    nx_c = int(values_pad.shape[1] // 2)

    v_blk = np.asarray(values_pad, dtype=np.float64).reshape(ny_c, 2, nx_c, 2)
    m_blk = np.asarray(mask_pad, dtype=np.float64).reshape(ny_c, 2, nx_c, 2)

    wsum = (v_blk * m_blk).sum(axis=(1, 3), dtype=np.float64)
    msum = m_blk.sum(axis=(1, 3), dtype=np.float64)

    out = np.zeros((ny_c, nx_c), dtype=np.float64)
    on = msum > 0.0
    out[on] = wsum[on] / msum[on]
    return out.astype(NP_FLOAT, copy=False)


def _sum_2x2_with_mask(values_pad: np.ndarray, mask_pad: np.ndarray) -> np.ndarray:
    ny_c = int(values_pad.shape[0] // 2)
    nx_c = int(values_pad.shape[1] // 2)

    v_blk = np.asarray(values_pad, dtype=np.float64).reshape(ny_c, 2, nx_c, 2)
    m_blk = np.asarray(mask_pad, dtype=np.float64).reshape(ny_c, 2, nx_c, 2)

    out = (v_blk * m_blk).sum(axis=(1, 3), dtype=np.float64)
    return out.astype(NP_FLOAT, copy=False)


def _coarsen_active_mask_2x2(active_pad: np.ndarray, valid_pad: np.ndarray) -> np.ndarray:
    ny_c = int(active_pad.shape[0] // 2)
    nx_c = int(active_pad.shape[1] // 2)

    a_blk = np.asarray(active_pad, dtype=np.int32).reshape(ny_c, 2, nx_c, 2)
    v_blk = np.asarray(valid_pad, dtype=np.int32).reshape(ny_c, 2, nx_c, 2)

    active_count = a_blk.sum(axis=(1, 3), dtype=np.int32)
    valid_count = v_blk.sum(axis=(1, 3), dtype=np.int32)

    active_c = np.zeros((ny_c, nx_c), dtype=np.int32)
    on = valid_count > 0
    active_c[on] = ((2 * active_count[on]) >= valid_count[on]).astype(np.int32, copy=False)
    return active_c


def _coarsen_level_host_2x2(
    T_f: np.ndarray,
    R_f: np.ndarray,
    active_f: np.ndarray,
    bc_mask_f: np.ndarray,
    bc_values_f: np.ndarray,
    gh_mask_f: np.ndarray | None,
    gh_head_f: np.ndarray | None,
    gh_width_f: np.ndarray | None,
    ghb_factor_f: np.ndarray | None = None,
    storage_diag_f: np.ndarray | None = None,
):
    del bc_values_f, gh_head_f

    ny_f, nx_f = np.asarray(T_f).shape
    nx_c = (int(nx_f) + 1) // 2
    ny_c = (int(ny_f) + 1) // 2

    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    valid_pad = _pad_to_coarse_block_shape(
        np.ones((ny_f, nx_f), dtype=np.int32),
        pad_y=pad_y,
        pad_x=pad_x,
        dtype=np.int32,
        fill_value=0,
    )
    active_pad = _pad_to_coarse_block_shape(
        active_f,
        pad_y=pad_y,
        pad_x=pad_x,
        dtype=np.int32,
        fill_value=0,
    )
    bc_mask_pad = _pad_to_coarse_block_shape(
        bc_mask_f,
        pad_y=pad_y,
        pad_x=pad_x,
        dtype=np.int32,
        fill_value=0,
    )

    T_pad = _pad_to_coarse_block_shape(
        T_f,
        pad_y=pad_y,
        pad_x=pad_x,
        dtype=NP_FLOAT,
        fill_value=0.0,
    )
    R_pad = _pad_to_coarse_block_shape(
        R_f,
        pad_y=pad_y,
        pad_x=pad_x,
        dtype=NP_FLOAT,
        fill_value=0.0,
    )

    active_c = _coarsen_active_mask_2x2(active_pad, valid_pad)

    m_blk = bc_mask_pad.reshape(ny_c, 2, nx_c, 2)
    bc_mask_c_raw = m_blk.max(axis=(1, 3)).astype(np.int32, copy=False)
    bc_mask_c = ((bc_mask_c_raw != 0) & (active_c != 0)).astype(np.int32, copy=False)

    T_c = _harmonic_mean_2x2_coarsen(T_pad=T_pad, active_pad=active_pad)
    R_c = _mean_2x2_with_mask(values_pad=R_pad, mask_pad=valid_pad)

    inactive = active_c == 0
    if np.any(inactive):
        T_c = np.asarray(T_c, dtype=NP_FLOAT).copy()
        R_c = np.asarray(R_c, dtype=NP_FLOAT).copy()
        T_c[inactive] = NP_FLOAT(0.0)
        R_c[inactive] = NP_FLOAT(0.0)

    ghb_enabled = (
        gh_mask_f is not None
        and (
            ghb_factor_f is not None
            or gh_width_f is not None
        )
    )

    if ghb_enabled:
        gh_mask_pad = _pad_to_coarse_block_shape(
            gh_mask_f,
            pad_y=pad_y,
            pad_x=pad_x,
            dtype=np.int32,
            fill_value=0,
        )
        gh_width_pad = _pad_to_coarse_block_shape(
            gh_width_f,
            pad_y=pad_y,
            pad_x=pad_x,
            dtype=NP_FLOAT,
            fill_value=0.0,
        )
        ghb_factor_pad = _pad_to_coarse_block_shape(
            ghb_factor_f,
            pad_y=pad_y,
            pad_x=pad_x,
            dtype=NP_FLOAT,
            fill_value=0.0,
        )

        ghm_blk = gh_mask_pad.reshape(ny_c, 2, nx_c, 2)
        gh_mask_c_raw = ghm_blk.max(axis=(1, 3)).astype(np.int32, copy=False)
        gh_mask_c = (
            (gh_mask_c_raw != 0)
            & (active_c != 0)
            & (bc_mask_c == 0)
        ).astype(np.int32, copy=False)

        gh_width_c = _mean_2x2_with_mask(values_pad=gh_width_pad, mask_pad=gh_mask_pad)
        gh_width_c = np.asarray(gh_width_c, dtype=NP_FLOAT)
        gh_width_c[gh_mask_c == 0] = NP_FLOAT(0.0)

        ghb_blk = np.asarray(ghb_factor_pad, dtype=np.float64).reshape(ny_c, 2, nx_c, 2)
        ghm_blk = np.asarray(gh_mask_pad, dtype=np.int32).reshape(ny_c, 2, nx_c, 2)
        valid_blk = (ghm_blk != 0) & np.isfinite(ghb_blk) & (ghb_blk > 0.0)
        valid_f = valid_blk.astype(np.float64, copy=False)
        ghb_sum = (ghb_blk * valid_f).sum(axis=(1, 3), dtype=np.float64)
        ghb_cnt = valid_f.sum(axis=(1, 3), dtype=np.float64)
        ghb_factor_c = np.zeros((ny_c, nx_c), dtype=NP_FLOAT)
        on = ghb_cnt > 0.0
        ghb_factor_c[on] = (ghb_sum[on] / ghb_cnt[on]).astype(NP_FLOAT, copy=False)
        ghb_factor_c[gh_mask_c == 0] = NP_FLOAT(0.0)
    else:
        gh_mask_c = np.zeros((ny_c, nx_c), dtype=np.int32)
        gh_width_c = np.zeros((ny_c, nx_c), dtype=NP_FLOAT)
        ghb_factor_c = np.zeros((ny_c, nx_c), dtype=NP_FLOAT)


    if storage_diag_f is not None:
        storage_diag_pad = _pad_to_coarse_block_shape(
            storage_diag_f,
            pad_y=pad_y,
            pad_x=pad_x,
            dtype=NP_FLOAT,
            fill_value=0.0,
        )
        storage_diag_c = _sum_2x2_with_mask(values_pad=storage_diag_pad, mask_pad=valid_pad)
        storage_diag_c = np.asarray(storage_diag_c, dtype=NP_FLOAT)
        storage_diag_c[inactive] = NP_FLOAT(0.0)
        storage_diag_c[bc_mask_c != 0] = NP_FLOAT(0.0)
    else:
        # No storage active on the fine level: carry "no storage" as None
        # instead of allocating a zero array. Steady coarsening must not pay for
        # storage it does not use.
        storage_diag_c = None

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
        ghb_factor_c,
        storage_diag_c,
    )


def _select_unconfined_inner_max_cycles(
    *,
    previous_dh_measure: float | None,
    early_cycles: int,
    middle_cycles: int,
    late_cycles: int,
    middle_dh: float,
    late_dh: float,
) -> int:
    """
    Choose an adaptive inner K-cycle cap for the transient unconfined Picard loop.

    :param previous_dh_measure: Previous outer head-change measure, usually max abs dh.
    :param early_cycles: Inner-cycle cap for early Picard iterations.
    :param middle_cycles: Inner-cycle cap for middle Picard iterations.
    :param late_cycles: Inner-cycle cap for late Picard iterations.
    :param middle_dh: Threshold above which the solve remains in the early phase.
    :param late_dh: Threshold above which the solve remains in the middle phase.
    :return: Selected inner-cycle cap.
    """
    if previous_dh_measure is None or not np.isfinite(float(previous_dh_measure)):
        return int(early_cycles)
    dh_value = float(previous_dh_measure)
    if dh_value > float(middle_dh):
        return int(early_cycles)
    if dh_value > float(late_dh):
        return int(middle_cycles)
    return int(late_cycles)


@dataclass
class AdaptiveInnerSolveConfig:
    enabled: bool = False
    initial_block_cycles: int = 4
    min_block_cycles: int = 2
    max_block_cycles: int = 16
    min_total_cycles: int = 2
    eta_initial: float = 0.25
    eta_min: float = 0.02
    eta_max: float = 0.30
    eta_gamma: float = 0.5
    eta_power: float = 1.5
    good_contraction_ratio: float = 0.35
    weak_contraction_ratio: float = 0.85
    stall_contraction_ratio: float = 0.98
    divergence_contraction_ratio: float = 1.05
    stall_patience: int = 2
    minimum_usable_reduction_ratio: float = 0.80
    residual_floor: float = 1.0e-12
    relative_flow_residual_target: float = 1.0e-4
    save_block_history: bool = False


@dataclass
class AdaptiveInnerSolveState:
    total_cycles: int = 0
    block_index: int = 0
    previous_residual_rms: float = float("nan")
    initial_residual_rms: float = float("nan")
    target_residual_rms: float = float("nan")
    initial_relative_flow_residual_rms: float = float("nan")
    target_relative_flow_residual_rms: float = float("inf")
    final_relative_flow_residual_rms: float = float("nan")
    previous_relative_flow_residual_rms: float = float("nan")
    previous_outer_residual_rms: float = float("nan")
    previous_outer_dh_rms: float = float("nan")
    consecutive_stall_blocks: int = 0
    converged: bool = False
    stalled: bool = False
    diverged: bool = False
    fallback_used: bool = False
    legacy_fallback_used: bool = False
    target_achieved: bool = False
    usable_for_picard: bool = False
    rollback_count: int = 0
    residual_check_count: int = 0
    termination_reason: str = ""
    fallback_reason: str = ""
    forcing_eta: float = float("nan")
    final_residual_rms: float = float("nan")
    cycles_per_block: list[int] = field(default_factory=list)
    residuals_per_block: list[float] = field(default_factory=list)
    contraction_ratios: list[float] = field(default_factory=list)
    per_cycle_convergence_factors: list[float] = field(default_factory=list)
    flow_contraction_ratios: list[float] = field(default_factory=list)
    head_per_cycle_convergence_factors: list[float] = field(default_factory=list)
    flow_per_cycle_convergence_factors: list[float] = field(default_factory=list)
    controller_per_cycle_convergence_factors: list[float] = field(default_factory=list)
    head_reduction_ratios: list[float] = field(default_factory=list)
    flow_reduction_ratios: list[float] = field(default_factory=list)
    predicted_cycles_per_block: list[int | None] = field(default_factory=list)
    block_classifications: list[str] = field(default_factory=list)


def _select_legacy_unconfined_inner_max_cycles_from_dh(
    *,
    previous_dh_measure: float | None,
    early_cycles: int,
    middle_cycles: int,
    late_cycles: int,
    middle_dh: float,
    late_dh: float,
) -> int:
    """
    Choose the legacy DH-threshold inner K-cycle cap for the transient fast path.
    """
    return _select_unconfined_inner_max_cycles(
        previous_dh_measure=previous_dh_measure,
        early_cycles=early_cycles,
        middle_cycles=middle_cycles,
        late_cycles=late_cycles,
        middle_dh=middle_dh,
        late_dh=late_dh,
    )


def _adaptive_state_requires_legacy_fallback(state: AdaptiveInnerSolveState) -> bool:
    """An adaptive result may update Picard only when it is explicitly usable."""
    return bool(state.fallback_used or state.diverged or not state.usable_for_picard)


def _remaining_legacy_fallback_cycles(*, max_cycles: int, adaptive_cycles_used: int, selected_legacy_cycles: int) -> int:
    return min(
        max(int(max_cycles) - int(adaptive_cycles_used), 0),
        max(int(selected_legacy_cycles), 0),
    )


def _should_continue_picard_after_refreshed_acceptance(
    *,
    provisional_picard_acceptance: bool,
    refreshed_picard_acceptance: bool,
) -> bool:
    """A provisional linearised check never ends a period without refresh."""
    return bool(not (provisional_picard_acceptance and refreshed_picard_acceptance))


def _adaptive_practical_acceptance_allowed(
    *,
    practical_acceptance_enabled: bool,
    adaptive_controller_used: bool,
    inner_target_achieved: bool,
    final_relative_flow_residual_rms: float | None = None,
    relative_flow_target: float | None = None,
) -> bool:
    return bool(
        practical_acceptance_enabled
        and (not adaptive_controller_used or inner_target_achieved)
    )


def _adaptive_dt_dh_contraction_estimate(
    dh_history: list[float | None],
) -> tuple[float, float] | None:
    """Estimate (geometric-mean contraction ratio, last dh) from dh_max history.

    Uses up to the last four finite positive values (three ratios). Returns
    None when fewer than two usable values are available.
    """
    values = [
        float(v)
        for v in dh_history
        if v is not None and np.isfinite(float(v)) and float(v) > 0.0
    ]
    if len(values) < 2:
        return None
    window = values[-4:]
    ratios = [cur / prev for prev, cur in zip(window[:-1], window[1:]) if prev > 0.0]
    if not ratios:
        return None
    ratio = float(np.exp(np.mean(np.log(np.clip(ratios, 1.0e-12, None)))))
    return ratio, values[-1]


def _adaptive_dt_projected_outer_to_tol(
    dh_history: list[float | None],
    *,
    tol: float,
) -> float | None:
    """Project additional outer iterations for dh_max <= tol from contraction.

    Returns 0.0 when already at/below tol, inf when not contracting, and None
    when the history is insufficient for a projection.
    """
    estimate = _adaptive_dt_dh_contraction_estimate(dh_history)
    if estimate is None:
        return None
    ratio, current = estimate
    if current <= tol:
        return 0.0
    if ratio >= 1.0:
        return float("inf")
    return float(np.log(tol / current) / np.log(ratio))


def _adaptive_dt_should_early_shrink(
    dh_history: list[float | None],
    *,
    tol: float,
    outer_iterations_done: int,
    budget: int,
    min_outer: int,
) -> bool:
    """True when the contraction projection says strict cannot make budget.

    Only meaningful while dh_max is the binding strict constraint: when dh is
    already <= tol the blocker is the residual, which contracts on its own
    schedule, so no early shrink is signalled.
    """
    if int(outer_iterations_done) < int(min_outer):
        return False
    estimate = _adaptive_dt_dh_contraction_estimate(dh_history)
    if estimate is None:
        return False
    _, current = estimate
    if current <= tol:
        return False
    needed = _adaptive_dt_projected_outer_to_tol(dh_history, tol=tol)
    if needed is None:
        return False
    return float(outer_iterations_done) + needed > float(budget)


def _adaptive_dt_should_extend_budget(
    dh_history: list[float | None],
    *,
    tol: float,
    extension_factor: float,
    extension_contraction_ratio: float,
) -> bool:
    """True when strict is close enough that extending the budget beats a shrink.

    Qualifies when dh_max is within ``extension_factor`` of tol and is either
    already below tol (residual is the remaining blocker) or still contracting
    faster than ``extension_contraction_ratio``.
    """
    estimate = _adaptive_dt_dh_contraction_estimate(dh_history)
    if estimate is None:
        # A single finite value still qualifies on closeness alone.
        values = [
            float(v)
            for v in dh_history
            if v is not None and np.isfinite(float(v)) and float(v) > 0.0
        ]
        if not values:
            return False
        return bool(values[-1] <= float(extension_factor) * tol)
    ratio, current = estimate
    if current > float(extension_factor) * tol:
        return False
    if current <= tol:
        return True
    return bool(ratio < float(extension_contraction_ratio))


def _validate_adaptive_inner_solve_config(
    *,
    config: AdaptiveInnerSolveConfig,
    max_cycles: int,
) -> None:
    if int(config.min_block_cycles) < 1:
        raise ValueError("adaptive_inner_min_block_cycles must be >= 1.")
    if int(config.max_block_cycles) < int(config.min_block_cycles):
        raise ValueError("adaptive_inner_max_block_cycles must be >= adaptive_inner_min_block_cycles.")
    if not (int(config.min_block_cycles) <= int(config.initial_block_cycles) <= int(config.max_block_cycles)):
        raise ValueError(
            "adaptive_inner_initial_block_cycles must satisfy "
            "adaptive_inner_min_block_cycles <= initial <= adaptive_inner_max_block_cycles."
        )
    if int(config.max_block_cycles) > int(max_cycles):
        raise ValueError("adaptive_inner_max_block_cycles must be <= max_cycles.")
    if int(config.min_total_cycles) < 0:
        raise ValueError("adaptive_inner_min_total_cycles must be >= 0.")
    if int(config.min_total_cycles) > int(max_cycles):
        raise ValueError("adaptive_inner_min_total_cycles must be <= max_cycles.")
    if not (0.0 < float(config.eta_min) <= float(config.eta_initial) <= float(config.eta_max) < 1.0):
        raise ValueError("adaptive inner eta controls must satisfy 0 < eta_min <= eta_initial <= eta_max < 1.")
    if float(config.eta_gamma) <= 0.0 or not np.isfinite(float(config.eta_gamma)):
        raise ValueError("adaptive_inner_eta_gamma must be finite and > 0.")
    if float(config.eta_power) <= 0.0 or not np.isfinite(float(config.eta_power)):
        raise ValueError("adaptive_inner_eta_power must be finite and > 0.")
    good = float(config.good_contraction_ratio)
    weak = float(config.weak_contraction_ratio)
    stall = float(config.stall_contraction_ratio)
    divergence = float(config.divergence_contraction_ratio)
    if not (0.0 < good <= weak <= stall < divergence):
        raise ValueError(
            "adaptive inner contraction ratios must satisfy "
            "0 < good <= weak <= stall < divergence."
        )
    if int(config.stall_patience) < 1:
        raise ValueError("adaptive_inner_stall_patience must be >= 1.")
    usable_ratio = float(config.minimum_usable_reduction_ratio)
    if not (0.0 < usable_ratio <= 1.0):
        raise ValueError("adaptive_inner_minimum_usable_reduction_ratio must be in (0, 1].")
    if float(config.residual_floor) <= 0.0 or not np.isfinite(float(config.residual_floor)):
        raise ValueError("adaptive_inner_residual_floor must be finite and > 0.")
    if float(config.relative_flow_residual_target) < 0.0 or not np.isfinite(float(config.relative_flow_residual_target)):
        raise ValueError("adaptive_inner_relative_flow_residual_target must be finite and >= 0.")


def _build_adaptive_inner_solve_config_from_controls(
    *,
    controls: dict,
    max_cycles: int,
) -> AdaptiveInnerSolveConfig:
    config = AdaptiveInnerSolveConfig(
        enabled=bool(controls.get("adaptive_unconfined_inner_enabled", True)),
        initial_block_cycles=int(controls.get("adaptive_inner_initial_block_cycles", 5)),
        min_block_cycles=int(controls.get("adaptive_inner_min_block_cycles", 5)),
        max_block_cycles=int(controls.get("adaptive_inner_max_block_cycles", 20)),
        min_total_cycles=int(controls.get("adaptive_inner_min_total_cycles", 5)),
        eta_initial=float(controls.get("adaptive_inner_eta_initial", 0.05)),
        eta_min=float(controls.get("adaptive_inner_eta_min", 0.005)),
        eta_max=float(controls.get("adaptive_inner_eta_max", 0.10)),
        eta_gamma=float(controls.get("adaptive_inner_eta_gamma", 0.25)),
        eta_power=float(controls.get("adaptive_inner_eta_power", 1.5)),
        good_contraction_ratio=float(controls.get("adaptive_inner_good_contraction_ratio", 0.40)),
        weak_contraction_ratio=float(controls.get("adaptive_inner_weak_contraction_ratio", 0.90)),
        stall_contraction_ratio=float(controls.get("adaptive_inner_stall_contraction_ratio", 0.9995)),
        divergence_contraction_ratio=float(controls.get("adaptive_inner_divergence_contraction_ratio", 1.10)),
        stall_patience=int(controls.get("adaptive_inner_stall_patience", 8)),
        minimum_usable_reduction_ratio=float(
            controls.get("adaptive_inner_minimum_usable_reduction_ratio", 0.10)
        ),
        residual_floor=float(controls.get("adaptive_inner_residual_floor", 1.0e-12)),
        relative_flow_residual_target=float(
            controls.get("adaptive_inner_relative_flow_residual_target", 1.0e-4)
        ),
        save_block_history=bool(controls.get("adaptive_inner_save_block_history", False)),
    )
    _validate_adaptive_inner_solve_config(config=config, max_cycles=int(max_cycles))
    return config


def _compute_inner_forcing_eta(
    *,
    current_outer_residual_rms: float,
    previous_outer_residual_rms: float | None,
    config: AdaptiveInnerSolveConfig,
) -> float:
    current = float(current_outer_residual_rms)
    if not np.isfinite(current) or current < 0.0:
        raise ValueError("current_outer_residual_rms must be finite and >= 0.")
    if current <= float(config.residual_floor):
        return float(np.clip(float(config.eta_initial), float(config.eta_min), float(config.eta_max)))

    previous = float(previous_outer_residual_rms) if previous_outer_residual_rms is not None else float("nan")
    if not np.isfinite(previous) or previous <= float(config.residual_floor):
        return float(np.clip(float(config.eta_initial), float(config.eta_min), float(config.eta_max)))

    eta = float(config.eta_gamma) * (current / max(previous, float(config.residual_floor))) ** float(config.eta_power)
    return float(np.clip(eta, float(config.eta_min), float(config.eta_max)))


def _compute_inner_target_residual(
    *,
    initial_residual_rms: float,
    forcing_eta: float,
    residual_floor: float,
    inner_head_residual_tol_min: float,
    inner_head_residual_tol_max: float,
    inner_picard_scale_max_fraction: float,
    previous_outer_dh_rms: float | None,
    hclose: float,
) -> float:
    initial_residual = float(initial_residual_rms)
    if not np.isfinite(initial_residual) or initial_residual < 0.0:
        raise ValueError("initial_residual_rms must be finite and >= 0.")

    # Adaptive forcing targets are governed by the solve residual and a
    # numerical floor, not the production Picard tolerance.
    lower = float(residual_floor)
    upper = max(lower, float(inner_head_residual_tol_max))
    target = float(forcing_eta) * initial_residual

    previous_dh = float(previous_outer_dh_rms) if previous_outer_dh_rms is not None else float("nan")
    if np.isfinite(previous_dh):
        picard_bound = float(inner_picard_scale_max_fraction) * max(float(hclose), previous_dh)
        target = min(target, picard_bound)

    return float(min(upper, max(lower, target)))


def _classify_inner_contraction(
    *,
    residual_before: float,
    residual_after: float,
    block_cycles: int,
    config: AdaptiveInnerSolveConfig,
) -> dict[str, Any]:
    before = max(float(residual_before), float(config.residual_floor))
    after = float(residual_after)
    block_cycles_i = max(int(block_cycles), 1)

    if not np.isfinite(after) or not np.isfinite(before):
        return {
            "rho": float("nan"),
            "q": float("nan"),
            "classification": "nonfinite",
        }

    rho = after / before
    q = float("nan")
    if rho > 0.0 and np.isfinite(rho):
        q = float(rho ** (1.0 / float(block_cycles_i)))

    if not np.isfinite(rho):
        classification = "nonfinite"
    elif rho > float(config.divergence_contraction_ratio):
        classification = "divergent"
    elif q >= float(config.stall_contraction_ratio):
        classification = "stalled"
    elif q > float(config.weak_contraction_ratio):
        classification = "weak"
    elif q > float(config.good_contraction_ratio):
        classification = "useful"
    else:
        classification = "strong"

    return {
        "rho": float(rho),
        "q": float(q),
        "classification": classification,
    }


def _classify_dual_inner_contraction(
    *,
    head_before: float,
    head_after: float,
    flow_before: float,
    flow_after: float,
    block_cycles: int,
    config: AdaptiveInnerSolveConfig,
    head_target: float | None = None,
    flow_target: float | None = None,
) -> dict[str, Any]:
    """Classify a block by its slower residual while retaining per-residual diagnostics."""
    floor = float(config.residual_floor)
    cycles = max(int(block_cycles), 1)
    values = (head_before, head_after, flow_before, flow_after)
    if not all(np.isfinite(float(value)) for value in values):
        return {
            "rho_head": float("nan"), "rho_flow": float("nan"),
            "q_head": float("nan"), "q_flow": float("nan"),
            "q_controller": float("nan"), "classification": "nonfinite",
        }

    rho_head = float(head_after) / max(float(head_before), floor)
    rho_flow = float(flow_after) / max(float(flow_before), floor)
    q_head = float(rho_head ** (1.0 / float(cycles))) if rho_head >= 0.0 else float("nan")
    q_flow = float(rho_flow ** (1.0 / float(cycles))) if rho_flow >= 0.0 else float("nan")
    head_active = head_target is None or float(head_after) > float(head_target)
    flow_active = flow_target is None or float(flow_after) > float(flow_target)
    active_q = []
    if head_active:
        active_q.append(q_head)
    if flow_active:
        active_q.append(q_flow)
    q_controller = max(active_q) if active_q else 0.0
    if (
        (head_active and rho_head > float(config.divergence_contraction_ratio))
        or (flow_active and rho_flow > float(config.divergence_contraction_ratio))
    ):
        classification = "divergent"
    elif q_controller >= float(config.stall_contraction_ratio):
        classification = "stalled"
    elif q_controller > float(config.weak_contraction_ratio):
        classification = "weak"
    elif q_controller > float(config.good_contraction_ratio):
        classification = "useful"
    else:
        classification = "strong"
    return {
        "rho_head": rho_head, "rho_flow": rho_flow,
        "q_head": q_head, "q_flow": q_flow,
        "q_controller": q_controller, "classification": classification,
        "head_active": head_active, "flow_active": flow_active,
    }


def _predict_next_inner_block_size(
    *,
    current_block_cycles: int,
    residual_after: float,
    target_residual: float,
    contraction_ratio: float,
    per_cycle_factor: float,
    classification: str,
    remaining_cycles: int,
    config: AdaptiveInnerSolveConfig,
    flow_residual_after: float | None = None,
    flow_target: float | None = None,
) -> int | None:
    if remaining_cycles <= 0:
        return None

    min_block = int(config.min_block_cycles)
    max_block = min(int(config.max_block_cycles), int(remaining_cycles))
    if max_block < 1:
        return None

    if not np.isfinite(float(residual_after)) or not np.isfinite(float(target_residual)):
        return min(max_block, max(1, min_block))

    residual_after_f = float(residual_after)
    target_f = max(float(target_residual), float(config.residual_floor))
    flow_after_f = float(flow_residual_after) if flow_residual_after is not None else residual_after_f
    flow_target_f = max(
        float(flow_target) if flow_target is not None else target_f,
        float(config.residual_floor),
    )
    if not np.isfinite(flow_after_f):
        return min(max_block, max(1, min_block))
    controller_gap = max(residual_after_f / target_f, flow_after_f / flow_target_f)
    if controller_gap <= 1.0:
        return min(max_block, 1 if max_block >= 1 else 0)

    q = float(per_cycle_factor)
    predicted_cycles: int | None = None
    if np.isfinite(q) and 0.0 < q < 1.0:
        numerator = np.log(1.0 / controller_gap)
        denominator = np.log(q)
        if np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0.0:
            predicted_cycles = max(1, int(np.ceil(numerator / denominator)))

    if classification == "strong":
        if predicted_cycles is None:
            predicted_cycles = min(max_block, max(current_block_cycles * 2, min_block))
        else:
            predicted_cycles = max(predicted_cycles, min(current_block_cycles * 2, max_block))
    elif classification == "useful":
        if predicted_cycles is None:
            predicted_cycles = current_block_cycles
    elif classification == "weak":
        predicted_cycles = min(max_block, max(1, min_block))
    else:
        predicted_cycles = min(max_block, max(1, min_block))

    if controller_gap <= 2.0:
        predicted_cycles = min(predicted_cycles, 2)
        predicted_cycles = max(predicted_cycles, 1)

    return int(np.clip(predicted_cycles, 1, max_block))


def _run_adaptive_inner_kcycle_blocks(
    *,
    initial_residual_rms: float,
    target_residual_rms: float,
    forcing_eta: float,
    previous_outer_residual_rms: float | None,
    previous_outer_dh_rms: float | None,
    max_cycles: int,
    config: AdaptiveInnerSolveConfig,
    run_block: Callable[[int], dict[str, Any]],
    rollback_block: Callable[[], None] | None = None,
    initial_relative_flow_residual_rms: float | None = None,
    target_relative_flow_residual_rms: float | None = None,
) -> AdaptiveInnerSolveState:
    state = AdaptiveInnerSolveState(
        previous_residual_rms=float(initial_residual_rms),
        initial_residual_rms=float(initial_residual_rms),
        target_residual_rms=float(target_residual_rms),
        previous_outer_residual_rms=(
            float(previous_outer_residual_rms) if previous_outer_residual_rms is not None else float("nan")
        ),
        previous_outer_dh_rms=(float(previous_outer_dh_rms) if previous_outer_dh_rms is not None else float("nan")),
        forcing_eta=float(forcing_eta),
        initial_relative_flow_residual_rms=(
            float(initial_relative_flow_residual_rms)
            if initial_relative_flow_residual_rms is not None else 0.0
        ),
        target_relative_flow_residual_rms=(
            float(target_relative_flow_residual_rms)
            if target_relative_flow_residual_rms is not None else float("inf")
        ),
        previous_relative_flow_residual_rms=(
            float(initial_relative_flow_residual_rms)
            if initial_relative_flow_residual_rms is not None else 0.0
        ),
    )

    initial_residual = float(initial_residual_rms)
    target_residual = float(target_residual_rms)
    initial_relative = float(state.initial_relative_flow_residual_rms)
    target_relative = float(state.target_relative_flow_residual_rms)
    if not np.isfinite(initial_residual) or not np.isfinite(initial_relative):
        state.fallback_used = True
        state.legacy_fallback_used = True
        state.fallback_reason = "nonfinite_initial_head_residual"
        state.termination_reason = "legacy_dh_fallback"
        return state

    if initial_residual <= target_residual and initial_relative <= target_relative:
        state.converged = True
        state.target_achieved = True
        state.usable_for_picard = True
        state.final_residual_rms = float(initial_residual)
        state.final_relative_flow_residual_rms = float(initial_relative)
        state.termination_reason = "initial_residual_already_below_target"
        return state

    current_block_cycles = int(np.clip(int(config.initial_block_cycles), 1, int(max_cycles)))
    while state.total_cycles < int(max_cycles):
        remaining_cycles = int(max_cycles) - state.total_cycles
        block_cycles = int(min(current_block_cycles, remaining_cycles))
        if block_cycles <= 0:
            break

        block_result = dict(run_block(int(block_cycles)))
        actual_cycles = int(block_result.get("actual_cycles", block_cycles))
        residual_after = float(block_result.get("residual_after_rms", float("nan")))
        relative_after = float(block_result.get("relative_flow_residual_rms", 0.0))
        rollback_required = bool(block_result.get("rollback_required", False))
        numerical_breakdown = bool(block_result.get("numerical_breakdown", False))
        head_nonfinite = bool(block_result.get("head_nonfinite", False))

        if actual_cycles <= 0:
            state.fallback_used = True
            state.legacy_fallback_used = True
            state.fallback_reason = "zero_cycle_block_without_initial_target"
            state.termination_reason = "legacy_dh_fallback"
            state.usable_for_picard = False
            return state

        state.total_cycles += actual_cycles
        state.block_index += 1
        state.residual_check_count += 1
        state.cycles_per_block.append(int(actual_cycles))
        state.residuals_per_block.append(float(residual_after))

        contraction = _classify_dual_inner_contraction(
            head_before=state.previous_residual_rms,
            head_after=residual_after,
            flow_before=state.previous_relative_flow_residual_rms,
            flow_after=relative_after,
            block_cycles=actual_cycles,
            config=config,
            head_target=target_residual,
            flow_target=target_relative,
        )
        rho = float(contraction["rho_head"])
        rho_flow = float(contraction["rho_flow"])
        q = float(contraction["q_controller"])
        q_head = float(contraction["q_head"])
        q_flow = float(contraction["q_flow"])
        classification = str(contraction["classification"])
        state.contraction_ratios.append(rho)
        state.per_cycle_convergence_factors.append(q)
        state.flow_contraction_ratios.append(rho_flow)
        state.head_per_cycle_convergence_factors.append(q_head)
        state.flow_per_cycle_convergence_factors.append(q_flow)
        state.controller_per_cycle_convergence_factors.append(q)
        state.block_classifications.append(classification)

        if rollback_required or numerical_breakdown or head_nonfinite or not np.isfinite(relative_after) or classification in {"divergent", "nonfinite"}:
            if rollback_block is not None:
                rollback_block()
            state.rollback_count += 1
            state.diverged = True
            state.termination_reason = "block_divergence_rolled_back"
            state.final_residual_rms = state.previous_residual_rms
            state.usable_for_picard = False
            state.predicted_cycles_per_block.append(None)
            return state

        state.final_residual_rms = residual_after
        state.final_relative_flow_residual_rms = relative_after
        state.previous_residual_rms = residual_after
        state.previous_relative_flow_residual_rms = relative_after

        if (
            residual_after <= target_residual
            and relative_after <= target_relative
            and state.total_cycles >= int(config.min_total_cycles)
        ):
            state.converged = True
            state.target_achieved = True
            state.termination_reason = "target_residual_reached"
            break

        if q >= float(config.stall_contraction_ratio):
            state.consecutive_stall_blocks += 1
        else:
            state.consecutive_stall_blocks = 0

        if state.consecutive_stall_blocks >= int(config.stall_patience):
            state.stalled = True
            state.termination_reason = "residual_stagnation"
            state.predicted_cycles_per_block.append(None)
            break

        predicted_cycles = _predict_next_inner_block_size(
            current_block_cycles=block_cycles,
            residual_after=residual_after,
            target_residual=target_residual,
            contraction_ratio=rho,
            per_cycle_factor=q,
            classification=classification,
            remaining_cycles=int(max_cycles) - state.total_cycles,
            config=config,
            flow_residual_after=relative_after,
            flow_target=target_relative,
        )
        state.predicted_cycles_per_block.append(predicted_cycles)
        if predicted_cycles is None:
            break
        current_block_cycles = int(predicted_cycles)

    if not state.termination_reason:
        state.termination_reason = (
            "max_cycles_hard_ceiling" if state.total_cycles >= int(max_cycles) else "completed_without_target"
        )

    head_reduction = float(state.final_residual_rms) / max(float(state.initial_residual_rms), float(config.residual_floor))
    flow_reduction = float(state.final_relative_flow_residual_rms) / max(
        float(state.initial_relative_flow_residual_rms), float(config.residual_floor)
    )
    state.head_reduction_ratios.append(head_reduction)
    state.flow_reduction_ratios.append(flow_reduction)
    state.usable_for_picard = bool(
        np.isfinite(float(state.final_residual_rms))
        and np.isfinite(float(state.final_relative_flow_residual_rms))
        and not state.diverged
        and head_reduction <= float(config.minimum_usable_reduction_ratio)
        and flow_reduction <= float(config.minimum_usable_reduction_ratio)
    )
    return state


def _format_unaccepted_transient_period_error(
    *,
    period_index: int,
    outer_iterations: int,
    final_max_abs_head_change: float,
    final_rms_head_change: float,
    final_head_residual_rms: float,
    final_flow_residual_rms: float,
    storage_diag_change_max: float,
    storage_diag_change_rms: float,
    storage_mode: str,
    storage_reference: str,
    coarse_operator_mode: str,
    coarse_krylov_method: str,
    total_inner_cycles: int,
    inner_controller_mode: str | None = None,
    last_inner_termination_reason: str | None = None,
    last_inner_initial_residual: float | None = None,
    last_inner_target_residual: float | None = None,
    last_inner_final_residual: float | None = None,
    last_inner_block_count: int | None = None,
    stalled_inner_solve_count: int | None = None,
    divergent_inner_solve_count: int | None = None,
) -> str:
    """
    Format a compact transient period production-acceptance failure message.
    """
    return (
        "Transient unconfined period did not achieve production acceptance: "
        f"period_index={int(period_index)} "
        f"outer_iterations={int(outer_iterations)} "
        f"final_max_abs_head_change={float(final_max_abs_head_change):.6g} "
        f"final_rms_head_change={float(final_rms_head_change):.6g} "
        f"final_head_residual_rms={float(final_head_residual_rms):.6g} "
        f"final_flow_residual_rms={float(final_flow_residual_rms):.6g} "
        f"storage_diag_change_max={float(storage_diag_change_max):.6g} "
        f"storage_diag_change_rms={float(storage_diag_change_rms):.6g} "
        f"storage_mode={storage_mode} "
        f"storage_reference={storage_reference} "
        f"coarse_operator_mode={coarse_operator_mode} "
        f"coarse_krylov_method={coarse_krylov_method} "
        f"total_inner_cycles={int(total_inner_cycles)} "
        f"inner_controller_mode={inner_controller_mode} "
        f"last_inner_termination_reason={last_inner_termination_reason} "
        f"last_inner_initial_residual={last_inner_initial_residual} "
        f"last_inner_target_residual={last_inner_target_residual} "
        f"last_inner_final_residual={last_inner_final_residual} "
        f"last_inner_block_count={last_inner_block_count} "
        f"stalled_inner_solve_count={stalled_inner_solve_count} "
        f"divergent_inner_solve_count={divergent_inner_solve_count}"
    )


@wp.kernel
def jacobi_applyA_fused_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    x_in: wp.array(dtype=WP_FLOAT, ndim=2),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    omega: float,
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    s_diag = wp.float64(storage_diag[j, i])
    sum_T = T_e + T_w + T_n + T_s + C_gh + s_diag

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
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    diagA = T_e + T_w + T_n + T_s + C_gh + wp.float64(storage_diag[j, i])

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
def compute_dual_residual_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    flow_rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    head_rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or bc_mask[j, i] != 0:
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    diagA = T_e + T_w + T_n + T_s + C_gh + wp.float64(storage_diag[j, i])
    Ax64 = wp.float64(0.0)
    flow_residual = wp.float64(0.0)
    head_residual = wp.float64(0.0)

    if diagA < tiny:
        Ax64 = hC
        flow_residual = wp.float64(b[j, i]) - Ax64
        head_residual = flow_residual
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
        flow_residual = wp.float64(b[j, i]) - Ax64
        head_residual = flow_residual / diagA

    wp.atomic_add(flow_rTr_buf, 0, flow_residual * flow_residual)
    wp.atomic_add(head_rTr_buf, 0, head_residual * head_residual)


@wp.kernel
def compute_active_rhs_l2_kernel(
    rhs: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    rhs_rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j < ny and i < nx and active[j, i] != 0 and bc_mask[j, i] == 0:
        value = wp.float64(rhs[j, i])
        wp.atomic_add(rhs_rTr_buf, 0, value * value)


@wp.kernel
def detect_nonfinite_field_kernel(
    field: wp.array(dtype=WP_FLOAT, ndim=2),
    nonfinite_flag: wp.array(dtype=wp.int32, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    value = wp.float64(field[j, i])
    delta = value - value
    if delta != delta:
        wp.atomic_max(nonfinite_flag, 0, 1)


@wp.kernel
def compute_residual_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = T_e + T_w + T_n + T_s + C_gh + wp.float64(storage_diag[j, i])

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
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    dh2_buf: wp.array(dtype=wp.float64, ndim=1),
    dh_max_buf: wp.array(dtype=wp.float64, ndim=1),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    use_ghb: int,
    nx: int,
    ny: int,
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
    if use_ghb != 0 and gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = T_e + T_w + T_n + T_s + C_gh + wp.float64(storage_diag[j, i])

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


@wp.kernel
def kcycle_check_dh_and_dual_residual_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    x_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    dh2_buf: wp.array(dtype=wp.float64, ndim=1),
    dh_max_buf: wp.array(dtype=wp.float64, ndim=1),
    flow_rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    head_rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    use_ghb: int,
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

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
    if use_ghb != 0 and gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    diagA = T_e + T_w + T_n + T_s + C_gh + wp.float64(storage_diag[j, i])
    Ax64 = wp.float64(0.0)
    flow_residual = wp.float64(0.0)
    head_residual = wp.float64(0.0)

    if diagA < tiny:
        Ax64 = hC
        flow_residual = wp.float64(b[j, i]) - Ax64
        head_residual = flow_residual
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
        flow_residual = wp.float64(b[j, i]) - Ax64
        head_residual = flow_residual / diagA

    wp.atomic_add(flow_rTr_buf, 0, flow_residual * flow_residual)
    wp.atomic_add(head_rTr_buf, 0, head_residual * head_residual)


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
        ghb_factor: np.ndarray | None = None,
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
    if (ghb_factor is None) and (aq_thickness <= 0.0):
        raise ValueError("aq_thickness must be positive.")

    # Interior RHS: b_phys = R * dx^2, then scale by head_scale
    b = (R_field * dx2 / head_scale).astype(NP_FLOAT)

    # Optional GHB contribution: b += C_gh * h_ext / head_scale
    if gh_mask is not None and gh_head is not None and ((ghb_factor is not None) or (gh_width is not None)):
        gh_mask_arr = np.asarray(gh_mask, dtype=np.int32)
        gh_head_arr = np.asarray(gh_head, dtype=NP_FLOAT)
        if ghb_factor is not None:
            ghb_factor_arr = np.asarray(ghb_factor, dtype=NP_FLOAT)
            C_gh = (T_field * ghb_factor_arr).astype(NP_FLOAT, copy=False)
            gh_ok = np.isfinite(ghb_factor_arr) & (ghb_factor_arr > NP_FLOAT(0.0))
        else:
            gh_width_arr = np.asarray(gh_width, dtype=NP_FLOAT)
            C_gh = (
                gh_alpha
                * T_field / float(aq_thickness)
                * gh_width_arr
                * dx
            ).astype(NP_FLOAT)
            gh_ok = np.isfinite(gh_width_arr) & (gh_width_arr > NP_FLOAT(0.0))

        mask_gh = (
            (gh_mask_arr != 0)
            & (active != 0)
            & gh_ok
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
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
    dx: float,
    head_scale: float,
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

    if gh_mask[j, i] != 0:
        T_c = wp.float64(T_field[j, i])
        ghbf = wp.float64(ghb_factor[j, i])
        if T_c > wp.float64(0.0) and ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf
            rhs = rhs + C_gh * (wp.float64(gh_head[j, i]) / wp.float64(head_scale))

    b_out[j, i] = WP_FLOAT(rhs)


@wp.kernel
def build_diag_preconditioner_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    use_ghb: int,
    nx: int,
    ny: int,
    M_inv_out: wp.array(dtype=WP_FLOAT, ndim=2),
):
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        M_inv_out[j, i] = WP_FLOAT(1.0)
        return

    tiny = wp.float64(1.0e-12)
    T_c = wp.float64(T_field[j, i])
    diagonal = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    diagonal = diagonal + wp.float64(storage_diag[j, i])

    if use_ghb != 0 and gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if T_c > wp.float64(0.0) and ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            diagonal = diagonal + T_c * ghbf

    if diagonal > tiny:
        M_inv_out[j, i] = WP_FLOAT(wp.float64(1.0) / diagonal)
    else:
        M_inv_out[j, i] = WP_FLOAT(1.0)


def build_diag_preconditioner(
    T_field: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    gh_mask: np.ndarray | None = None,
    gh_width: np.ndarray | None = None,
    ghb_factor: np.ndarray | None = None,
    dx: float | None = None,
    gh_alpha: float = 1.0,
    aq_thickness: float = 1.0,
    assume_finite_T: bool = False,
    storage_diag: np.ndarray | None = None,
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

    use_ghb = (gh_mask is not None) and ((ghb_factor is not None) or (gh_width is not None))
    if use_ghb and dx is None:
        raise ValueError("dx must be provided when gh_mask is provided.")

    if use_ghb:
        gh_mask_arr = np.asarray(gh_mask, dtype=np.int32)
        dx_f = float(dx)
        if ghb_factor is not None:
            ghb_factor_arr = np.asarray(ghb_factor, dtype=NP_FLOAT)
            gh_width_arr = None
            gh_alpha_f = 1.0
            thickness_f = 1.0
        else:
            ghb_factor_arr = None
            gh_width_arr = np.asarray(gh_width, dtype=NP_FLOAT)
            gh_alpha_f = float(gh_alpha)
            thickness_f = float(aq_thickness)
            if thickness_f <= 0.0:
                thickness_f = 1.0
    else:
        gh_mask_arr = None
        gh_width_arr = None
        ghb_factor_arr = None
        dx_f = 1.0
        gh_alpha_f = 1.0
        thickness_f = 1.0

    ny, nx = T_field.shape
    tiny = np.float64(1.0e-12)

    if assume_finite_T:
        T_pos = T_field > NP_FLOAT(0.0)
    else:
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
        if ghb_factor_arr is not None:
            ghb_ok = np.isfinite(ghb_factor_arr) & (ghb_factor_arr > NP_FLOAT(0.0))
            gh_on = (gh_mask_arr != 0) & ghb_ok & T_pos
            if np.any(gh_on):
                C_gh = T_field.astype(np.float64, copy=False) * ghb_factor_arr.astype(np.float64, copy=False)
                sum_T[gh_on] += C_gh[gh_on]
        else:
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


    if storage_diag is not None:
        sd = np.asarray(storage_diag, dtype=np.float64)
        sd_ok = np.isfinite(sd) & (sd > 0.0)
        on = free & sd_ok
        if np.any(on):
            sum_T[on] += sd[on]

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
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    h: wp.array(dtype=WP_FLOAT, ndim=2),
    Ah: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = ghb_factor[j, i]
        if ghbf > WP_FLOAT(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = wp.float64(T_e) + wp.float64(T_w) + wp.float64(T_n) + wp.float64(T_s) + wp.float64(C_gh) + wp.float64(storage_diag[j, i])

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
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    Ap: wp.array(dtype=WP_FLOAT, ndim=2),
    pAp_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = T_e + T_w + T_n + T_s + C_gh + wp.float64(storage_diag[j, i])

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
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = T_e + T_w + T_n + T_s + C_gh + wp.float64(storage_diag[j, i])

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


# =============================================================================
# No-storage (steady) kernel variants.
#
# These are byte-for-byte copies of the storage-aware kernels above, minus the
# ``storage_diag`` parameter and the ``+ storage_diag[j, i]`` diagonal term. They
# are used on every hot path when ``transient=False`` so steady solves never
# allocate, stage, or read a storage diagonal. For steady solves the storage
# term is identically zero, so these reproduce the storage-aware operator
# exactly when storage_diag == 0. Transient solves keep using the kernels above.
#
# The transient ``apply_A_and_pAp_kernel`` / ``init_pcg_with_A_kernel`` include
# ``storage_diag`` in the diagonal. Their no-storage twins below drop that
# argument only on steady paths where storage is identically zero.
# =============================================================================
@wp.kernel
def jacobi_applyA_fused_no_storage_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    x_in: wp.array(dtype=WP_FLOAT, ndim=2),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    omega: float,
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = T_e + T_w + T_n + T_s + C_gh

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
def compute_head_residual_no_storage_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    diagA = T_e + T_w + T_n + T_s + C_gh

    Ax64 = wp.float64(0.0)
    if diagA < tiny:
        Ax64 = hC
        rh64 = wp.float64(b[j, i]) - Ax64
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

        rf64 = wp.float64(b[j, i]) - Ax64
        rh64 = rf64 / diagA

    r[j, i] = WP_FLOAT(rh64)
    wp.atomic_add(rTr_buf, 0, rh64 * rh64)


@wp.kernel
def compute_residual_no_storage_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    r: wp.array(dtype=WP_FLOAT, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

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
def kcycle_check_dh_and_residual_no_storage_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    x_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    dh2_buf: wp.array(dtype=wp.float64, ndim=1),
    dh_max_buf: wp.array(dtype=wp.float64, ndim=1),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    use_ghb: int,
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

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
    if use_ghb != 0 and gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

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


@wp.kernel
def build_diag_preconditioner_no_storage_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    use_ghb: int,
    nx: int,
    ny: int,
    M_inv_out: wp.array(dtype=WP_FLOAT, ndim=2),
):
    j, i = wp.tid()

    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        M_inv_out[j, i] = WP_FLOAT(1.0)
        return

    tiny = wp.float64(1.0e-12)
    T_c = wp.float64(T_field[j, i])
    diagonal = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            diagonal = diagonal + wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if use_ghb != 0 and gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if T_c > wp.float64(0.0) and ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            diagonal = diagonal + T_c * ghbf

    if diagonal > tiny:
        M_inv_out[j, i] = WP_FLOAT(wp.float64(1.0) / diagonal)
    else:
        M_inv_out[j, i] = WP_FLOAT(1.0)


@wp.kernel
def apply_A_and_pAp_no_storage_kernel(
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    p: wp.array(dtype=WP_FLOAT, ndim=2),
    Ap: wp.array(dtype=WP_FLOAT, ndim=2),
    pAp_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

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
def init_pcg_with_A_no_storage_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    b: wp.array(dtype=WP_FLOAT, ndim=2),
    T_field: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=2),
    Ap: wp.array(dtype=WP_FLOAT, ndim=2),
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
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

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
def zero_int_scalar_kernel(
    buf: wp.array(dtype=wp.int32, ndim=1),
):
    """
    Zero a 1D Warp int32 array (length >= 1).
    """
    k = wp.tid()
    if k >= buf.shape[0]:
        return
    buf[k] = wp.int32(0)


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

    if rho64 > wp.float64(0.0) and pAp64 > wp.float64(1.0e-30):
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

    if rho_old > wp.float64(0.0) and rho_new > wp.float64(0.0):
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
    if wp.float64(num_buf[0]) > wp.float64(0.0) and wp.float64(den) > wp.float64(1.0e-30):
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
def apply_relaxed_clipped_picard_update_kernel(
    candidate_head: wp.array(dtype=WP_FLOAT, ndim=2),
    previous_head: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    omega: WP_FLOAT,
    max_head_change: WP_FLOAT,
    nx: int,
    ny: int,
    output_head: wp.array(dtype=WP_FLOAT, ndim=2),
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        output_head[j, i] = WP_FLOAT(0.0)
        return

    if bc_mask[j, i] != 0:
        output_head[j, i] = bc_values[j, i]
        return

    raw_update = omega * (candidate_head[j, i] - previous_head[j, i])
    if wp.isnan(raw_update):
        raw_update = WP_FLOAT(0.0)
    if raw_update > max_head_change:
        raw_update = max_head_change
    elif raw_update < -max_head_change:
        raw_update = -max_head_change
    output_head[j, i] = previous_head[j, i] + raw_update


@wp.kernel
def apply_relaxed_correction_kernel(
    previous_head: wp.array(dtype=WP_FLOAT, ndim=2),
    correction: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    omega: WP_FLOAT,
    max_head_change: WP_FLOAT,
    nx: int,
    ny: int,
    output_head: wp.array(dtype=WP_FLOAT, ndim=2),
):
    """Incremental-Picard relaxed correction.

    ``output = previous + omega * clip(correction)`` for active non-BC cells,
    ``bc_values`` for Dirichlet cells, 0 for inactive cells. Mirrors
    :func:`apply_relaxed_clipped_picard_update_kernel` but consumes the
    correction ``delta`` directly (the inner solve already produced
    ``h_lin - h^k``). With ``omega = 1`` and a very large ``max_head_change``
    this also serves as the per-block ``h_iter = h^k + delta`` sync used by
    the adaptive inner controller while it solves for ``delta``.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        output_head[j, i] = WP_FLOAT(0.0)
        return

    if bc_mask[j, i] != 0:
        output_head[j, i] = bc_values[j, i]
        return

    raw_update = omega * correction[j, i]
    if wp.isnan(raw_update):
        raw_update = WP_FLOAT(0.0)
    if raw_update > max_head_change:
        raw_update = max_head_change
    elif raw_update < -max_head_change:
        raw_update = -max_head_change
    output_head[j, i] = previous_head[j, i] + raw_update


@wp.kernel
def clamp_unconfined_head_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    bottom: wp.array(dtype=WP_FLOAT, ndim=2),
    top: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    min_sat: WP_FLOAT,
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0:
        head[j, i] = WP_FLOAT(0.0)
        return
    if bc_mask[j, i] != 0:
        head[j, i] = bc_values[j, i]
        return
    lower = bottom[j, i] + min_sat
    upper = top[j, i]
    h = head[j, i]
    if wp.isnan(h):
        h = lower
    if h < lower:
        h = lower
    if h > upper:
        h = upper
    head[j, i] = h


@wp.kernel
def fill_uniform_recharge_kernel(
    R: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    recharge_value: WP_FLOAT,
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] != 0:
        R[j, i] = recharge_value
    else:
        R[j, i] = WP_FLOAT(0.0)


@wp.kernel
def update_unconfined_transmissivity_from_head_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    k_field: wp.array(dtype=WP_FLOAT, ndim=2),
    bottom: wp.array(dtype=WP_FLOAT, ndim=2),
    top: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    min_sat: WP_FLOAT,
    nx: int,
    ny: int,
    T_out: wp.array(dtype=WP_FLOAT, ndim=2),
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0:
        T_out[j, i] = WP_FLOAT(0.0)
        return
    full = wp.max(top[j, i] - bottom[j, i], min_sat)
    sat = wp.min(wp.max(head[j, i] - bottom[j, i], min_sat), full)
    T_out[j, i] = k_field[j, i] * sat


@wp.kernel
def update_secant_sy_storage_kernel(
    head_ref: wp.array(dtype=WP_FLOAT, ndim=2),
    head_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    bottom: wp.array(dtype=WP_FLOAT, ndim=2),
    top: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    sy: WP_FLOAT,
    ss: WP_FLOAT,
    dx: WP_FLOAT,
    dt: WP_FLOAT,
    min_sat: WP_FLOAT,
    eps: WP_FLOAT,
    nx: int,
    ny: int,
    storage_coeff_out: wp.array(dtype=WP_FLOAT, ndim=2),
    sy_coeff_out: wp.array(dtype=WP_FLOAT, ndim=2),
    ss_coeff_out: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag_out: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_change_sum_sq: wp.array(dtype=wp.float64, ndim=1),
    storage_change_max: wp.array(dtype=wp.float64, ndim=1),
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or bc_mask[j, i] != 0:
        storage_coeff_out[j, i] = WP_FLOAT(0.0)
        sy_coeff_out[j, i] = WP_FLOAT(0.0)
        ss_coeff_out[j, i] = WP_FLOAT(0.0)
        storage_diag_out[j, i] = WP_FLOAT(0.0)
        return

    full = wp.max(top[j, i] - bottom[j, i], WP_FLOAT(0.0))
    sat_old = wp.min(wp.max(head_prev[j, i] - bottom[j, i], WP_FLOAT(0.0)), full)
    sat_ref_zero = wp.min(wp.max(head_ref[j, i] - bottom[j, i], WP_FLOAT(0.0)), full)
    dh = head_ref[j, i] - head_prev[j, i]

    sy_coeff = WP_FLOAT(0.0)
    if wp.abs(dh) > eps:
        sy_coeff = sy * ((sat_ref_zero - sat_old) / dh)
    elif head_ref[j, i] > bottom[j, i] and head_ref[j, i] < top[j, i]:
        sy_coeff = sy

    sy_coeff = wp.min(wp.max(sy_coeff, WP_FLOAT(0.0)), sy)

    rel_old = head_prev[j, i] - bottom[j, i]
    rel_ref = head_ref[j, i] - bottom[j, i]
    phi_old = WP_FLOAT(0.0)
    phi_ref = WP_FLOAT(0.0)
    if rel_old > WP_FLOAT(0.0) and rel_old < full:
        phi_old = WP_FLOAT(0.5) * ss * rel_old * rel_old
    elif rel_old >= full:
        phi_old = WP_FLOAT(0.5) * ss * full * full + ss * full * (rel_old - full)
    if rel_ref > WP_FLOAT(0.0) and rel_ref < full:
        phi_ref = WP_FLOAT(0.5) * ss * rel_ref * rel_ref
    elif rel_ref >= full:
        phi_ref = WP_FLOAT(0.5) * ss * full * full + ss * full * (rel_ref - full)

    ss_coeff = WP_FLOAT(0.0)
    if wp.abs(dh) > eps:
        ss_coeff = (phi_ref - phi_old) / dh
    else:
        sat_ref_ss = wp.min(wp.max(rel_ref, WP_FLOAT(0.0)), full)
        ss_coeff = ss * sat_ref_ss
    ss_coeff = wp.max(ss_coeff, WP_FLOAT(0.0))
    storage_coeff = sy_coeff + ss_coeff
    storage_diag = storage_coeff * dx * dx / dt
    delta = wp.float64(storage_diag - storage_diag_prev[j, i])

    sy_coeff_out[j, i] = sy_coeff
    ss_coeff_out[j, i] = ss_coeff
    storage_coeff_out[j, i] = storage_coeff
    storage_diag_out[j, i] = storage_diag
    wp.atomic_add(storage_change_sum_sq, 0, delta * delta)
    wp.atomic_max(storage_change_max, 0, wp.abs(delta))


@wp.kernel
def build_transient_rhs_from_storage_kernel(
    recharge_rate: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=2),
    head_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=2),
    dx: WP_FLOAT,
    nx: int,
    ny: int,
    rhs_out: wp.array(dtype=WP_FLOAT, ndim=2),
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0:
        rhs_out[j, i] = WP_FLOAT(0.0)
    elif bc_mask[j, i] != 0:
        rhs_out[j, i] = bc_values[j, i]
    else:
        rhs_out[j, i] = recharge_rate[j, i] * dx * dx + storage_diag[j, i] * head_prev[j, i]


@wp.kernel
def coarsen_transient_operator_level_kernel(
    T_f: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag_f: wp.array(dtype=WP_FLOAT, ndim=2),
    active_f: wp.array(dtype=wp.int32, ndim=2),
    active_c: wp.array(dtype=wp.int32, ndim=2),
    bc_mask_c: wp.array(dtype=wp.int32, ndim=2),
    nx_f: int,
    ny_f: int,
    nx_c: int,
    ny_c: int,
    T_c_out: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag_c_out: wp.array(dtype=WP_FLOAT, ndim=2),
):
    j, i = wp.tid()
    if j >= ny_c or i >= nx_c:
        return

    if active_c[j, i] == 0:
        T_c_out[j, i] = WP_FLOAT(0.0)
        storage_diag_c_out[j, i] = WP_FLOAT(0.0)
        return

    inv_sum = wp.float64(0.0)
    count = wp.int32(0)
    storage_sum = wp.float64(0.0)

    for dj in range(2):
        fj = j * 2 + dj
        if fj >= ny_f:
            continue
        for di in range(2):
            fi = i * 2 + di
            if fi >= nx_f:
                continue
            storage_sum = storage_sum + wp.float64(storage_diag_f[fj, fi])
            if active_f[fj, fi] != 0:
                t_val = wp.float64(T_f[fj, fi])
                if t_val > wp.float64(0.0) and not wp.isnan(t_val):
                    inv_sum = inv_sum + wp.float64(1.0) / t_val
                    count = count + wp.int32(1)

    if count > wp.int32(0) and inv_sum > wp.float64(0.0):
        T_c_out[j, i] = WP_FLOAT(wp.float64(count) / inv_sum)
    else:
        T_c_out[j, i] = WP_FLOAT(0.0)

    if bc_mask_c[j, i] != 0:
        storage_diag_c_out[j, i] = WP_FLOAT(0.0)
    else:
        storage_diag_c_out[j, i] = WP_FLOAT(storage_sum)


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
    ghb_factor_host: Optional[np.ndarray]
    storage_diag_host: np.ndarray | None

    # Device fields (always 2D)
    T_wp: wp.array
    R_wp: wp.array
    active_wp: wp.array
    bc_mask_wp: wp.array
    bc_values_wp: wp.array
    gh_mask_wp: Optional[wp.array]
    gh_head_wp: Optional[wp.array]
    gh_width_wp: Optional[wp.array]
    ghb_factor_wp: Optional[wp.array]
    storage_diag_wp: wp.array | None

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
            "ghb_factor_wp",
            "storage_diag_wp",
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
            ghb_factor_wp,
            storage_diag_wp=None,
            M_inv_wp=None,
            nx: int = 0,
            ny: int = 0,
            dx: float = 0.0,
        ):
            self.T_wp = T_wp
            self.active_wp = active_wp
            self.bc_mask_wp = bc_mask_wp
            self.gh_mask_wp = gh_mask_wp
            self.gh_width_wp = gh_width_wp
            self.ghb_factor_wp = ghb_factor_wp
            self.storage_diag_wp = storage_diag_wp
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
            "ghb_factor_host",
            "storage_diag_host",
            # device fields
            "T_wp",
            "R_wp",
            "active_wp",
            "bc_mask_wp",
            "bc_values_wp",
            "gh_mask_wp",
            "gh_head_wp",
            "gh_width_wp",
            "ghb_factor_wp",
            "storage_diag_wp",
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
            ghb_factor_host,
            storage_diag_host,
            T_wp,
            R_wp,
            active_wp,
            bc_mask_wp,
            bc_values_wp,
            gh_mask_wp,
            gh_head_wp,
            gh_width_wp,
            ghb_factor_wp,
            storage_diag_wp,
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
            self.ghb_factor_host = ghb_factor_host
            self.storage_diag_host = storage_diag_host

            self.T_wp = T_wp
            self.R_wp = R_wp
            self.active_wp = active_wp
            self.bc_mask_wp = bc_mask_wp
            self.bc_values_wp = bc_values_wp
            self.gh_mask_wp = gh_mask_wp
            self.gh_head_wp = gh_head_wp
            self.gh_width_wp = gh_width_wp
            self.ghb_factor_wp = ghb_factor_wp
            self.storage_diag_wp = storage_diag_wp
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
        aq_thickness: float | np.ndarray = 1.0,
        trust_ghb_params_for_graph: bool = False,
        diag_preconditioner_backend: str = "auto",
    ):
        """
        :param nx: number of columns
        :param ny: number of rows
        :param dx: cell size
        :param device: Warp device string, for example "cuda:0"
        :param use_ghb: if True, include GHB terms in operator and RHS assembly
        :param solver_type: "pcg" or "jacobi" (future)
        :param head_scale: characteristic head scale, h_scaled = h / head_scale
        :param aq_thickness: aquifer thickness (scalar or grid) used in GHB conductance scaling
        :param trust_ghb_params_for_graph: if True, do not rebuild CUDA graph when gh_alpha or aq_thickness change.
        """
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.device_str = str(device)
        self.use_ghb = bool(use_ghb)
        self.solver_type = str(solver_type)
        self.trust_ghb_params_for_graph = bool(trust_ghb_params_for_graph)
        backend_mode = str(diag_preconditioner_backend).strip().lower()
        if backend_mode not in {"auto", "host", "device"}:
            raise ValueError("diag_preconditioner_backend must be 'auto', 'host', or 'device'.")
        self.diag_preconditioner_backend = backend_mode

        if head_scale != 1.0:
            raise ValueError("head_scale has been removed. Use physical heads everywhere and set head_scale=1.0.")
        self.head_scale = 1.0

        if np.asarray(aq_thickness).ndim == 0 and float(aq_thickness) <= 0.0:
            raise ValueError("aq_thickness must be positive.")
        self._aq_thickness_input = aq_thickness
        self.aq_thickness = float(aq_thickness) if np.asarray(aq_thickness).ndim == 0 else 1.0
        self.aq_thickness_host = None

        # Host side storage for fields
        self.T_field_host = None
        self.R_field_host = None
        self.active_host = None
        self.bc_mask_host = None
        self.bc_values_host = None
        self.gh_mask_host = None
        self.gh_head_host = None
        self.gh_width_host = None
        self._gh_alpha_input = 1.0
        self.gh_alpha = 1.0
        self.gh_alpha_host = None
        self.ghb_factor_host = None
        self.storage_diag_host = None

        # Device side Warp arrays (set in build_from_truth_inputs)
        self.T_wp = None
        self.R_wp = None
        self.active_wp = None
        self.bc_mask_wp = None
        self.bc_values_wp = None
        self.gh_mask_wp = None
        self.gh_head_wp = None
        self.gh_width_wp = None
        self.ghb_factor_wp = None
        self.storage_diag_wp = None

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
        self.ghb_factor_c_host = None
        self.storage_diag_c_host = None

        # Coarse device arrays
        self.T_c_wp = None
        self.R_c_wp = None
        self.active_c_wp = None
        self.bc_mask_c_wp = None
        self.bc_values_c_wp = None
        self.gh_mask_c_wp = None
        self.gh_head_c_wp = None
        self.gh_width_c_wp = None
        self.ghb_factor_c_wp = None
        self.storage_diag_c_wp = None
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
        self._stage_G0_host = None
        self._stage_G0 = None
        self._stage_Gc_2lvl = None
        self._stage_Sc_2lvl = None

        self._stage_T_levels = None
        self._stage_M_levels = None
        self._stage_G_levels = None
        self._stage_S_levels = None

        self.T_field_host = None
        self._T_field_wp_host = None
        self.T_field_dev = None

        self._operator_dirty = True

        # True only while a transient storage diagonal is active. When False
        # (steady solves) the storage diagonal is identically zero, so every
        # multigrid level shares a single fine-size zero storage array instead
        # of allocating one per level. See build_hierarchy().
        self._storage_active = False

        # Hierarchy storage (for K cycle later)
        self.mg_levels = None
        self._mg_coarsening_diagnostics = []
        self._two_level_coarsening_diag = None
        self._last_device_m_inv_validation = []
        self._last_update_T_profile = None
        self._update_T_profile_totals = None
        self._transient_replay_counters = {}
        # ---------------- CUDA graph cache (K-cycle path) ----------------
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

    def _active_storage_diag_host(self) -> np.ndarray | None:
        if not bool(self._storage_active):
            return None
        return self.storage_diag_host

    def _active_storage_diag_wp(self):
        if not bool(self._storage_active):
            return None
        return getattr(self, "storage_diag_wp", None)

    def _clear_transient_storage_state(self) -> bool:
        """
        Drop transient storage contributions so the next steady solve sees a
        pure steady operator without host/device storage staging.

        Returns True only when real (transient) storage state was present and
        got cleared. A steady solve carries no storage arrays at all (they are
        ``None`` and the no-storage kernels never reference them), so a steady
        solve after a steady solve returns False and triggers no rebuild. A
        steady solve after a transient solve still sees the real transient
        arrays and clears/deactivates them here.
        """
        had_storage = (
            bool(self._storage_active)
            or (self.storage_diag_host is not None)
            or (getattr(self, "storage_diag_wp", None) is not None)
            or (getattr(self, "storage_diag_c_host", None) is not None)
            or (getattr(self, "storage_diag_c_wp", None) is not None)
        )

        self._storage_active = False
        self.storage_diag_host = None
        self.storage_diag_wp = None
        self.storage_diag_c_host = None
        self.storage_diag_c_wp = None

        return had_storage

    def _invalidate_kcycle_graph(self) -> None:
        self._kcycle_graph = None
        self._kcycle_graph_shape = None

    def _diag_backend_env_or_default(self) -> str:
        if self.diag_preconditioner_backend != "auto":
            return self.diag_preconditioner_backend
        mode = str(os.environ.get("DARCY_M_INV_BACKEND", "auto")).strip().lower()
        if mode not in {"auto", "host", "device"}:
            mode = "auto"
        return mode

    def _select_diag_preconditioner_backend(
        self,
        *,
        T_wp,
        active_wp,
        bc_mask_wp,
        gh_mask_wp,
        ghb_factor_wp,
    ) -> str:
        mode = self._diag_backend_env_or_default()
        if mode == "host":
            return "host"

        device_ready = (
            T_wp is not None
            and active_wp is not None
            and bc_mask_wp is not None
            and gh_mask_wp is not None
            and ghb_factor_wp is not None
        )
        if mode == "device":
            return "device" if device_ready else "host"

        if (not str(self.device_str).startswith("cuda")) or (not device_ready):
            return "host"
        return "device"

    def _profile_update_T_enabled(self) -> bool:
        raw = str(os.environ.get("DARCY_PROFILE_UPDATE_T", "")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _validate_device_m_inv_enabled(self) -> bool:
        raw = str(os.environ.get("DARCY_VALIDATE_DEVICE_M_INV", "")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _coarsening_diag_entry(
        self,
        *,
        level_id: int,
        active_f: np.ndarray,
        active_c: np.ndarray,
        bc_mask_c: np.ndarray,
    ) -> dict[str, float | int]:
        n_active_fine = int(np.count_nonzero(active_f))
        n_active_coarse = int(np.count_nonzero(active_c))
        n_bc_coarse = int(np.count_nonzero(bc_mask_c))
        n_bc_off_active = int(np.count_nonzero((np.asarray(bc_mask_c, dtype=np.int32) != 0) & (np.asarray(active_c, dtype=np.int32) == 0)))

        if n_active_coarse <= 0:
            coarsening_ratio = float("inf") if n_active_fine > 0 else 1.0
        else:
            coarsening_ratio = float(n_active_fine) / float(n_active_coarse * 4)

        diag = {
            "level_id": int(level_id),
            "n_active_fine": int(n_active_fine),
            "n_active_coarse": int(n_active_coarse),
            "n_bc_coarse": int(n_bc_coarse),
            "n_bc_off_active": int(n_bc_off_active),
            "coarsening_ratio": float(coarsening_ratio),
        }

        ratio_ok = np.isfinite(coarsening_ratio) and abs(coarsening_ratio - 1.0) <= 0.3
        if not ratio_ok:
            warnings.warn(
                (
                    f"MG level {int(level_id)} coarsening ratio {coarsening_ratio:.2f} "
                    "is outside [0.7, 1.3]; coarse-grid geometry may be inconsistent."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        if n_bc_off_active != 0:
            warnings.warn(
                (
                    f"MG level {int(level_id)} has {n_bc_off_active} coarse boundary cells "
                    "outside the active mask after coarsening."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        return diag

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

        # Release the previous hierarchy's device arrays before rebuilding.
        # Old _MGLevel objects survive reassignment of self.mg_levels because
        # their wp.arrays participate in reference cycles (array -> device
        # context) that CPython refcounting cannot break on its own. Without
        # this, every K-cycle rebuild (e.g. each unconfined Picard iteration)
        # accumulates a full hierarchy in the mempool until close() runs gc,
        # which OOMs large grids. Nulling the slots drops the array refs;
        # gc.collect() then reclaims the cyclic ones so the pool reuses the
        # memory for the new build. This mirrors what close() does at teardown.
        if self.mg_levels is not None:
            try:
                wp.synchronize_device(device)
            except Exception:
                pass
            for _prev_lvl in self.mg_levels:
                try:
                    for _name in _prev_lvl.__slots__:
                        setattr(_prev_lvl, _name, None)
                except Exception:
                    pass
            self.mg_levels = None
            gc.collect()

        levels = []
        self._mg_coarsening_diagnostics = []

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
                ghb_factor_c,
                storage_diag_c,
            ) = self._mg_coarsen_host_any(
                T_f=fine.T_host,
                R_f=fine.R_host,
                active_f=fine.active_host,
                bc_mask_f=fine.bc_mask_host,
                bc_values_f=fine.bc_values_host,
                gh_mask_f=fine.gh_mask_host if self.use_ghb else None,
                gh_head_f=fine.gh_head_host if self.use_ghb else None,
                gh_width_f=fine.gh_width_host if self.use_ghb else None,
                ghb_factor_f=fine.ghb_factor_host if self.use_ghb else None,
                dx_c=float(dx_c),
                storage_diag_f=getattr(fine, 'storage_diag_host', None),
            )

            # Homogeneous BCs on coarse levels for error equation (correction scheme).
            bc_values_c.fill(0.0)
            if gh_head_c is not None:
                gh_head_c.fill(0.0)

            n_active_c = int(np.count_nonzero(active_c))
            self._mg_coarsening_diagnostics.append(
                self._coarsening_diag_entry(
                    level_id=int(level_id),
                    active_f=fine.active_host,
                    active_c=active_c,
                    bc_mask_c=bc_mask_c,
                )
            )

            # Upload coarse fields
            T_c_wp = wp.array(T_c, dtype=WP_FLOAT, device=device)
            R_c_wp = wp.array(R_c, dtype=WP_FLOAT, device=device)
            active_c_wp = wp.array(active_c, dtype=wp.int32, device=device)
            bc_mask_c_wp = wp.array(bc_mask_c, dtype=wp.int32, device=device)
            bc_values_c_wp = wp.array(bc_values_c, dtype=WP_FLOAT, device=device)
            if self._storage_active:
                storage_diag_c_wp = wp.array(storage_diag_c, dtype=WP_FLOAT, device=device)
            else:
                # Steady solve: no transient storage diagonal exists, so coarse
                # levels carry no storage array at all. The no-storage hot
                # kernels do not reference storage_diag, so None is safe and
                # avoids a per-level device allocation of zeros.
                storage_diag_c_wp = None

            if self.use_ghb and gh_mask_c is not None:
                gh_mask_c_wp = wp.array(gh_mask_c, dtype=wp.int32, device=device)
                gh_head_c_wp = wp.array(gh_head_c, dtype=WP_FLOAT, device=device)
                gh_width_c_wp = wp.array(gh_width_c, dtype=WP_FLOAT, device=device)
                ghb_factor_c_wp = wp.array(ghb_factor_c, dtype=WP_FLOAT, device=device)
            else:
                gh_mask_c_wp, gh_head_c_wp, gh_width_c_wp, ghb_factor_c_wp = self._zero_ghb_device_arrays(
                    (int(ny_c), int(nx_c)),
                    device,
                )

            diag_backend = self._select_diag_preconditioner_backend(
                T_wp=T_c_wp,
                active_wp=active_c_wp,
                bc_mask_wp=bc_mask_c_wp,
                gh_mask_wp=gh_mask_c_wp,
                ghb_factor_wp=ghb_factor_c_wp,
            )
            if diag_backend == "device":
                M_inv_c_wp = wp.empty((int(ny_c), int(nx_c)), dtype=WP_FLOAT, device=device)
                self._update_diag_preconditioner_device(
                    T_wp=T_c_wp,
                    active_wp=active_c_wp,
                    bc_mask_wp=bc_mask_c_wp,
                    gh_mask_wp=gh_mask_c_wp,
                    ghb_factor_wp=ghb_factor_c_wp,
                    M_inv_wp=M_inv_c_wp,
                    nx=int(nx_c),
                    ny=int(ny_c),
                    use_ghb=bool(self.use_ghb),
                    storage_diag_wp=storage_diag_c_wp,
                )
                self._validate_device_diag_preconditioner(
                    level_name=f"hierarchy_level_{int(level_id)}",
                    T_field=T_c,
                    active=active_c,
                    bc_mask=bc_mask_c,
                    gh_mask=gh_mask_c if self.use_ghb else None,
                    ghb_factor=ghb_factor_c if self.use_ghb else None,
                    dx=float(dx_c) if self.use_ghb else None,
                    M_inv_wp=M_inv_c_wp,
                    storage_diag=storage_diag_c,
                )
            else:
                M_inv_c_host = build_diag_preconditioner(
                    T_field=T_c,
                    active=active_c,
                    bc_mask=bc_mask_c,
                    gh_mask=gh_mask_c if self.use_ghb else None,
                    ghb_factor=ghb_factor_c if self.use_ghb else None,
                    dx=float(dx_c) if self.use_ghb else None,
                    storage_diag=storage_diag_c,
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
                ghb_factor_host=ghb_factor_c if self.use_ghb else None,
                storage_diag_host=storage_diag_c,
                T_wp=T_c_wp,
                R_wp=R_c_wp,
                active_wp=active_c_wp,
                bc_mask_wp=bc_mask_c_wp,
                bc_values_wp=bc_values_c_wp,
                gh_mask_wp=gh_mask_c_wp,
                gh_head_wp=gh_head_c_wp,
                gh_width_wp=gh_width_c_wp,
                ghb_factor_wp=ghb_factor_c_wp,
                storage_diag_wp=storage_diag_c_wp,
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
        self._operator_dirty = False

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
            ghb_factor0 = np.asarray(self.ghb_factor_host, dtype=np.float32)
        else:
            gh_mask0 = None
            gh_head0 = None
            gh_width0 = None
            ghb_factor0 = None

        n_active0 = int(np.count_nonzero(active0))

        T0_wp = self.T_wp
        R0_wp = self.R_wp
        active0_wp = self.active_wp
        bc_mask0_wp = self.bc_mask_wp
        bc_values0_wp = self.bc_values_wp

        if self.use_ghb:
            gh_mask0_wp = self.gh_mask_wp
            gh_head0_wp = self.gh_head_wp
            gh_width0_wp = self.gh_width_wp
            ghb_factor0_wp = self.ghb_factor_wp
        else:
            gh_mask0_wp, gh_head0_wp, gh_width0_wp, ghb_factor0_wp = self._zero_ghb_device_arrays(
                (int(ny), int(nx)),
                device,
            )

        storage_diag0_host = getattr(self, 'storage_diag_host', None)
        storage_diag0_wp = getattr(self, 'storage_diag_wp', None)
        # Steady no-storage path: leave storage arrays as None. The no-storage
        # hot kernels never reference storage_diag, so level 0 does not need a
        # placeholder zero array. Only transient solves carry a real diagonal.

        if self.M_inv_wp is None:
            diag_backend = self._select_diag_preconditioner_backend(
                T_wp=T0_wp,
                active_wp=active0_wp,
                bc_mask_wp=bc_mask0_wp,
                gh_mask_wp=gh_mask0_wp,
                ghb_factor_wp=ghb_factor0_wp,
            )
            if diag_backend == "device":
                M_inv0_wp = wp.empty((int(ny), int(nx)), dtype=WP_FLOAT, device=device)
                self._update_diag_preconditioner_device(
                    T_wp=T0_wp,
                    active_wp=active0_wp,
                    bc_mask_wp=bc_mask0_wp,
                    gh_mask_wp=gh_mask0_wp,
                    ghb_factor_wp=ghb_factor0_wp,
                    M_inv_wp=M_inv0_wp,
                    nx=int(nx),
                    ny=int(ny),
                    use_ghb=bool(self.use_ghb),
                    storage_diag_wp=storage_diag0_wp,
                )
                self._validate_device_diag_preconditioner(
                    level_name="hierarchy_level_0",
                    T_field=T0,
                    active=active0,
                    bc_mask=bc_mask0,
                    gh_mask=gh_mask0 if self.use_ghb else None,
                    ghb_factor=ghb_factor0 if self.use_ghb else None,
                    dx=float(self.dx) if self.use_ghb else None,
                    M_inv_wp=M_inv0_wp,
                    storage_diag=storage_diag0_host,
                )
            else:
                M_inv0_host = build_diag_preconditioner(
                    T_field=T0,
                    active=active0,
                    bc_mask=bc_mask0,
                    gh_mask=gh_mask0 if self.use_ghb else None,
                    ghb_factor=ghb_factor0 if self.use_ghb else None,
                    dx=float(self.dx) if self.use_ghb else None,
                    storage_diag=getattr(self, 'storage_diag_host', None),
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
            ghb_factor_host=ghb_factor0,
            storage_diag_host=storage_diag0_host,
            T_wp=T0_wp,
            R_wp=R0_wp,
            active_wp=active0_wp,
            bc_mask_wp=bc_mask0_wp,
            bc_values_wp=bc_values0_wp,
            gh_mask_wp=gh_mask0_wp,
            gh_head_wp=gh_head0_wp,
            gh_width_wp=gh_width0_wp,
            ghb_factor_wp=ghb_factor0_wp,
            storage_diag_wp=storage_diag0_wp,
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
        ghb_factor_f,
        dx_c: float,
        storage_diag_f=None,
    ):
        """
        Odd-safe 2:1 coarsening via padding to even and block operations.
        Uses conservative mask coarsening and harmonic transmissivity aggregation
        so coarse levels stay closer to fine-grid conductance.

        Returns host arrays for the coarse level.
        """
        (
            T_c,
            R_c,
            active_c,
            bc_mask_c,
            bc_values_c,
            gh_mask_c,
            gh_head_c,
            gh_width_c,
            ghb_factor_c,
            storage_diag_c,
        ) = _coarsen_level_host_2x2(
            T_f=T_f,
            R_f=R_f,
            active_f=active_f,
            bc_mask_f=bc_mask_f,
            bc_values_f=bc_values_f,
            gh_mask_f=gh_mask_f,
            gh_head_f=gh_head_f,
            gh_width_f=gh_width_f,
            ghb_factor_f=ghb_factor_f,
            storage_diag_f=storage_diag_f,
        )

        if self.use_ghb:
            gh_mask_c, gh_width_c, gh_head_c = self._mg_sanitize_ghb_level_host(
                active=active_c,
                bc_mask=bc_mask_c,
                gh_mask=gh_mask_c,
                gh_width=gh_width_c,
                gh_head=gh_head_c,
                dx=float(dx_c),
            )
            if ghb_factor_c is not None:
                ghb_factor_c = np.asarray(ghb_factor_c, dtype=NP_FLOAT)
                ghb_factor_c[np.asarray(gh_mask_c, dtype=np.int32) == 0] = NP_FLOAT(0.0)
        else:
            gh_mask_c = None
            gh_head_c = None
            gh_width_c = None
            ghb_factor_c = None

        return T_c, R_c, active_c, bc_mask_c, bc_values_c, gh_mask_c, gh_head_c, gh_width_c, ghb_factor_c, storage_diag_c

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

    def _summarize_grid_to_scalar_for_reporting(self, grid: np.ndarray, mask: np.ndarray) -> float:
        vals = np.asarray(grid, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        if vals.size == 0:
            vals = np.asarray(grid, dtype=np.float64).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 1.0
        return float(np.median(vals))

    def _recompute_ghb_factor_host(
        self,
        *,
        gh_alpha: float | np.ndarray | None = None,
        aq_thickness: float | np.ndarray | None = None,
    ) -> None:
        """
        Refresh ghb_factor_host from raw gh_width and scalar-or-grid gh_alpha/aq_thickness.
        """
        if gh_alpha is not None:
            self._gh_alpha_input = gh_alpha
        if aq_thickness is not None:
            self._aq_thickness_input = aq_thickness

        shape = (int(self.ny), int(self.nx))
        gh_alpha_grid, _ = _normalize_scalar_or_grid_to_shape(
            self._gh_alpha_input,
            shape=shape,
            name="gh_alpha",
        )
        aq_thickness_grid, _ = _normalize_scalar_or_grid_to_shape(
            self._aq_thickness_input,
            shape=shape,
            name="aq_thickness",
        )
        self.gh_alpha_host = gh_alpha_grid.astype(NP_FLOAT, copy=False)
        self.aq_thickness_host = aq_thickness_grid.astype(NP_FLOAT, copy=False)

        if (
            (not self.use_ghb)
            or self.gh_mask_host is None
            or self.gh_width_host is None
            or self.active_host is None
            or self.bc_mask_host is None
        ):
            self.ghb_factor_host = np.zeros(shape, dtype=NP_FLOAT)
            self.gh_alpha = float(
                self._summarize_grid_to_scalar_for_reporting(self.gh_alpha_host, np.ones(shape, dtype=bool))
            )
            self.aq_thickness = float(
                self._summarize_grid_to_scalar_for_reporting(self.aq_thickness_host, np.ones(shape, dtype=bool))
            )
            return

        ghb_factor, gh_alpha_full, aq_thickness_full, _ = _compute_ghb_factor_from_raw_fields(
            gh_mask=self.gh_mask_host,
            gh_width=self.gh_width_host,
            gh_alpha=self.gh_alpha_host,
            aq_thickness=self.aq_thickness_host,
            dx=float(self.dx),
            active=self.active_host,
            bc_mask=self.bc_mask_host,
        )
        self.ghb_factor_host = np.asarray(ghb_factor, dtype=NP_FLOAT)
        self.gh_alpha_host = np.asarray(gh_alpha_full, dtype=NP_FLOAT)
        self.aq_thickness_host = np.asarray(aq_thickness_full, dtype=NP_FLOAT)

        gh_on = (
            (np.asarray(self.gh_mask_host, dtype=np.int32) != 0)
            & (np.asarray(self.active_host, dtype=np.int32) != 0)
            & (np.asarray(self.bc_mask_host, dtype=np.int32) == 0)
            & np.isfinite(np.asarray(self.gh_width_host, dtype=np.float64))
            & (np.asarray(self.gh_width_host, dtype=np.float64) > 0.0)
        )
        self.gh_alpha = float(self._summarize_grid_to_scalar_for_reporting(self.gh_alpha_host, gh_on))
        self.aq_thickness = float(self._summarize_grid_to_scalar_for_reporting(self.aq_thickness_host, gh_on))

    def _upload_ghb_factor_to_device(self, device: str) -> None:
        """
        Upload current ghb_factor_host to ghb_factor_wp in-place.
        """
        if self.ghb_factor_host is None:
            self.ghb_factor_host = np.zeros((int(self.ny), int(self.nx)), dtype=NP_FLOAT)

        shape = (int(self.ny), int(self.nx))
        if self.ghb_factor_wp is None or tuple(self.ghb_factor_wp.shape) != shape:
            self.ghb_factor_wp = wp.array(self.ghb_factor_host, dtype=WP_FLOAT, device=device)
            self._stage_G0_host = self.ghb_factor_host
            self._stage_G0 = wp.array(self._stage_G0_host, dtype=WP_FLOAT, device="cpu")
            return

        if self._stage_G0 is None or tuple(self._stage_G0.shape) != shape:
            self._stage_G0 = wp.zeros(shape, dtype=WP_FLOAT, device="cpu")
        self._stage_G0.numpy()[:, :] = np.asarray(self.ghb_factor_host, dtype=NP_FLOAT)
        wp.copy(self.ghb_factor_wp, self._stage_G0)


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

        if self.use_ghb and self.gh_mask_host is not None and self.ghb_factor_host is not None:
            gh_mask = np.asarray(self.gh_mask_host, dtype=np.int32)
            ghb_factor = np.asarray(self.ghb_factor_host, dtype=np.float64)
            ghb_ok = np.isfinite(ghb_factor) & (ghb_factor > 0.0)
            gh_on = (gh_mask != 0) & ghb_ok & T_pos & act
            if np.any(gh_on):
                C_gh = T * ghb_factor
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
        if self.ghb_factor_host is not None:
            ghb_factor = np.asarray(self.ghb_factor_host, dtype=NP_FLOAT).copy()
            ghb_factor[isolated] = NP_FLOAT(0.0)
            self.ghb_factor_host = ghb_factor

        self._n_isolated_pruned = int(np.count_nonzero(isolated))

    # -------------------------------------------------------------------------
    # Build and upload
    # -------------------------------------------------------------------------
    def build_from_truth_inputs(
        self,
        T_truth,
        R_truth,
        gh_alpha: float | np.ndarray = 1.0,
        aq_thickness: float | np.ndarray | None = None,
        width: float = None,
    ):
        """
        Build FD style inpts from T_truth and R_truth and upload to device.

        :param T_truth: scalar or array transmissivity
        :param R_truth: scalar or array recharge
        :param gh_alpha: GHB scaling factor (scalar or grid)
        :param aq_thickness: aquifer thickness (scalar or grid). Uses solver default when None.
        """
        if width is None:
            width = float(self.dx)
        self._gh_alpha_input = gh_alpha
        if aq_thickness is not None:
            self._aq_thickness_input = aq_thickness

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
        self.storage_diag_host = None
        self.storage_diag_wp = None
        self.storage_diag_c_host = None
        self.storage_diag_c_wp = None

        self._sanitize_ghb_host_fields()
        self._recompute_ghb_factor_host(gh_alpha=self._gh_alpha_input, aq_thickness=aq_thickness)
        self._prune_isolated_active_host_cells()
        self._recompute_ghb_factor_host()
        self.n_active = int(np.count_nonzero(self.active_host))

        device = self.device_str

        self.T_wp = wp.array(self.T_field_host, dtype=WP_FLOAT, device=device)
        self.R_wp = wp.array(self.R_field_host, dtype=WP_FLOAT, device=device)
        self.active_wp = wp.array(self.active_host, dtype=wp.int32, device=device)
        self.bc_mask_wp = wp.array(self.bc_mask_host, dtype=wp.int32, device=device)
        self.bc_values_wp = wp.array(self.bc_values_host, dtype=WP_FLOAT, device=device)

        # Always allocate GHB arrays and pass them to kernels (mask is zero if unused)
        self.gh_mask_wp = wp.array(self.gh_mask_host, dtype=wp.int32, device=device)
        self.gh_head_wp = wp.array(self.gh_head_host, dtype=WP_FLOAT, device=device)
        self.gh_width_wp = wp.array(self.gh_width_host, dtype=WP_FLOAT, device=device)
        self._upload_ghb_factor_to_device(device=device)

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
            ghb_factor=self.ghb_factor_host if self.use_ghb else None,
            dx=float(self.dx) if self.use_ghb else None,
            storage_diag=self.storage_diag_host,
        )
        self.M_inv_wp = wp.empty((int(self.ny), int(self.nx)), dtype=WP_FLOAT, device=device)
        fine_backend = self._select_diag_preconditioner_backend(
            T_wp=self.T_wp,
            active_wp=self.active_wp,
            bc_mask_wp=self.bc_mask_wp,
            gh_mask_wp=self.gh_mask_wp,
            ghb_factor_wp=self.ghb_factor_wp,
        )
        if fine_backend == "device":
            self._update_diag_preconditioner_device(
                T_wp=self.T_wp,
                active_wp=self.active_wp,
                bc_mask_wp=self.bc_mask_wp,
                gh_mask_wp=self.gh_mask_wp,
                ghb_factor_wp=self.ghb_factor_wp,
                M_inv_wp=self.M_inv_wp,
                nx=int(self.nx),
                ny=int(self.ny),
                use_ghb=bool(self.use_ghb),
                storage_diag_wp=getattr(self, "storage_diag_wp", None),
            )
            self._validate_device_diag_preconditioner(
                level_name="fine_build_from_truth_inputs",
                T_field=self.T_field_host,
                active=self.active_host,
                bc_mask=self.bc_mask_host,
                gh_mask=self.gh_mask_host if self.use_ghb else None,
                ghb_factor=self.ghb_factor_host if self.use_ghb else None,
                dx=float(self.dx) if self.use_ghb else None,
                M_inv_wp=self.M_inv_wp,
                storage_diag=self.storage_diag_host,
            )
        else:
            wp.copy(self.M_inv_wp, wp.array(M_inv_host, dtype=WP_FLOAT, device="cpu"))

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
            ghb_factor_wp=self.ghb_factor_wp,
            storage_diag_wp=getattr(self, 'storage_diag_wp', None),
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
                ghb_factor_wp=self.ghb_factor_c_wp,
                storage_diag_wp=getattr(self, 'storage_diag_c_wp', None),
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
        gh_alpha: float | np.ndarray = 1.0,
        aq_thickness: float | np.ndarray | None = None,
    ) -> None:
        """
        Build solver state from explicitly provided fields (no synthetic builder).
        """
        self._gh_alpha_input = gh_alpha
        if aq_thickness is not None:
            self._aq_thickness_input = aq_thickness

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
        self.storage_diag_host = None
        self.storage_diag_wp = None
        self.storage_diag_c_host = None
        self.storage_diag_c_wp = None

        self._sanitize_ghb_host_fields()
        self._recompute_ghb_factor_host(gh_alpha=self._gh_alpha_input, aq_thickness=aq_thickness)
        self._prune_isolated_active_host_cells()
        self._recompute_ghb_factor_host()
        self.n_active = int(np.count_nonzero(self.active_host))

        device = self.device_str

        self.T_wp = wp.array(self.T_field_host, dtype=WP_FLOAT, device=device)
        self.R_wp = wp.array(self.R_field_host, dtype=WP_FLOAT, device=device)
        self.active_wp = wp.array(self.active_host, dtype=wp.int32, device=device)
        self.bc_mask_wp = wp.array(self.bc_mask_host, dtype=wp.int32, device=device)
        self.bc_values_wp = wp.array(self.bc_values_host, dtype=WP_FLOAT, device=device)

        self.gh_mask_wp = wp.array(self.gh_mask_host, dtype=wp.int32, device=device)
        self.gh_head_wp = wp.array(self.gh_head_host, dtype=WP_FLOAT, device=device)
        self.gh_width_wp = wp.array(self.gh_width_host, dtype=WP_FLOAT, device=device)
        self._upload_ghb_factor_to_device(device=device)

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
            ghb_factor=self.ghb_factor_host if self.use_ghb else None,
            dx=float(self.dx) if self.use_ghb else None,
            storage_diag=self.storage_diag_host,
        )
        self.M_inv_wp = wp.empty((int(self.ny), int(self.nx)), dtype=WP_FLOAT, device=device)
        fine_backend = self._select_diag_preconditioner_backend(
            T_wp=self.T_wp,
            active_wp=self.active_wp,
            bc_mask_wp=self.bc_mask_wp,
            gh_mask_wp=self.gh_mask_wp,
            ghb_factor_wp=self.ghb_factor_wp,
        )
        if fine_backend == "device":
            self._update_diag_preconditioner_device(
                T_wp=self.T_wp,
                active_wp=self.active_wp,
                bc_mask_wp=self.bc_mask_wp,
                gh_mask_wp=self.gh_mask_wp,
                ghb_factor_wp=self.ghb_factor_wp,
                M_inv_wp=self.M_inv_wp,
                nx=int(self.nx),
                ny=int(self.ny),
                use_ghb=bool(self.use_ghb),
                storage_diag_wp=getattr(self, "storage_diag_wp", None),
            )
            self._validate_device_diag_preconditioner(
                level_name="fine_build_from_fields",
                T_field=self.T_field_host,
                active=self.active_host,
                bc_mask=self.bc_mask_host,
                gh_mask=self.gh_mask_host if self.use_ghb else None,
                ghb_factor=self.ghb_factor_host if self.use_ghb else None,
                dx=float(self.dx) if self.use_ghb else None,
                M_inv_wp=self.M_inv_wp,
                storage_diag=self.storage_diag_host,
            )
        else:
            wp.copy(self.M_inv_wp, wp.array(M_inv_host, dtype=WP_FLOAT, device="cpu"))

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
            ghb_factor_wp=self.ghb_factor_wp,
            storage_diag_wp=getattr(self, 'storage_diag_wp', None),
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
                ghb_factor_wp=self.ghb_factor_c_wp,
                storage_diag_wp=getattr(self, 'storage_diag_c_wp', None),
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
            ghb_factor_c_host,
            storage_diag_c_host,
        ) = build_coarse_level_from_fine(
            T_f=self.T_field_host,
            R_f=self.R_field_host,
            active_f=self.active_host,
            bc_mask_f=self.bc_mask_host,
            bc_values_f=self.bc_values_host,
            gh_mask_f=self.gh_mask_host if self.use_ghb else None,
            gh_head_f=self.gh_head_host if self.use_ghb else None,
            gh_width_f=self.gh_width_host if self.use_ghb else None,
            ghb_factor_f=self.ghb_factor_host if self.use_ghb else None,
            storage_diag_f=getattr(self, 'storage_diag_host', None),
        )

        bc_values_c_host[...] = 0.0
        gh_head_c_host[...] = 0.0

        ny_c, nx_c = T_c_host.shape

        self.nx_c = int(nx_c)
        self.ny_c = int(ny_c)
        self.dx_c = 2.0 * float(self.dx)
        self.n_active_c = int(np.count_nonzero(active_c_host))
        self._two_level_coarsening_diag = self._coarsening_diag_entry(
            level_id=1,
            active_f=self.active_host,
            active_c=active_c_host,
            bc_mask_c=bc_mask_c_host,
        )

        self.T_c_host = T_c_host
        self.R_c_host = R_c_host
        self.active_c_host = active_c_host
        self.bc_mask_c_host = bc_mask_c_host
        self.bc_values_c_host = bc_values_c_host
        self.gh_mask_c_host = gh_mask_c_host
        self.gh_head_c_host = gh_head_c_host
        self.gh_width_c_host = gh_width_c_host
        self.ghb_factor_c_host = ghb_factor_c_host
        storage_active = bool(self._storage_active)
        self.storage_diag_c_host = storage_diag_c_host if storage_active else None

        self.T_c_wp = wp.array(T_c_host, dtype=WP_FLOAT, device=device)
        self.R_c_wp = wp.array(R_c_host, dtype=WP_FLOAT, device=device)
        self.active_c_wp = wp.array(active_c_host, dtype=wp.int32, device=device)
        self.bc_mask_c_wp = wp.array(bc_mask_c_host, dtype=wp.int32, device=device)
        self.bc_values_c_wp = wp.array(bc_values_c_host, dtype=WP_FLOAT, device=device)
        self.gh_mask_c_wp = wp.array(gh_mask_c_host, dtype=wp.int32, device=device)
        self.gh_head_c_wp = wp.array(gh_head_c_host, dtype=WP_FLOAT, device=device)
        self.gh_width_c_wp = wp.array(gh_width_c_host, dtype=WP_FLOAT, device=device)
        self.ghb_factor_c_wp = wp.array(ghb_factor_c_host, dtype=WP_FLOAT, device=device)
        if storage_active and storage_diag_c_host is not None:
            self.storage_diag_c_wp = wp.array(storage_diag_c_host, dtype=WP_FLOAT, device=device)
        else:
            self.storage_diag_c_wp = None

        self.M_inv_c_wp = wp.empty((int(ny_c), int(nx_c)), dtype=WP_FLOAT, device=device)
        coarse_backend = self._select_diag_preconditioner_backend(
            T_wp=self.T_c_wp,
            active_wp=self.active_c_wp,
            bc_mask_wp=self.bc_mask_c_wp,
            gh_mask_wp=self.gh_mask_c_wp,
            ghb_factor_wp=self.ghb_factor_c_wp,
        )
        if coarse_backend == "device":
            self._update_diag_preconditioner_device(
                T_wp=self.T_c_wp,
                active_wp=self.active_c_wp,
                bc_mask_wp=self.bc_mask_c_wp,
                gh_mask_wp=self.gh_mask_c_wp,
                ghb_factor_wp=self.ghb_factor_c_wp,
                M_inv_wp=self.M_inv_c_wp,
                nx=int(nx_c),
                ny=int(ny_c),
                use_ghb=bool(self.use_ghb),
                storage_diag_wp=self.storage_diag_c_wp if storage_active else None,
            )
            self._validate_device_diag_preconditioner(
                level_name="two_level_cache",
                T_field=T_c_host,
                active=active_c_host,
                bc_mask=bc_mask_c_host,
                gh_mask=gh_mask_c_host if self.use_ghb else None,
                ghb_factor=ghb_factor_c_host if self.use_ghb else None,
                dx=float(self.dx_c) if self.use_ghb else None,
                M_inv_wp=self.M_inv_c_wp,
                storage_diag=storage_diag_c_host if storage_active else None,
            )
        else:
            M_inv_c_host = build_diag_preconditioner(
                T_field=T_c_host,
                active=active_c_host,
                bc_mask=bc_mask_c_host,
                gh_mask=gh_mask_c_host if self.use_ghb else None,
                ghb_factor=ghb_factor_c_host if self.use_ghb else None,
                dx=float(self.dx_c) if self.use_ghb else None,
                storage_diag=storage_diag_c_host if storage_active else None,
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

    def _update_diag_preconditioner_device(
        self,
        *,
        T_wp,
        active_wp,
        bc_mask_wp,
        gh_mask_wp,
        ghb_factor_wp,
        M_inv_wp,
        nx: int,
        ny: int,
        use_ghb: bool,
        storage_diag_wp=None,
    ) -> None:
        """
        Rebuild a diagonal preconditioner directly on the active device.

        :param T_wp: device transmissivity array
        :param active_wp: device active mask
        :param bc_mask_wp: device Dirichlet mask
        :param gh_mask_wp: device GHB mask
        :param ghb_factor_wp: device GHB factor array
        :param M_inv_wp: device output preconditioner array
        :param nx: number of columns
        :param ny: number of rows
        :param use_ghb: whether GHB diagonal terms should be included
        """
        if storage_diag_wp is None:
            # Steady no-storage path: no storage array to read, so use the
            # no-storage kernel variant and skip allocating a throwaway zero.
            wp.launch(
                build_diag_preconditioner_no_storage_kernel,
                dim=(int(ny), int(nx)),
                inputs=[
                    T_wp,
                    active_wp,
                    bc_mask_wp,
                    gh_mask_wp,
                    ghb_factor_wp,
                    int(1 if use_ghb else 0),
                    int(nx),
                    int(ny),
                    M_inv_wp,
                ],
                device=self.device_str,
            )
        else:
            wp.launch(
                build_diag_preconditioner_kernel,
                dim=(int(ny), int(nx)),
                inputs=[
                    T_wp,
                    active_wp,
                    bc_mask_wp,
                    gh_mask_wp,
                    ghb_factor_wp,
                    storage_diag_wp,
                    int(1 if use_ghb else 0),
                    int(nx),
                    int(ny),
                    M_inv_wp,
                ],
                device=self.device_str,
            )

    def _refresh_transient_device_hierarchy_values(
        self,
        *,
        levels,
    ) -> None:
        """
        Refresh dynamic transient operator values on device-only K-cycle levels.

        The transient fast path updates the fine-grid transmissivity and storage
        diagonal in device buffers. Coarse masks and hierarchy topology are
        static, but coarse transmissivity, storage diagonal, and diagonal
        preconditioner values must track the current Picard linearisation.

        :param levels: Multigrid levels whose level 0 arrays are already current.
        """
        if levels is None or len(levels) <= 1:
            return
        if self.use_ghb:
            raise NotImplementedError(
                "device-side transient fast path does not yet support GHB RHS/coarse refresh assembly"
            )

        device = self.device_str
        for lid in range(1, len(levels)):
            fine = levels[lid - 1]
            coarse = levels[lid]
            if getattr(fine, "storage_diag_wp", None) is None or getattr(coarse, "storage_diag_wp", None) is None:
                raise RuntimeError("transient device hierarchy is missing storage diagonal buffers")

            wp.launch(
                kernel=coarsen_transient_operator_level_kernel,
                dim=(int(coarse.ny), int(coarse.nx)),
                inputs=[
                    fine.T_wp,
                    fine.storage_diag_wp,
                    fine.active_wp,
                    coarse.active_wp,
                    coarse.bc_mask_wp,
                    int(fine.nx),
                    int(fine.ny),
                    int(coarse.nx),
                    int(coarse.ny),
                    coarse.T_wp,
                    coarse.storage_diag_wp,
                ],
                device=device,
            )
            self._update_diag_preconditioner_device(
                T_wp=coarse.T_wp,
                active_wp=coarse.active_wp,
                bc_mask_wp=coarse.bc_mask_wp,
                gh_mask_wp=coarse.gh_mask_wp,
                ghb_factor_wp=coarse.ghb_factor_wp,
                M_inv_wp=coarse.M_inv_wp,
                nx=int(coarse.nx),
                ny=int(coarse.ny),
                use_ghb=False,
                storage_diag_wp=coarse.storage_diag_wp,
            )

    def _validate_device_diag_preconditioner(
        self,
        *,
        level_name: str,
        T_field: np.ndarray,
        active: np.ndarray,
        bc_mask: np.ndarray,
        gh_mask: np.ndarray | None,
        ghb_factor: np.ndarray | None,
        dx: float | None,
        M_inv_wp,
        assume_finite_T: bool = False,
        storage_diag: np.ndarray | None = None,
    ) -> None:
        if not self._validate_device_m_inv_enabled():
            return

        M_host = build_diag_preconditioner(
            T_field=T_field,
            active=active,
            bc_mask=bc_mask,
            gh_mask=gh_mask,
            ghb_factor=ghb_factor,
            dx=dx,
            assume_finite_T=assume_finite_T,
            storage_diag=storage_diag,
        )
        if str(self.device_str).startswith("cuda"):
            wp.synchronize_device(self.device_str)
        M_device = np.asarray(M_inv_wp.numpy(), dtype=np.float64)
        M_host64 = np.asarray(M_host, dtype=np.float64)
        abs_diff = np.abs(M_device - M_host64)
        rel_base = np.maximum(np.abs(M_host64), 1.0e-30)
        rel_diff = abs_diff / rel_base
        max_abs_diff = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
        max_rel_diff = float(np.max(rel_diff)) if rel_diff.size > 0 else 0.0
        num_bad = int(np.count_nonzero(abs_diff > 1.0e-12))
        summary = {
            "level": str(level_name),
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "num_bad": num_bad,
        }
        self._last_device_m_inv_validation.append(summary)
        print(
            "[DARCY_VALIDATE_DEVICE_M_INV] "
            f"level={level_name} max_abs_diff={max_abs_diff:.6e} "
            f"max_rel_diff={max_rel_diff:.6e} num_bad={num_bad}"
        )

    def _zero_ghb_device_arrays(self, shape: tuple[int, int], device: str) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        gh_mask_wp = wp.zeros(shape, dtype=wp.int32, device=device)
        gh_head_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
        gh_width_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
        ghb_factor_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
        return gh_mask_wp, gh_head_wp, gh_width_wp, ghb_factor_wp

    def _build_rhs_fine_host(self, b_out_wp) -> None:
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
            ghb_factor=self.ghb_factor_host,
            head_scale=self.head_scale,
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

    def _build_rhs_fine_device(self, b_out_wp) -> None:
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
            or self.ghb_factor_wp is None
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
                self.ghb_factor_wp,
                nx,
                ny,
                float(self.dx),
                float(self.head_scale),
                b_out_wp,
            ],
            device=self.device_str,
        )

    def _build_rhs_fine(self, b_out_wp) -> None:
        """
        Assemble fine-grid RHS using configured backend.
        """
        backend = self._select_rhs_backend()
        if backend == "device":
            self._build_rhs_fine_device(b_out_wp)
        else:
            self._build_rhs_fine_host(b_out_wp)

    def _pcg_build_rhs_and_upload(self) -> None:
        """
        Build RHS for PCG backend.
        """
        self._ensure_pcg_buffers_fine(device=self.device_str)
        self._build_rhs_fine(self.b_wp)

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
        history_every: int | None = None,
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

        # PCG is steady-state only (transient PCG is rejected in solve()), so it
        # never carries a storage diagonal. Use the no-storage kernels and do not
        # allocate a placeholder storage array.

        self._pcg_build_rhs_and_upload()
        self._pcg_initialize_guess_and_upload(initial_head=initial_head)
        self._pcg_reset_work_vectors()

        dim = (ny, nx)

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.rho_buf], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.rTr_buf], device=device)

        wp.launch(
            kernel=init_pcg_with_A_no_storage_kernel,
            dim=dim,
            inputs=[
                self.x_wp,
                self.b_wp,
                self.T_wp,
                self.active_wp,
                self.bc_mask_wp,
                self.gh_mask_wp,
                self.ghb_factor_wp,
                self.M_inv_wp,
                self.Ap_wp,
                self.r_wp,
                self.z_wp,
                self.p_wp,
                self.rho_buf,
                self.rTr_buf,
                nx,
                ny,
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
        history_every_i = None if history_every is None else int(history_every)
        if history_every_i is not None and history_every_i <= 0:
            history_every_i = None
        history: list[dict[str, float | int | bool]] = []
        if history_every_i is not None:
            history.append(
                {
                    "iter": 0,
                    "rms_res_phys": float(r_rms0_phys),
                    "tol_abs_phys": float(tol_abs_scaled * float(self.head_scale)),
                }
            )

        for it in range(int(max_iter)):
            n_iter_used = it + 1

            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[self.pAp_buf], device=device)
            wp.launch(
                kernel=apply_A_and_pAp_no_storage_kernel,
                dim=dim,
                inputs=[
                    self.T_wp,
                    self.active_wp,
                    self.bc_mask_wp,
                    self.gh_mask_wp,
                    self.ghb_factor_wp,
                    self.p_wp,
                    self.Ap_wp,
                    self.pAp_buf,
                    nx,
                    ny,
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

            if history_every_i is not None and (
                (n_iter_used % history_every_i) == 0 or n_iter_used == int(max_iter)
            ):
                rTr_now = float(self.rTr_buf.numpy()[0]) if self.n_active > 0 else 0.0
                if self.n_active > 0 and rTr_now >= 0.0:
                    r_rms_now_scaled = float(np.sqrt(rTr_now / float(self.n_active)))
                else:
                    r_rms_now_scaled = 0.0
                history.append(
                    {
                        "iter": int(n_iter_used),
                        "rms_res_phys": float(r_rms_now_scaled * float(self.head_scale)),
                        "tol_abs_phys": float(tol_abs_scaled * float(self.head_scale)),
                    }
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
        if history_every_i is not None:
            if (not history) or int(history[-1]["iter"]) != int(n_iter_used):
                history.append(
                    {
                        "iter": int(n_iter_used),
                        "rms_res_phys": float(r_rms_final_phys),
                        "tol_abs_phys": float(tol_abs_phys),
                    }
                )
            info["history_every"] = int(history_every_i)
            info["history"] = history

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
                ghb_factor_wp=self.ghb_factor_wp,
                storage_diag_wp=getattr(self, "storage_diag_wp", None),
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
                ghb_factor_wp=self.ghb_factor_c_wp,
                storage_diag_wp=getattr(self, "storage_diag_c_wp", None),
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

        if self.M_inv_wp is None:
            self.M_inv_wp = wp.empty((ny0, nx0), dtype=WP_FLOAT, device=device)
        backend = self._select_diag_preconditioner_backend(
            T_wp=self.T_wp,
            active_wp=self.active_wp,
            bc_mask_wp=self.bc_mask_wp,
            gh_mask_wp=self.gh_mask_wp,
            ghb_factor_wp=self.ghb_factor_wp,
        )
        if backend == "device":
            self._update_diag_preconditioner_device(
                T_wp=self.T_wp,
                active_wp=self.active_wp,
                bc_mask_wp=self.bc_mask_wp,
                gh_mask_wp=self.gh_mask_wp,
                ghb_factor_wp=self.ghb_factor_wp,
                M_inv_wp=self.M_inv_wp,
                nx=nx0,
                ny=ny0,
                use_ghb=bool(self.use_ghb),
                storage_diag_wp=getattr(self, 'storage_diag_wp', None),
            )
            self._validate_device_diag_preconditioner(
                level_name="fine_update",
                T_field=self.T_field_host,
                active=self.active_host,
                bc_mask=self.bc_mask_host,
                gh_mask=self.gh_mask_host if self.use_ghb else None,
                ghb_factor=self.ghb_factor_host if self.use_ghb else None,
                dx=float(self.dx) if self.use_ghb else None,
                M_inv_wp=self.M_inv_wp,
                storage_diag=getattr(self, 'storage_diag_host', None),
            )
        else:
            M_inv_host = build_diag_preconditioner(
                T_field=self.T_field_host,
                active=self.active_host,
                bc_mask=self.bc_mask_host,
                gh_mask=self.gh_mask_host if self.use_ghb else None,
                ghb_factor=self.ghb_factor_host if self.use_ghb else None,
                dx=float(self.dx) if self.use_ghb else None,
                storage_diag=getattr(self, 'storage_diag_host', None),
            ).astype(NP_FLOAT, copy=False)

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
        profile_enabled = self._profile_update_T_enabled()
        coarse_coarsening_s = 0.0
        coarse_m_inv_build_s = 0.0
        t_total_start = time.perf_counter() if profile_enabled else None

        def _profile_sync() -> None:
            if profile_enabled and str(device).startswith("cuda"):
                wp.synchronize_device(device)

        # -------- fine host update + upload --------
        t_phase = time.perf_counter() if profile_enabled else None
        self._update_fine_T_and_upload(T_truth)
        if profile_enabled:
            _profile_sync()
            fine_t_upload_s = time.perf_counter() - t_phase

        # -------- rebuild fine diagonal preconditioner --------
        t_phase = time.perf_counter() if profile_enabled else None
        self._update_fine_diag_preconditioner()
        if profile_enabled:
            _profile_sync()
            fine_m_inv_build_s = time.perf_counter() - t_phase

        # -------- update 2-level cache (if built) --------
        if self.mg_cache_built and (self.T_c_host is not None) and (self.T_c_wp is not None):
            t_phase = time.perf_counter() if profile_enabled else None
            storage_diag_fine = self._active_storage_diag_host()
            (
                T_c_new,
                R_c_new,
                active_c_new,
                bc_mask_c_new,
                bc_values_c_new,
                gh_mask_c_new,
                gh_head_c_new,
                gh_width_c_new,
                ghb_factor_c_new,
                storage_diag_c_new,
            ) = build_coarse_level_from_fine(
                T_f=self.T_field_host,
                R_f=self.R_field_host,
                active_f=self.active_host,
                bc_mask_f=self.bc_mask_host,
                bc_values_f=self.bc_values_host,
                gh_mask_f=self.gh_mask_host if self.use_ghb else None,
                gh_head_f=self.gh_head_host if self.use_ghb else None,
                gh_width_f=self.gh_width_host if self.use_ghb else None,
                ghb_factor_f=self.ghb_factor_host if self.use_ghb else None,
                storage_diag_f=storage_diag_fine,
            )
            if profile_enabled:
                coarse_coarsening_s += time.perf_counter() - t_phase

            # correction scheme conventions
            bc_values_c_new[...] = NP_FLOAT(0.0)
            if gh_head_c_new is not None:
                gh_head_c_new[...] = NP_FLOAT(0.0)
            self._two_level_coarsening_diag = self._coarsening_diag_entry(
                level_id=1,
                active_f=self.active_host,
                active_c=active_c_new,
                bc_mask_c=bc_mask_c_new,
            )

            # copy into existing coarse host arrays (no realloc)
            np.copyto(self.T_c_host, np.asarray(T_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.R_c_host, np.asarray(R_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.active_c_host, np.asarray(active_c_new, dtype=np.int32, order="C"))
            np.copyto(self.bc_mask_c_host, np.asarray(bc_mask_c_new, dtype=np.int32, order="C"))
            np.copyto(self.bc_values_c_host, np.asarray(bc_values_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.gh_mask_c_host, np.asarray(gh_mask_c_new, dtype=np.int32, order="C"))
            np.copyto(self.gh_width_c_host, np.asarray(gh_width_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.gh_head_c_host, np.asarray(gh_head_c_new, dtype=NP_FLOAT, order="C"))
            np.copyto(self.ghb_factor_c_host, np.asarray(ghb_factor_c_new, dtype=NP_FLOAT, order="C"))
            if bool(self._storage_active) and hasattr(self, 'storage_diag_c_host') and self.storage_diag_c_host is not None:
                np.copyto(self.storage_diag_c_host, np.asarray(storage_diag_c_new, dtype=NP_FLOAT, order="C"))

            nyc = int(self.ny_c)
            nxc = int(self.nx_c)

            if self._stage_Tc_2lvl is None or tuple(self._stage_Tc_2lvl.shape) != (nyc, nxc):
                self._stage_Tc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")
            if self._stage_Mc_2lvl is None or tuple(self._stage_Mc_2lvl.shape) != (nyc, nxc):
                self._stage_Mc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")
            if self._stage_Gc_2lvl is None or tuple(self._stage_Gc_2lvl.shape) != (nyc, nxc):
                self._stage_Gc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")
            if bool(self._storage_active) and (
                self._stage_Sc_2lvl is None or tuple(self._stage_Sc_2lvl.shape) != (nyc, nxc)
            ):
                self._stage_Sc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")

            self._stage_Tc_2lvl.numpy()[:, :] = self.T_c_host
            wp.copy(self.T_c_wp, self._stage_Tc_2lvl)
            self._stage_Gc_2lvl.numpy()[:, :] = self.ghb_factor_c_host
            wp.copy(self.ghb_factor_c_wp, self._stage_Gc_2lvl)
            if bool(self._storage_active) and self.storage_diag_c_wp is not None and self.storage_diag_c_host is not None:
                self._stage_Sc_2lvl.numpy()[:, :] = self.storage_diag_c_host
                wp.copy(self.storage_diag_c_wp, self._stage_Sc_2lvl)

            t_phase = time.perf_counter() if profile_enabled else None
            coarse_backend = self._select_diag_preconditioner_backend(
                T_wp=self.T_c_wp,
                active_wp=self.active_c_wp,
                bc_mask_wp=self.bc_mask_c_wp,
                gh_mask_wp=self.gh_mask_c_wp,
                ghb_factor_wp=self.ghb_factor_c_wp,
            )
            if coarse_backend == "device":
                self._update_diag_preconditioner_device(
                    T_wp=self.T_c_wp,
                    active_wp=self.active_c_wp,
                    bc_mask_wp=self.bc_mask_c_wp,
                    gh_mask_wp=self.gh_mask_c_wp,
                    ghb_factor_wp=self.ghb_factor_c_wp,
                    M_inv_wp=self.M_inv_c_wp,
                    nx=int(self.nx_c),
                    ny=int(self.ny_c),
                    use_ghb=bool(self.use_ghb),
                    storage_diag_wp=self.storage_diag_c_wp if bool(self._storage_active) else None,
                )
                self._validate_device_diag_preconditioner(
                    level_name="two_level_update",
                    T_field=self.T_c_host,
                    active=self.active_c_host,
                    bc_mask=self.bc_mask_c_host,
                    gh_mask=self.gh_mask_c_host if self.use_ghb else None,
                    ghb_factor=self.ghb_factor_c_host if self.use_ghb else None,
                    dx=float(self.dx_c) if self.use_ghb else None,
                    M_inv_wp=self.M_inv_c_wp,
                    storage_diag=self.storage_diag_c_host if bool(self._storage_active) else None,
                )
            else:
                M_inv_c_host = build_diag_preconditioner(
                    T_field=self.T_c_host,
                    active=self.active_c_host,
                    bc_mask=self.bc_mask_c_host,
                    gh_mask=self.gh_mask_c_host if self.use_ghb else None,
                    ghb_factor=self.ghb_factor_c_host if self.use_ghb else None,
                    dx=float(self.dx_c) if self.use_ghb else None,
                    storage_diag=self.storage_diag_c_host if bool(self._storage_active) else None,
                ).astype(NP_FLOAT, copy=False)
                self._stage_Mc_2lvl.numpy()[:, :] = M_inv_c_host
                wp.copy(self.M_inv_c_wp, self._stage_Mc_2lvl)
            if profile_enabled:
                _profile_sync()
                coarse_m_inv_build_s += time.perf_counter() - t_phase

            if self._coarse_level is not None:
                self._coarse_level.T_wp = self.T_c_wp
                self._coarse_level.ghb_factor_wp = self.ghb_factor_c_wp
                self._coarse_level.M_inv_wp = self.M_inv_c_wp

        # -------- update full MG hierarchy (K-cycle) if it exists --------
        if self.mg_levels is not None:
            levels = self.mg_levels
            nL = int(len(levels))
            updated_diags = []

            if (
                self._stage_T_levels is None
                or self._stage_M_levels is None
                or self._stage_G_levels is None
                or (bool(self._storage_active) and self._stage_S_levels is None)
                or len(self._stage_T_levels) != nL
                or (bool(self._storage_active) and len(self._stage_S_levels) != nL)
            ):
                self._stage_T_levels = []
                self._stage_M_levels = []
                self._stage_G_levels = []
                self._stage_S_levels = [] if bool(self._storage_active) else None
                for lvl in levels:
                    self._stage_T_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))
                    self._stage_M_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))
                    self._stage_G_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))
                    if bool(self._storage_active):
                        self._stage_S_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))

            # Level 0: make sure level 0 host matches solver host
            lvl0 = levels[0]
            if tuple(lvl0.T_host.shape) == tuple(self.T_field_host.shape):
                np.copyto(lvl0.T_host, self.T_field_host)
            else:
                raise RuntimeError("Level 0 host shape mismatch. Rebuild hierarchy.")

            self._stage_T_levels[0].numpy()[:, :] = lvl0.T_host
            wp.copy(lvl0.T_wp, self._stage_T_levels[0])
            if getattr(lvl0, "ghb_factor_host", None) is not None and getattr(lvl0, "ghb_factor_wp", None) is not None:
                self._stage_G_levels[0].numpy()[:, :] = lvl0.ghb_factor_host
                wp.copy(lvl0.ghb_factor_wp, self._stage_G_levels[0])
            if bool(self._storage_active) and getattr(lvl0, "storage_diag_wp", None) is not None:
                if getattr(lvl0, "storage_diag_host", None) is None:
                    lvl0.storage_diag_host = np.zeros((int(lvl0.ny), int(lvl0.nx)), dtype=NP_FLOAT)
                if self.storage_diag_host is not None:
                    np.copyto(lvl0.storage_diag_host, self.storage_diag_host)
                else:
                    lvl0.storage_diag_host.fill(NP_FLOAT(0.0))
                self._stage_S_levels[0].numpy()[:, :] = lvl0.storage_diag_host
                wp.copy(lvl0.storage_diag_wp, self._stage_S_levels[0])
            else:
                lvl0.storage_diag_host = None
                if bool(self._storage_active):
                    self._operator_dirty = True

            t_phase = time.perf_counter() if profile_enabled else None
            lvl0_backend = self._select_diag_preconditioner_backend(
                T_wp=lvl0.T_wp,
                active_wp=lvl0.active_wp,
                bc_mask_wp=lvl0.bc_mask_wp,
                gh_mask_wp=lvl0.gh_mask_wp,
                ghb_factor_wp=lvl0.ghb_factor_wp,
            )
            if lvl0_backend == "device":
                self._update_diag_preconditioner_device(
                    T_wp=lvl0.T_wp,
                    active_wp=lvl0.active_wp,
                    bc_mask_wp=lvl0.bc_mask_wp,
                    gh_mask_wp=lvl0.gh_mask_wp,
                    ghb_factor_wp=lvl0.ghb_factor_wp,
                    M_inv_wp=lvl0.M_inv_wp,
                    nx=int(lvl0.nx),
                    ny=int(lvl0.ny),
                    use_ghb=bool(self.use_ghb),
                    storage_diag_wp=getattr(lvl0, "storage_diag_wp", None) if bool(self._storage_active) else None,
                )
                self._validate_device_diag_preconditioner(
                    level_name="mg_level_0_update",
                    T_field=lvl0.T_host,
                    active=lvl0.active_host,
                    bc_mask=lvl0.bc_mask_host,
                    gh_mask=lvl0.gh_mask_host if self.use_ghb else None,
                    ghb_factor=lvl0.ghb_factor_host if self.use_ghb else None,
                    dx=float(lvl0.dx) if self.use_ghb else None,
                    M_inv_wp=lvl0.M_inv_wp,
                    storage_diag=getattr(lvl0, "storage_diag_host", None) if bool(self._storage_active) else None,
                )
            else:
                M0 = build_diag_preconditioner(
                    T_field=lvl0.T_host,
                    active=lvl0.active_host,
                    bc_mask=lvl0.bc_mask_host,
                    gh_mask=lvl0.gh_mask_host if self.use_ghb else None,
                    ghb_factor=lvl0.ghb_factor_host if self.use_ghb else None,
                    dx=float(lvl0.dx) if self.use_ghb else None,
                    storage_diag=getattr(lvl0, "storage_diag_host", None) if bool(self._storage_active) else None,
                ).astype(NP_FLOAT, copy=False)
                self._stage_M_levels[0].numpy()[:, :] = M0
                wp.copy(lvl0.M_inv_wp, self._stage_M_levels[0])
            if profile_enabled:
                _profile_sync()
                coarse_m_inv_build_s += time.perf_counter() - t_phase

            # Coarse levels: re-coarsen from previous level and update T + M_inv
            for lid in range(1, nL):
                fine = levels[lid - 1]
                coarse = levels[lid]

                t_phase = time.perf_counter() if profile_enabled else None
                storage_diag_fine = getattr(fine, 'storage_diag_host', None) if bool(self._storage_active) else None
                (
                    T_c,
                    R_c,
                    active_c,
                    bc_mask_c,
                    bc_values_c,
                    gh_mask_c,
                    gh_head_c,
                    gh_width_c,
                    ghb_factor_c,
                    storage_diag_c,
                ) = self._mg_coarsen_host_any(
                    T_f=fine.T_host,
                    R_f=fine.R_host,
                    active_f=fine.active_host,
                    bc_mask_f=fine.bc_mask_host,
                    bc_values_f=fine.bc_values_host,
                    gh_mask_f=fine.gh_mask_host if self.use_ghb else None,
                    gh_head_f=fine.gh_head_host if self.use_ghb else None,
                    gh_width_f=fine.gh_width_host if self.use_ghb else None,
                    ghb_factor_f=fine.ghb_factor_host if self.use_ghb else None,
                    dx_c=float(coarse.dx),
                    storage_diag_f=storage_diag_fine,
                )
                if profile_enabled:
                    coarse_coarsening_s += time.perf_counter() - t_phase

                bc_values_c.fill(NP_FLOAT(0.0))
                if gh_head_c is not None:
                    gh_head_c.fill(NP_FLOAT(0.0))
                updated_diags.append(
                    self._coarsening_diag_entry(
                        level_id=int(lid),
                        active_f=fine.active_host,
                        active_c=active_c,
                        bc_mask_c=bc_mask_c,
                    )
                )

                if T_c.shape != coarse.T_host.shape:
                    raise RuntimeError(f"Level {lid} shape mismatch. Rebuild hierarchy.")

                np.copyto(coarse.T_host, T_c)
                if coarse.ghb_factor_host is not None and ghb_factor_c is not None:
                    np.copyto(coarse.ghb_factor_host, ghb_factor_c)
                if bool(self._storage_active) and getattr(coarse, 'storage_diag_host', None) is not None and storage_diag_c is not None:
                    np.copyto(coarse.storage_diag_host, storage_diag_c)
                else:
                    coarse.storage_diag_host = None

                self._stage_T_levels[lid].numpy()[:, :] = coarse.T_host
                wp.copy(coarse.T_wp, self._stage_T_levels[lid])
                if coarse.ghb_factor_wp is not None and coarse.ghb_factor_host is not None:
                    self._stage_G_levels[lid].numpy()[:, :] = coarse.ghb_factor_host
                    wp.copy(coarse.ghb_factor_wp, self._stage_G_levels[lid])
                if bool(self._storage_active) and getattr(coarse, 'storage_diag_wp', None) is not None and storage_diag_c is not None:
                    self._stage_S_levels[lid].numpy()[:, :] = coarse.storage_diag_host
                    wp.copy(coarse.storage_diag_wp, self._stage_S_levels[lid])

                t_phase = time.perf_counter() if profile_enabled else None
                coarse_backend = self._select_diag_preconditioner_backend(
                    T_wp=coarse.T_wp,
                    active_wp=coarse.active_wp,
                    bc_mask_wp=coarse.bc_mask_wp,
                    gh_mask_wp=coarse.gh_mask_wp,
                    ghb_factor_wp=coarse.ghb_factor_wp,
                )
                if coarse_backend == "device":
                    self._update_diag_preconditioner_device(
                        T_wp=coarse.T_wp,
                        active_wp=coarse.active_wp,
                        bc_mask_wp=coarse.bc_mask_wp,
                        gh_mask_wp=coarse.gh_mask_wp,
                        ghb_factor_wp=coarse.ghb_factor_wp,
                        M_inv_wp=coarse.M_inv_wp,
                        nx=int(coarse.nx),
                        ny=int(coarse.ny),
                        use_ghb=bool(self.use_ghb),
                        storage_diag_wp=getattr(coarse, "storage_diag_wp", None) if bool(self._storage_active) else None,
                    )
                    self._validate_device_diag_preconditioner(
                        level_name=f"mg_level_{int(lid)}_update",
                        T_field=coarse.T_host,
                        active=coarse.active_host,
                        bc_mask=coarse.bc_mask_host,
                        gh_mask=coarse.gh_mask_host if self.use_ghb else None,
                        ghb_factor=coarse.ghb_factor_host if self.use_ghb else None,
                        dx=float(coarse.dx) if self.use_ghb else None,
                        M_inv_wp=coarse.M_inv_wp,
                        storage_diag=getattr(coarse, "storage_diag_host", None) if bool(self._storage_active) else None,
                    )
                else:
                    Mc = build_diag_preconditioner(
                        T_field=coarse.T_host,
                        active=coarse.active_host,
                        bc_mask=coarse.bc_mask_host,
                        gh_mask=coarse.gh_mask_host if self.use_ghb else None,
                        ghb_factor=coarse.ghb_factor_host if self.use_ghb else None,
                        dx=float(coarse.dx) if self.use_ghb else None,
                        storage_diag=getattr(coarse, "storage_diag_host", None) if bool(self._storage_active) else None,
                    ).astype(NP_FLOAT, copy=False)

                    self._stage_M_levels[lid].numpy()[:, :] = Mc
                    wp.copy(coarse.M_inv_wp, self._stage_M_levels[lid])
                if profile_enabled:
                    _profile_sync()
                    coarse_m_inv_build_s += time.perf_counter() - t_phase

            self._mg_coarsening_diagnostics = updated_diags

        # Operator was updated in place, hierarchy shape and structure remains unchanged.
        # Do not mark dirty unless we explicitly know a rebuild is needed.
        if profile_enabled:
            profile = {
                "fine_t_upload_s": float(fine_t_upload_s),
                "fine_m_inv_build_s": float(fine_m_inv_build_s),
                "coarse_coarsening_s": float(coarse_coarsening_s),
                "coarse_m_inv_build_s": float(coarse_m_inv_build_s),
                "total_update_T_in_place_s": float(time.perf_counter() - t_total_start),
            }
            self._last_update_T_profile = profile
            if self._update_T_profile_totals is None:
                self._update_T_profile_totals = dict(profile)
                self._update_T_profile_totals["count"] = 1
            else:
                for key, value in profile.items():
                    self._update_T_profile_totals[key] = float(self._update_T_profile_totals.get(key, 0.0)) + float(value)
                self._update_T_profile_totals["count"] = int(self._update_T_profile_totals.get("count", 0)) + 1

    def update_ghb_factor_in_place(
        self,
        *,
        gh_alpha: float | np.ndarray | None = None,
        aq_thickness: float | np.ndarray | None = None,
    ) -> None:
        """
        Recompute and upload ghb_factor from raw gh_width, then refresh diagonal preconditioners.
        """
        if self.T_field_host is None:
            raise RuntimeError("Call build_from_truth_inputs() once before update_ghb_factor_in_place().")

        device = self.device_str
        self._sanitize_ghb_host_fields()
        self._recompute_ghb_factor_host(gh_alpha=gh_alpha, aq_thickness=aq_thickness)
        self._upload_ghb_factor_to_device(device=device)

        if self._fine_level is not None:
            self._fine_level.ghb_factor_wp = self.ghb_factor_wp

        self._update_fine_diag_preconditioner()

        if self.mg_cache_built and (self.ghb_factor_c_host is not None) and (self.ghb_factor_c_wp is not None):
            storage_diag_fine = self._active_storage_diag_host()
            (
                _T_c_new,
                _R_c_new,
                _active_c_new,
                _bc_mask_c_new,
                _bc_values_c_new,
                _gh_mask_c_new,
                _gh_head_c_new,
                _gh_width_c_new,
                ghb_factor_c_new,
                storage_diag_c_new,
            ) = build_coarse_level_from_fine(
                T_f=self.T_field_host,
                R_f=self.R_field_host,
                active_f=self.active_host,
                bc_mask_f=self.bc_mask_host,
                bc_values_f=self.bc_values_host,
                gh_mask_f=self.gh_mask_host if self.use_ghb else None,
                gh_head_f=self.gh_head_host if self.use_ghb else None,
                gh_width_f=self.gh_width_host if self.use_ghb else None,
                ghb_factor_f=self.ghb_factor_host if self.use_ghb else None,
                storage_diag_f=storage_diag_fine,
            )

            np.copyto(self.ghb_factor_c_host, np.asarray(ghb_factor_c_new, dtype=NP_FLOAT, order="C"))
            nyc = int(self.ny_c)
            nxc = int(self.nx_c)
            if self._stage_Gc_2lvl is None or tuple(self._stage_Gc_2lvl.shape) != (nyc, nxc):
                self._stage_Gc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")
            if self._stage_Mc_2lvl is None or tuple(self._stage_Mc_2lvl.shape) != (nyc, nxc):
                self._stage_Mc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")
            if bool(self._storage_active) and (
                self._stage_Sc_2lvl is None or tuple(self._stage_Sc_2lvl.shape) != (nyc, nxc)
            ):
                self._stage_Sc_2lvl = wp.zeros((nyc, nxc), dtype=WP_FLOAT, device="cpu")

            self._stage_Gc_2lvl.numpy()[:, :] = self.ghb_factor_c_host
            wp.copy(self.ghb_factor_c_wp, self._stage_Gc_2lvl)
            if bool(self._storage_active) and self.storage_diag_c_host is not None and self.storage_diag_c_wp is not None:
                np.copyto(self.storage_diag_c_host, np.asarray(storage_diag_c_new, dtype=NP_FLOAT, order="C"))
                self._stage_Sc_2lvl.numpy()[:, :] = self.storage_diag_c_host
                wp.copy(self.storage_diag_c_wp, self._stage_Sc_2lvl)

            coarse_backend = self._select_diag_preconditioner_backend(
                T_wp=self.T_c_wp,
                active_wp=self.active_c_wp,
                bc_mask_wp=self.bc_mask_c_wp,
                gh_mask_wp=self.gh_mask_c_wp,
                ghb_factor_wp=self.ghb_factor_c_wp,
            )
            if coarse_backend == "device":
                self._update_diag_preconditioner_device(
                    T_wp=self.T_c_wp,
                    active_wp=self.active_c_wp,
                    bc_mask_wp=self.bc_mask_c_wp,
                    gh_mask_wp=self.gh_mask_c_wp,
                    ghb_factor_wp=self.ghb_factor_c_wp,
                    M_inv_wp=self.M_inv_c_wp,
                    nx=int(self.nx_c),
                    ny=int(self.ny_c),
                    use_ghb=bool(self.use_ghb),
                    storage_diag_wp=self.storage_diag_c_wp if bool(self._storage_active) else None,
                )
                self._validate_device_diag_preconditioner(
                    level_name="two_level_ghb_update",
                    T_field=self.T_c_host,
                    active=self.active_c_host,
                    bc_mask=self.bc_mask_c_host,
                    gh_mask=self.gh_mask_c_host if self.use_ghb else None,
                    ghb_factor=self.ghb_factor_c_host if self.use_ghb else None,
                    dx=float(self.dx_c) if self.use_ghb else None,
                    M_inv_wp=self.M_inv_c_wp,
                    storage_diag=self.storage_diag_c_host if bool(self._storage_active) else None,
                )
            else:
                M_inv_c_host = build_diag_preconditioner(
                    T_field=self.T_c_host,
                    active=self.active_c_host,
                    bc_mask=self.bc_mask_c_host,
                    gh_mask=self.gh_mask_c_host if self.use_ghb else None,
                    ghb_factor=self.ghb_factor_c_host if self.use_ghb else None,
                    dx=float(self.dx_c) if self.use_ghb else None,
                    storage_diag=self.storage_diag_c_host if bool(self._storage_active) else None,
                ).astype(NP_FLOAT, copy=False)

                self._stage_Mc_2lvl.numpy()[:, :] = M_inv_c_host
                wp.copy(self.M_inv_c_wp, self._stage_Mc_2lvl)

            if self._coarse_level is not None:
                self._coarse_level.ghb_factor_wp = self.ghb_factor_c_wp
                self._coarse_level.M_inv_wp = self.M_inv_c_wp

        if self.mg_levels is not None:
            levels = self.mg_levels
            nL = int(len(levels))
            if (
                self._stage_M_levels is None
                or self._stage_G_levels is None
                or (bool(self._storage_active) and self._stage_S_levels is None)
                or len(self._stage_M_levels) != nL
                or (bool(self._storage_active) and len(self._stage_S_levels) != nL)
            ):
                self._stage_M_levels = []
                self._stage_G_levels = []
                self._stage_S_levels = [] if bool(self._storage_active) else None
                for lvl in levels:
                    self._stage_M_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))
                    self._stage_G_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))
                    if bool(self._storage_active):
                        self._stage_S_levels.append(wp.zeros((int(lvl.ny), int(lvl.nx)), dtype=WP_FLOAT, device="cpu"))

            lvl0 = levels[0]
            if lvl0.ghb_factor_host is not None:
                np.copyto(lvl0.ghb_factor_host, self.ghb_factor_host)
                self._stage_G_levels[0].numpy()[:, :] = lvl0.ghb_factor_host
                wp.copy(lvl0.ghb_factor_wp, self._stage_G_levels[0])
            if bool(self._storage_active) and getattr(lvl0, "storage_diag_wp", None) is not None:
                if getattr(lvl0, "storage_diag_host", None) is None:
                    lvl0.storage_diag_host = np.zeros((int(lvl0.ny), int(lvl0.nx)), dtype=NP_FLOAT)
                if self.storage_diag_host is not None:
                    np.copyto(lvl0.storage_diag_host, self.storage_diag_host)
                else:
                    lvl0.storage_diag_host.fill(NP_FLOAT(0.0))
                self._stage_S_levels[0].numpy()[:, :] = lvl0.storage_diag_host
                wp.copy(lvl0.storage_diag_wp, self._stage_S_levels[0])
            else:
                lvl0.storage_diag_host = None
                if bool(self._storage_active):
                    self._operator_dirty = True

            lvl0_backend = self._select_diag_preconditioner_backend(
                T_wp=lvl0.T_wp,
                active_wp=lvl0.active_wp,
                bc_mask_wp=lvl0.bc_mask_wp,
                gh_mask_wp=lvl0.gh_mask_wp,
                ghb_factor_wp=lvl0.ghb_factor_wp,
            )
            if lvl0_backend == "device":
                self._update_diag_preconditioner_device(
                    T_wp=lvl0.T_wp,
                    active_wp=lvl0.active_wp,
                    bc_mask_wp=lvl0.bc_mask_wp,
                    gh_mask_wp=lvl0.gh_mask_wp,
                    ghb_factor_wp=lvl0.ghb_factor_wp,
                    M_inv_wp=lvl0.M_inv_wp,
                    nx=int(lvl0.nx),
                    ny=int(lvl0.ny),
                    use_ghb=bool(self.use_ghb),
                    storage_diag_wp=getattr(lvl0, "storage_diag_wp", None) if bool(self._storage_active) else None,
                )
                self._validate_device_diag_preconditioner(
                    level_name="mg_level_0_ghb_update",
                    T_field=lvl0.T_host,
                    active=lvl0.active_host,
                    bc_mask=lvl0.bc_mask_host,
                    gh_mask=lvl0.gh_mask_host if self.use_ghb else None,
                    ghb_factor=lvl0.ghb_factor_host if self.use_ghb else None,
                    dx=float(lvl0.dx) if self.use_ghb else None,
                    M_inv_wp=lvl0.M_inv_wp,
                    storage_diag=getattr(lvl0, "storage_diag_host", None) if bool(self._storage_active) else None,
                )
            else:
                M0 = build_diag_preconditioner(
                    T_field=lvl0.T_host,
                    active=lvl0.active_host,
                    bc_mask=lvl0.bc_mask_host,
                    gh_mask=lvl0.gh_mask_host if self.use_ghb else None,
                    ghb_factor=lvl0.ghb_factor_host if self.use_ghb else None,
                    dx=float(lvl0.dx) if self.use_ghb else None,
                    storage_diag=getattr(lvl0, "storage_diag_host", None) if bool(self._storage_active) else None,
                ).astype(NP_FLOAT, copy=False)
                self._stage_M_levels[0].numpy()[:, :] = M0
                wp.copy(lvl0.M_inv_wp, self._stage_M_levels[0])

            for lid in range(1, nL):
                fine = levels[lid - 1]
                coarse = levels[lid]

                storage_diag_fine = getattr(fine, 'storage_diag_host', None) if bool(self._storage_active) else None
                (
                    _T_c,
                    _R_c,
                    _active_c,
                    _bc_mask_c,
                    _bc_values_c,
                    _gh_mask_c,
                    _gh_head_c,
                    _gh_width_c,
                    ghb_factor_c,
                    storage_diag_c,
                ) = self._mg_coarsen_host_any(
                    T_f=fine.T_host,
                    R_f=fine.R_host,
                    active_f=fine.active_host,
                    bc_mask_f=fine.bc_mask_host,
                    bc_values_f=fine.bc_values_host,
                    gh_mask_f=fine.gh_mask_host if self.use_ghb else None,
                    gh_head_f=fine.gh_head_host if self.use_ghb else None,
                    gh_width_f=fine.gh_width_host if self.use_ghb else None,
                    ghb_factor_f=fine.ghb_factor_host if self.use_ghb else None,
                    dx_c=float(coarse.dx),
                    storage_diag_f=storage_diag_fine,
                )

                if coarse.ghb_factor_host is not None and ghb_factor_c is not None:
                    np.copyto(coarse.ghb_factor_host, ghb_factor_c)
                    self._stage_G_levels[lid].numpy()[:, :] = coarse.ghb_factor_host
                    wp.copy(coarse.ghb_factor_wp, self._stage_G_levels[lid])
                if bool(self._storage_active) and getattr(coarse, "storage_diag_host", None) is not None and storage_diag_c is not None:
                    np.copyto(coarse.storage_diag_host, storage_diag_c)
                    self._stage_S_levels[lid].numpy()[:, :] = coarse.storage_diag_host
                    wp.copy(coarse.storage_diag_wp, self._stage_S_levels[lid])
                else:
                    coarse.storage_diag_host = None

                coarse_backend = self._select_diag_preconditioner_backend(
                    T_wp=coarse.T_wp,
                    active_wp=coarse.active_wp,
                    bc_mask_wp=coarse.bc_mask_wp,
                    gh_mask_wp=coarse.gh_mask_wp,
                    ghb_factor_wp=coarse.ghb_factor_wp,
                )
                if coarse_backend == "device":
                    self._update_diag_preconditioner_device(
                        T_wp=coarse.T_wp,
                        active_wp=coarse.active_wp,
                        bc_mask_wp=coarse.bc_mask_wp,
                        gh_mask_wp=coarse.gh_mask_wp,
                        ghb_factor_wp=coarse.ghb_factor_wp,
                        M_inv_wp=coarse.M_inv_wp,
                        nx=int(coarse.nx),
                        ny=int(coarse.ny),
                        use_ghb=bool(self.use_ghb),
                        storage_diag_wp=getattr(coarse, "storage_diag_wp", None) if bool(self._storage_active) else None,
                    )
                    self._validate_device_diag_preconditioner(
                        level_name=f"mg_level_{int(lid)}_ghb_update",
                        T_field=coarse.T_host,
                        active=coarse.active_host,
                        bc_mask=coarse.bc_mask_host,
                        gh_mask=coarse.gh_mask_host if self.use_ghb else None,
                        ghb_factor=coarse.ghb_factor_host if self.use_ghb else None,
                        dx=float(coarse.dx) if self.use_ghb else None,
                        M_inv_wp=coarse.M_inv_wp,
                        storage_diag=getattr(coarse, "storage_diag_host", None) if bool(self._storage_active) else None,
                    )
                else:
                    Mc = build_diag_preconditioner(
                        T_field=coarse.T_host,
                        active=coarse.active_host,
                        bc_mask=coarse.bc_mask_host,
                        gh_mask=coarse.gh_mask_host if self.use_ghb else None,
                        ghb_factor=coarse.ghb_factor_host if self.use_ghb else None,
                        dx=float(coarse.dx) if self.use_ghb else None,
                        storage_diag=getattr(coarse, "storage_diag_host", None) if bool(self._storage_active) else None,
                    ).astype(NP_FLOAT, copy=False)
                    self._stage_M_levels[lid].numpy()[:, :] = Mc
                    wp.copy(coarse.M_inv_wp, self._stage_M_levels[lid])

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

    def update_uniform_recharge_in_place(self, recharge_rate: float) -> None:
        """
        Update a uniform recharge field on device from a scalar.

        The host mirror is kept in sync for reporting/debug paths, but the
        device field no longer requires staging or uploading a full recharge
        grid every transient period.

        :param recharge_rate: Uniform recharge rate for active cells.
        """
        if self.R_field_host is None or self.R_wp is None or self.active_wp is None:
            raise RuntimeError("Call build_from_truth_inputs() once before update_uniform_recharge_in_place().")
        rate = NP_FLOAT(float(recharge_rate))
        self.R_field_host[:, :] = rate
        if self.active_host is not None:
            self.R_field_host[np.asarray(self.active_host, dtype=np.int32) == 0] = NP_FLOAT(0.0)
        wp.launch(
            kernel=fill_uniform_recharge_kernel,
            dim=(int(self.ny), int(self.nx)),
            inputs=[
                self.R_wp,
                self.active_wp,
                WP_FLOAT(rate),
                int(self.nx),
                int(self.ny),
            ],
            device=self.device_str,
        )

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



    def _solve_multigrid_kcycle_device_buffers(
        self,
        *,
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
    ):
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
        rTr_end = float(lvl0.rTr_buf.numpy()[0])
        gpu_scalar_sync_count += 1
        r_rms_end = float(np.sqrt(max(rTr_end, 0.0) / float(n_free0)))

        # Also get head residual
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
        if storage_diag_wp is not None:
            _hr_k = compute_head_residual_kernel
            _hr_in = [
                lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
                lvl0.gh_mask_wp, lvl0.ghb_factor_wp, storage_diag_wp,
                lvl0.r_wp, lvl0.rTr_buf, nx0, ny0
            ]
        else:
            _hr_k = compute_head_residual_no_storage_kernel
            _hr_in = [
                lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
                lvl0.gh_mask_wp, lvl0.ghb_factor_wp,
                lvl0.r_wp, lvl0.rTr_buf, nx0, ny0
            ]
        wp.launch(kernel=_hr_k, dim=dim0, inputs=_hr_in, device=device)
        hrTr_end = float(lvl0.rTr_buf.numpy()[0])
        gpu_scalar_sync_count += 1
        h_rms_end = float(np.sqrt(max(hrTr_end, 0.0) / float(n_free0)))

        info = {
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
        return info


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
            if K_field is None or zbot_field is None:
                raise ValueError("unconfined=True requires K_field and zbot_field.")
            if self.active_host is None or self.bc_mask_host is None or self.bc_values_host is None:
                raise RuntimeError("build_from_truth_inputs or build_from_fields must be called before solve.")

            ny0 = int(self.ny)
            nx0 = int(self.nx)
            shape0 = (ny0, nx0)

            K_arr = np.asarray(K_field, dtype=np.float64)
            zbot_arr = np.asarray(zbot_field, dtype=np.float64)
            if K_arr.shape != shape0:
                raise ValueError(f"K_field shape {K_arr.shape} expected {shape0}.")
            if zbot_arr.shape != shape0:
                raise ValueError(f"zbot_field shape {zbot_arr.shape} expected {shape0}.")
            if not np.all(np.isfinite(K_arr)) or np.any(K_arr < 0.0):
                raise ValueError("K_field must be finite and non-negative.")
            if not np.all(np.isfinite(zbot_arr)):
                raise ValueError("zbot_field must be finite.")

            ztop_arr = None
            if ztop_field is not None:
                ztop_arr = np.asarray(ztop_field, dtype=np.float64)
                if ztop_arr.shape != shape0:
                    raise ValueError(f"ztop_field shape {ztop_arr.shape} expected {shape0}.")
                if not np.all(np.isfinite(ztop_arr)):
                    raise ValueError("ztop_field must be finite.")

            min_sat = float(
                unconfined_min_sat
                if unconfined_min_sat is not None
                else (0.1 if min_saturated_thickness is None else min_saturated_thickness)
            )
            if min_sat <= 0.0 or not np.isfinite(min_sat):
                raise ValueError("min_saturated_thickness must be positive and finite.")

            max_outer = int(
                unconfined_max_picard_iter
                if unconfined_max_picard_iter is not None
                else (100 if max_outer_iterations is None else max_outer_iterations)
            )
            if max_outer < 1:
                raise ValueError("max_outer_iterations must be >= 1.")

            omega_current = float(unconfined_relax if unconfined_relax is not None else omega)
            omega_min_f = float(omega_min)
            omega_max_f = float(omega_max)
            if not (0.0 < omega_min_f <= omega_max_f):
                raise ValueError("omega_min and omega_max must satisfy 0 < omega_min <= omega_max.")
            omega_current = min(max(omega_current, omega_min_f), omega_max_f)

            hclose_f = float(
                unconfined_head_tol
                if unconfined_head_tol is not None
                else (1.0e-4 if hclose is None else hclose)
            )
            if hclose_f < 0.0 or not np.isfinite(hclose_f):
                raise ValueError("hclose must be non-negative and finite.")

            residual_floor_tol_f = float(residual_floor_tol) if residual_floor_tol is not None else None
            if residual_floor_tol_f is not None and residual_floor_tol_f < 0.0:
                raise ValueError("residual_floor_tol must be non-negative.")

            inner_head_residual_tol_f = float(
                inner_head_residual_tol if inner_head_residual_tol is not None else hclose_f
            )
            if inner_head_residual_tol_f < 0.0 or not np.isfinite(inner_head_residual_tol_f):
                raise ValueError("inner_head_residual_tol must be non-negative and finite.")

            inner_max_cycles_early = int(unconfined_inner_max_cycles_early)
            inner_max_cycles_middle = int(unconfined_inner_max_cycles_middle)
            inner_max_cycles_late = int(unconfined_inner_max_cycles_late)
            if min(inner_max_cycles_early, inner_max_cycles_middle, inner_max_cycles_late) < 1:
                raise ValueError("unconfined inner max cycles must be >= 1.")

            inner_late_dh_f = float(unconfined_inner_late_dh)
            inner_middle_dh_f = float(unconfined_inner_middle_dh)
            if inner_late_dh_f < 0.0 or inner_middle_dh_f < 0.0:
                raise ValueError("unconfined inner dh thresholds must be non-negative.")

            inner_forcing_eta_f = float(inner_forcing_eta)
            if inner_forcing_eta_f < 0.0 or inner_forcing_eta_f > 1.0:
                raise ValueError("inner_forcing_eta must be in [0, 1].")

            inner_head_residual_tol_min_f = float(
                inner_head_residual_tol_min if inner_head_residual_tol_min is not None else hclose_f
            )
            if inner_head_residual_tol_min_f < 0.0 or not np.isfinite(inner_head_residual_tol_min_f):
                raise ValueError("inner_head_residual_tol_min must be non-negative and finite.")

            inner_head_residual_tol_max_f = float(inner_head_residual_tol_max)
            if inner_head_residual_tol_max_f < inner_head_residual_tol_min_f:
                raise ValueError("inner_head_residual_tol_max must be >= inner_head_residual_tol_min.")

            inner_picard_scale_max_fraction_f = float(inner_picard_scale_max_fraction)
            if inner_picard_scale_max_fraction_f < 0.0 or inner_picard_scale_max_fraction_f > 1.0:
                raise ValueError("inner_picard_scale_max_fraction must be in [0, 1].")

            chebyshev_reset_factor_f = float(chebyshev_reset_factor)
            if chebyshev_reset_factor_f <= 1.0 or not np.isfinite(chebyshev_reset_factor_f):
                raise ValueError("chebyshev_reset_factor must be finite and > 1.")

            chebyshev_minor_increase_patience_i = int(chebyshev_minor_increase_patience)
            if chebyshev_minor_increase_patience_i < 0:
                raise ValueError("chebyshev_minor_increase_patience must be >= 0.")

            transmissivity_relaxation_enabled_b = bool(transmissivity_relaxation_enabled)
            T_relax_early_f = float(transmissivity_relaxation_early)
            T_relax_middle_f = float(transmissivity_relaxation_middle)
            T_relax_late_f = float(transmissivity_relaxation_late)
            if not all(0.0 <= v <= 1.0 for v in (T_relax_early_f, T_relax_middle_f, T_relax_late_f)):
                raise ValueError("transmissivity relaxation factors must be in [0, 1].")

            T_relax_middle_iter = int(transmissivity_relaxation_middle_iteration)
            T_relax_late_iter = int(transmissivity_relaxation_late_iteration)
            if T_relax_middle_iter < 1 or T_relax_late_iter < T_relax_middle_iter:
                raise ValueError("transmissivity relaxation iterations must satisfy 1 <= middle <= late.")

            startup_mode = str(unconfined_startup_mode).strip().lower()
            if startup_mode not in {"initial_head", "confined_pre_solve", "unconfined_pre_solve"}:
                raise ValueError(
                    "unconfined_startup_mode must be 'initial_head', 'confined_pre_solve', "
                    "or 'unconfined_pre_solve'."
                )
            unconfined_pre_solve_iterations_i = int(unconfined_pre_solve_iterations)
            if unconfined_pre_solve_iterations_i < 1 or not np.isfinite(unconfined_pre_solve_iterations_i):
                raise ValueError("unconfined_pre_solve_iterations must be a finite integer >= 1.")

            storage_reference_mode = str(storage_reference).strip().lower()
            if storage_reference_mode not in {"previous_period", "current_picard"}:
                raise ValueError("storage_reference must be 'previous_period' or 'current_picard'.")
            storage_mode_2d = None if unconfined_storage_mode_2d is None else str(unconfined_storage_mode_2d).strip().lower()
            current_picard_storage = bool(transient) and storage_reference_mode == "current_picard"
            if current_picard_storage:
                if storage_mode_2d != "mf6_convertible_secant_sy":
                    raise ValueError(
                        "current_picard storage requires unconfined_storage_mode_2d to be "
                        "'mf6_convertible_secant_sy'."
                    )
                if sy is None or ss is None:
                    raise ValueError("current_picard storage requires sy and ss.")
                if ztop_arr is None:
                    raise ValueError("current_picard storage requires ztop_field.")
                sy_f = float(sy)
                ss_f = float(ss)
                if sy_f < 0.0 or ss_f < 0.0 or not np.isfinite(sy_f) or not np.isfinite(ss_f):
                    raise ValueError("sy and ss must be finite and non-negative.")
            else:
                sy_f = float("nan")
                ss_f = float("nan")

            max_update_f = float(max_head_change_per_outer_iteration)
            if max_update_f <= 0.0 or not np.isfinite(max_update_f):
                raise ValueError("max_head_change_per_outer_iteration must be positive and finite.")

            practical_picard_acceptance_enabled_b = bool(practical_picard_acceptance_enabled)
            min_practical_outer_iterations_i = int(min_practical_outer_iterations)
            if min_practical_outer_iterations_i < 1:
                raise ValueError("min_practical_outer_iterations must be >= 1.")
            practical_residual_tol_f = float(practical_residual_tol)
            practical_dh_rms_tol_f = float(practical_dh_rms_tol)
            practical_storage_diag_change_rms_tol_f = float(practical_storage_diag_change_rms_tol)
            if (
                practical_residual_tol_f < 0.0
                or practical_dh_rms_tol_f < 0.0
                or practical_storage_diag_change_rms_tol_f < 0.0
            ):
                raise ValueError("practical Picard tolerances must be non-negative.")
            save_transient_diagnostics_b = bool(save_transient_diagnostics)
            secant_sy_practical_mode = bool(
                practical_picard_acceptance_enabled_b
                and current_picard_storage
                and storage_mode_2d == "mf6_convertible_secant_sy"
            )

            initial_sat_f = float(initial_saturated_thickness)
            if initial_sat_f <= 0.0 or not np.isfinite(initial_sat_f):
                raise ValueError("initial_saturated_thickness must be positive and finite.")

            rejection_factor_f = float(chebyshev_rejection_factor)
            if rejection_factor_f <= 1.0 or not np.isfinite(rejection_factor_f):
                raise ValueError("chebyshev_rejection_factor must be finite and > 1.")

            active_mask = np.asarray(self.active_host, dtype=np.int32) != 0
            bc_mask0 = np.asarray(self.bc_mask_host, dtype=np.int32) != 0
            free_mask0 = active_mask & (~bc_mask0)
            bc_values0 = np.asarray(self.bc_values_host, dtype=NP_FLOAT)

            if initial_head is None:
                h_iter = (zbot_arr + max(initial_sat_f, min_sat)).astype(NP_FLOAT, copy=False)
            else:
                h_iter = np.asarray(initial_head, dtype=NP_FLOAT).copy()
                if h_iter.shape != shape0:
                    raise ValueError(f"initial_head must have shape {shape0}, got {h_iter.shape}.")
            h_iter[bc_mask0] = bc_values0[bc_mask0]
            h_iter[~active_mask] = NP_FLOAT(0.0)
            if not np.all(np.isfinite(h_iter)):
                raise ValueError("initial head for unconfined solve must be finite.")

            kc_base_kwargs = {
                "nu_pre": int(nu_pre),
                "nu_post": int(nu_post),
                "nu_coarse": int(nu_coarse),
                "omega": float(omega),
                "rel_tol": float(rel_tol),
                "abs_tol_min": float(abs_tol_min),
                "aq_thickness": aq_thickness,
                "gh_alpha": gh_alpha,
                "max_levels": int(max_levels),
                "check_every_no": int(check_every_no),
                "dh_rms_tol": dh_rms_tol,
                "dh_max_tol": dh_max_tol,
                "dh_max_factor": float(dh_max_factor),
                "min_coarse_cells": min_coarse_cells,
                "fallback_to_pcg": bool(fallback_to_pcg),
                "divergence_cycle_start": int(divergence_cycle_start),
                "divergence_residual_factor": float(divergence_residual_factor),
                "fallback_pcg_max_iter": fallback_pcg_max_iter,
                "fallback_pcg_history_every": fallback_pcg_history_every,
                "smoother": str(smoother_mode),
                "cheby_lambda_min": float(cheby_lambda_min),
                "cheby_lambda_max": float(cheby_lambda_max),
            }

            def _storage_from_picard_head(
                    head_ref_arr: np.ndarray,
            ) -> dict[str, np.ndarray]:
                head_ref64 = np.asarray(head_ref_arr, dtype=np.float64)
                if ztop_arr is None:
                    raise ValueError("ztop_field is required for current Picard storage.")
                full_thickness = np.maximum(ztop_arr - zbot_arr, min_sat)
                head_old64 = np.asarray(head_prev, dtype=np.float64)
                sat_ref_zero = np.clip(head_ref64 - zbot_arr, 0.0, full_thickness)
                sat_old_zero = np.clip(head_old64 - zbot_arr, 0.0, full_thickness)
                sat_ref_ss = np.clip(head_ref64 - zbot_arr, min_sat, full_thickness)
                sy_coeff = np.zeros(shape0, dtype=np.float64)
                ss_coeff = np.zeros(shape0, dtype=np.float64)
                dh_ref = head_ref64 - head_old64
                moving = np.abs(dh_ref) > 1.0e-12
                sy_coeff[moving] = sy_f * ((sat_ref_zero[moving] - sat_old_zero[moving]) / dh_ref[moving])
                fallback = (~moving) & (head_ref64 < ztop_arr) & (head_ref64 > zbot_arr)
                sy_coeff[fallback] = sy_f
                sy_coeff = np.clip(sy_coeff, 0.0, sy_f)
                ss_coeff[:, :] = ss_f * sat_ref_ss
                storage = sy_coeff + ss_coeff
                storage = storage.astype(NP_FLOAT, copy=False)
                storage[~free_mask0] = NP_FLOAT(0.0)
                sy_coeff = sy_coeff.astype(np.float64, copy=False)
                ss_coeff = ss_coeff.astype(np.float64, copy=False)
                sy_coeff[~free_mask0] = 0.0
                ss_coeff[~free_mask0] = 0.0
                sat_ref_zero = sat_ref_zero.astype(np.float64, copy=False)
                sat_old_zero = sat_old_zero.astype(np.float64, copy=False)
                sat_ref_ss = sat_ref_ss.astype(np.float64, copy=False)
                sat_ref_zero[~free_mask0] = 0.0
                sat_old_zero[~free_mask0] = 0.0
                sat_ref_ss[~free_mask0] = 0.0
                return {
                    "storage": storage,
                    "sy_coeff": sy_coeff,
                    "ss_coeff": ss_coeff,
                    "sat_ref_zero": sat_ref_zero,
                    "sat_old_zero": sat_old_zero,
                    "sat_ref_ss": sat_ref_ss,
                    "full_thickness": full_thickness.astype(np.float64, copy=False),
                    "head_ref": head_ref64.astype(np.float64, copy=False),
                }

            if startup_mode == "confined_pre_solve":
                sat_startup = h_iter.astype(np.float64, copy=False) - zbot_arr
                sat_startup = np.maximum(sat_startup, min_sat)
                if ztop_arr is not None:
                    sat_cap = np.maximum(ztop_arr - zbot_arr, min_sat)
                    sat_startup = np.minimum(sat_startup, sat_cap)
                T_startup = (K_arr * sat_startup).astype(NP_FLOAT, copy=False)
                T_startup[~active_mask] = NP_FLOAT(0.0)
                self.update_T_in_place(T_startup)
                storage_coeff_startup = (
                    _storage_from_picard_head(h_iter)["storage"]
                    if current_picard_storage
                    else storage_coeff
                )

                h_startup = self.solve_multigrid_kcycle(
                    max_cycles=int(max_cycles),
                    initial_head=h_iter,
                    return_info=False,
                    unconfined=False,
                    transient=transient,
                    storage_coeff=storage_coeff_startup,
                    dt=dt,
                    head_prev=head_prev,
                    refresh_diag_with_transient_storage=True,
                    **kc_base_kwargs,
                )
                h_startup = np.asarray(h_startup, dtype=np.float64)
                h_startup = np.maximum(h_startup, zbot_arr + min_sat)
                if ztop_arr is not None:
                    h_startup = np.minimum(h_startup, ztop_arr)
                h_startup[~active_mask] = 0.0
                h_startup[bc_mask0] = bc_values0[bc_mask0]
                if not np.all(np.isfinite(h_startup)):
                    raise FloatingPointError("confined pre-solve produced non-finite heads.")
                h_iter = h_startup.astype(NP_FLOAT, copy=False)

            elif startup_mode == "unconfined_pre_solve":
                # Unconfined warm start: a small fixed number of Picard
                # sub-iterations that rebuild transmissivity from the current
                # head (unconfined linearisation), each solved as a linearised
                # K-cycle step. Unlike ``confined_pre_solve`` (one fixed-T
                # solve), this lets the saturated thickness relax before the
                # main transient Picard loop begins. The storage term follows
                # the same reference as the main solve (current Picard head when
                # ``storage_reference='current_picard'``).
                h_pre = np.asarray(h_iter, dtype=np.float64)
                if ztop_arr is not None:
                    sat_cap_pre = np.maximum(ztop_arr - zbot_arr, min_sat)
                for _ in range(unconfined_pre_solve_iterations_i):
                    sat_pre = np.maximum(h_pre - zbot_arr, min_sat)
                    if ztop_arr is not None:
                        sat_pre = np.minimum(sat_pre, sat_cap_pre)
                    T_pre = (K_arr * sat_pre).astype(NP_FLOAT, copy=False)
                    T_pre[~active_mask] = NP_FLOAT(0.0)
                    self.update_T_in_place(T_pre)
                    storage_coeff_pre = (
                        _storage_from_picard_head(h_pre)["storage"]
                        if current_picard_storage
                        else storage_coeff
                    )
                    h_pre = self.solve_multigrid_kcycle(
                        max_cycles=int(max_cycles),
                        initial_head=h_pre,
                        return_info=False,
                        unconfined=False,
                        transient=transient,
                        storage_coeff=storage_coeff_pre,
                        dt=dt,
                        head_prev=head_prev,
                        refresh_diag_with_transient_storage=True,
                        **kc_base_kwargs,
                    )
                    h_pre = np.asarray(h_pre, dtype=np.float64)
                    h_pre = np.maximum(h_pre, zbot_arr + min_sat)
                    if ztop_arr is not None:
                        h_pre = np.minimum(h_pre, ztop_arr)
                    h_pre[~active_mask] = 0.0
                    h_pre[bc_mask0] = bc_values0[bc_mask0]
                    if not np.all(np.isfinite(h_pre)):
                        raise FloatingPointError("unconfined pre-solve produced non-finite heads.")
                h_iter = h_pre.astype(NP_FLOAT, copy=False)

            cheb_weights = _chebyshev_update_weights(
                order=int(chebyshev_order),
                lambda_min_fraction=float(chebyshev_lambda_min_fraction),
            )
            previous_update = np.zeros(shape0, dtype=np.float64)
            previous_measure = float("inf")
            chebyshev_rejections = 0
            chebyshev_resets = 0
            inner_solve_failures = 0
            strict_inner_nonconvergence_count = 0
            unusable_inner_solve_count = 0
            practical_inner_acceptances = 0
            accepted_picard_update_count = 0
            outer_chebyshev_ready_count = 0
            outer_chebyshev_used_count = 0
            outer_chebyshev_reset_count = 0
            improvement_streak = 0
            minor_increase_count = 0
            final_residual = None
            final_h_rms_end = float("nan")
            final_inner_max_cycles = 0
            final_max_abs_head_change = float("nan")
            last_linear_info: dict = {}
            outer_history: list[dict] = []
            strict_picard_convergence_passed = False
            practical_picard_acceptance_passed = False
            production_acceptance_passed = False
            T_previous: np.ndarray | None = None
            T_relax = float("nan")
            previous_storage_diag_arr: np.ndarray | None = None
            max_storage_diag_change_max = 0.0
            max_storage_diag_change_rms = 0.0
            last_storage_coeff_array: np.ndarray | None = None
            last_sy_storage_coeff_array: np.ndarray | None = None
            last_ss_storage_coeff_array: np.ndarray | None = None
            last_storage_reference_head_array: np.ndarray | None = None

            def _to_finite(value):
                try:
                    f = float(value)
                    return f if np.isfinite(f) else None
                except Exception:
                    return None

            for outer_idx in range(max_outer):
                if not np.isfinite(previous_measure):
                    inner_max_cycles = inner_max_cycles_early
                elif previous_measure > inner_middle_dh_f:
                    inner_max_cycles = inner_max_cycles_early
                elif previous_measure > inner_late_dh_f:
                    inner_max_cycles = inner_max_cycles_middle
                else:
                    inner_max_cycles = inner_max_cycles_late

                sat = h_iter.astype(np.float64, copy=False) - zbot_arr
                sat = np.maximum(sat, min_sat)
                if ztop_arr is not None:
                    sat_cap = np.maximum(ztop_arr - zbot_arr, min_sat)
                    sat = np.minimum(sat, sat_cap)
                if not np.all(np.isfinite(sat)) or np.any(sat <= 0.0):
                    raise FloatingPointError("unconfined saturated thickness became invalid.")

                T_candidate = (K_arr * sat).astype(NP_FLOAT, copy=False)
                T_candidate[~active_mask] = NP_FLOAT(0.0)
                if not np.all(np.isfinite(T_candidate)):
                    raise FloatingPointError("unconfined transmissivity became non-finite.")

                if transmissivity_relaxation_enabled_b and outer_idx > 0 and T_previous is not None:
                    if outer_idx < T_relax_middle_iter:
                        T_relax = T_relax_early_f
                    elif outer_idx < T_relax_late_iter:
                        T_relax = T_relax_middle_f
                    else:
                        T_relax = T_relax_late_f
                    T_pic = (1.0 - T_relax) * T_previous + T_relax * T_candidate
                else:
                    T_pic = T_candidate
                    T_relax = float("nan")

                T_pic[~active_mask] = NP_FLOAT(0.0)
                if not np.all(np.isfinite(T_pic)):
                    raise FloatingPointError("unconfined transmissivity became non-finite.")

                self.update_T_in_place(T_pic)
                T_previous = T_pic.copy()
                storage_sy_coeff_arr = None
                storage_ss_coeff_arr = None
                storage_reference_head_arr = None
                if current_picard_storage:
                    storage_state = _storage_from_picard_head(h_iter)
                    storage_coeff_inner = storage_state["storage"]
                    storage_sy_coeff_arr = np.asarray(storage_state["sy_coeff"], dtype=np.float64)
                    storage_ss_coeff_arr = np.asarray(storage_state["ss_coeff"], dtype=np.float64)
                    storage_reference_head_arr = np.asarray(storage_state["head_ref"], dtype=np.float64)
                else:
                    storage_coeff_inner = storage_coeff
                if storage_coeff_inner is not None:
                    storage_inner_arr = np.asarray(storage_coeff_inner, dtype=np.float64)
                    if storage_inner_arr.ndim == 0:
                        storage_inner_arr = np.full(shape0, float(storage_inner_arr.reshape(())), dtype=np.float64)
                    storage_inner_diag = storage_inner_arr * float(self.dx) * float(self.dx) / float(dt)
                    storage_inner_free = storage_inner_diag[free_mask0]
                    storage_coeff_free = storage_inner_arr[free_mask0]
                    storage_diag_min = float(np.min(storage_inner_free)) if storage_inner_free.size else None
                    storage_diag_max = float(np.max(storage_inner_free)) if storage_inner_free.size else None
                    storage_diag_mean = float(np.mean(storage_inner_free)) if storage_inner_free.size else None
                    storage_coeff_min = float(np.min(storage_coeff_free)) if storage_coeff_free.size else None
                    storage_coeff_max = float(np.max(storage_coeff_free)) if storage_coeff_free.size else None
                    storage_coeff_mean = float(np.mean(storage_coeff_free)) if storage_coeff_free.size else None
                    if previous_storage_diag_arr is None:
                        storage_diag_change_max = None
                        storage_diag_change_rms = None
                    else:
                        storage_diag_delta = storage_inner_diag - previous_storage_diag_arr
                        storage_diag_delta_free = storage_diag_delta[free_mask0]
                        if storage_diag_delta_free.size > 0:
                            storage_diag_change_max = float(np.max(np.abs(storage_diag_delta_free)))
                            storage_diag_change_rms = float(
                                np.sqrt(np.mean(storage_diag_delta_free * storage_diag_delta_free))
                            )
                        else:
                            storage_diag_change_max = 0.0
                            storage_diag_change_rms = 0.0
                else:
                    storage_diag_min = None
                    storage_diag_max = None
                    storage_diag_mean = None
                    storage_coeff_min = None
                    storage_coeff_max = None
                    storage_coeff_mean = None
                    storage_diag_change_max = None
                    storage_diag_change_rms = None

                head_lin, info_lin = self.solve_multigrid_kcycle(
                    max_cycles=int(inner_max_cycles),
                    initial_head=h_iter,
                    return_info=True,
                    unconfined=False,
                    transient=transient,
                    storage_coeff=storage_coeff_inner,
                    dt=dt,
                    head_prev=head_prev,
                    # Rebuild hierarchy when period-dependent storage changes;
                    # otherwise coarse MG levels can retain stale storage.
                    refresh_diag_with_transient_storage=True,
                    **kc_base_kwargs,
                )
                last_linear_info = dict(info_lin) if isinstance(info_lin, dict) else {}
                inner_converged = bool(last_linear_info.get("converged", False))

                h_lin = np.asarray(head_lin, dtype=np.float64)
                if h_lin.shape != shape0:
                    raise RuntimeError(f"inner linear solve returned shape {h_lin.shape}, expected {shape0}.")

                picard_update = h_lin - h_iter.astype(np.float64, copy=False)

                # Dynamic inexact inner tolerance based on the Picard update scale.
                if np.any(free_mask0):
                    picard_update_free_raw = picard_update[free_mask0]
                else:
                    picard_update_free_raw = np.array([], dtype=np.float64)
                if picard_update_free_raw.size > 0 and np.all(np.isfinite(picard_update_free_raw)):
                    picard_update_abs = np.abs(picard_update_free_raw)
                    picard_update_max = float(np.max(picard_update_abs))
                    picard_update_rms = float(np.sqrt(np.mean(picard_update_free_raw * picard_update_free_raw)))
                    picard_scale = max(
                        picard_update_rms,
                        inner_picard_scale_max_fraction_f * picard_update_max,
                    )
                    inner_head_residual_tol_used = min(
                        inner_head_residual_tol_max_f,
                        max(inner_head_residual_tol_min_f, inner_forcing_eta_f * picard_scale),
                    )
                    inner_usable_fallback = False
                else:
                    picard_update_max = 0.0
                    picard_update_rms = 0.0
                    picard_scale = 0.0
                    inner_head_residual_tol_used = float(inner_head_residual_tol_min_f)
                    inner_usable_fallback = True

                r_rms_end = _to_finite(last_linear_info.get("r_rms_end"))
                h_rms_end = _to_finite(last_linear_info.get("h_rms_end"))
                tol_abs_inner = _to_finite(last_linear_info.get("tol_abs"))
                dh_rms_lastcheck = _to_finite(last_linear_info.get("dh_rms_lastcheck"))
                inner_residual_converged = (
                    r_rms_end is not None and tol_abs_inner is not None and r_rms_end <= tol_abs_inner
                )
                inner_head_change_converged = (
                    dh_rms_lastcheck is not None and dh_rms_tol_f is not None and dh_rms_lastcheck <= dh_rms_tol_f
                )
                inner_practically_converged = (
                    inner_head_change_converged
                    and residual_floor_tol_f is not None
                    and r_rms_end is not None
                    and r_rms_end <= residual_floor_tol_f
                )
                inner_usable_for_picard = (
                    inner_converged
                    or inner_head_change_converged
                    or (
                        not inner_usable_fallback
                        and h_rms_end is not None
                        and np.isfinite(float(h_rms_end))
                        and float(h_rms_end) <= inner_head_residual_tol_used
                    )
                )

                if not inner_converged:
                    strict_inner_nonconvergence_count += 1

                picard_update[bc_mask0] = 0.0
                picard_update[~active_mask] = 0.0

                chebyshev_used = False
                chebyshev_rejected = False
                chebyshev_reset = False
                clipped_update = False

                if not inner_usable_for_picard:
                    unusable_inner_solve_count += 1
                    inner_solve_failures += 1
                    chebyshev_resets += 1
                    chebyshev_reset = True
                    previous_update.fill(0.0)
                else:
                    if not inner_converged:
                        practical_inner_acceptances += 1
                    accepted_picard_update_count += 1

                outer_chebyshev_ready = (
                    bool(chebyshev_enabled)
                    and accepted_picard_update_count >= 2
                    and len(cheb_weights) > 0
                    and inner_usable_for_picard
                )
                use_cheb = outer_chebyshev_ready
                if outer_chebyshev_ready:
                    outer_chebyshev_ready_count += 1
                if use_cheb:
                    weight = float(cheb_weights[(outer_idx - 1) % len(cheb_weights)])
                    alpha = min(max(omega_current * weight, omega_min_f), omega_max_f)
                    beta = 0.2 * max(0.0, alpha - omega_current)
                    proposed_update = alpha * picard_update + beta * previous_update
                    chebyshev_used = True
                else:
                    proposed_update = omega_current * picard_update

                clipped = np.clip(proposed_update, -max_update_f, max_update_f)
                clipped_update = bool(np.any(clipped != proposed_update))
                h_trial = h_iter.astype(np.float64, copy=False) + clipped
                h_trial[bc_mask0] = bc_values0[bc_mask0]
                h_trial[~active_mask] = 0.0

                if np.any(free_mask0):
                    trial_dh = (h_trial - h_iter.astype(np.float64, copy=False))[free_mask0]
                    trial_measure = float(np.max(np.abs(trial_dh)))
                    trial_rms = float(np.sqrt(np.mean(trial_dh * trial_dh)))
                else:
                    trial_measure = 0.0
                    trial_rms = 0.0

                reject_cheb = False
                if chebyshev_used:
                    if clipped_update or not np.all(np.isfinite(h_trial)):
                        reject_cheb = True
                    elif np.isfinite(previous_measure) and trial_measure > rejection_factor_f * previous_measure:
                        reject_cheb = True

                if reject_cheb:
                    chebyshev_rejected = True
                    chebyshev_used = False
                    chebyshev_rejections += 1
                    chebyshev_resets += 1
                    chebyshev_reset = True
                    previous_update.fill(0.0)
                    fallback_update = omega_current * picard_update
                    clipped = np.clip(fallback_update, -max_update_f, max_update_f)
                    clipped_update = bool(np.any(clipped != fallback_update))
                    h_trial = h_iter.astype(np.float64, copy=False) + clipped
                    h_trial[bc_mask0] = bc_values0[bc_mask0]
                    h_trial[~active_mask] = 0.0
                    if np.any(free_mask0):
                        trial_dh = (h_trial - h_iter.astype(np.float64, copy=False))[free_mask0]
                        trial_measure = float(np.max(np.abs(trial_dh)))
                        trial_rms = float(np.sqrt(np.mean(trial_dh * trial_dh)))
                    else:
                        trial_measure = 0.0
                        trial_rms = 0.0

                if not np.all(np.isfinite(h_trial)):
                    chebyshev_resets += 1
                    raise FloatingPointError("unconfined nonlinear update produced non-finite heads.")

                if np.isfinite(previous_measure) and trial_measure > rejection_factor_f * previous_measure:
                    omega_current = max(omega_min_f, 0.5 * omega_current)
                    improvement_streak = 0
                else:
                    improvement_streak += 1
                    if improvement_streak >= 3:
                        omega_current = min(omega_max_f, 1.1 * omega_current)
                        improvement_streak = 0

                previous_update[:, :] = clipped
                h_iter = h_trial.astype(NP_FLOAT, copy=False)
                final_max_abs_head_change = float(trial_measure)
                final_residual = last_linear_info.get("r_rms_end")
                final_h_rms_end = h_rms_end if h_rms_end is not None else float("nan")
                final_inner_max_cycles = int(inner_max_cycles)

                if clipped_update:
                    chebyshev_resets += 1
                    chebyshev_reset = True
                    previous_update.fill(0.0)

                if bool(chebyshev_reset_on_residual_increase) and np.isfinite(previous_measure):
                    if trial_measure > chebyshev_reset_factor_f * previous_measure:
                        chebyshev_resets += 1
                        chebyshev_reset = True
                        previous_update.fill(0.0)
                        minor_increase_count = 0
                    elif trial_measure > previous_measure:
                        minor_increase_count += 1
                        if minor_increase_count > chebyshev_minor_increase_patience_i:
                            chebyshev_resets += 1
                            chebyshev_reset = True
                            previous_update.fill(0.0)
                            minor_increase_count = 0
                    else:
                        minor_increase_count = 0

                previous_measure = trial_measure

                if chebyshev_used:
                    outer_chebyshev_used_count += 1
                if chebyshev_reset:
                    outer_chebyshev_reset_count += 1

                if storage_diag_change_max is not None:
                    max_storage_diag_change_max = max(max_storage_diag_change_max, float(storage_diag_change_max))
                if storage_diag_change_rms is not None:
                    max_storage_diag_change_rms = max(max_storage_diag_change_rms, float(storage_diag_change_rms))

                outer_history.append(
                    {
                        "outer_iteration": int(outer_idx + 1),
                        "inner_max_cycles_used": int(inner_max_cycles),
                        "inner_converged": bool(inner_converged),
                        "inner_head_change_converged": bool(inner_head_change_converged),
                        "inner_usable_for_picard": bool(inner_usable_for_picard),
                        "h_rms_end": float(h_rms_end) if h_rms_end is not None else None,
                        "inner_head_residual_tol_used": float(inner_head_residual_tol_used),
                        "picard_update_max": float(picard_update_max),
                        "picard_update_rms": float(picard_update_rms),
                        "picard_scale": float(picard_scale),
                        "accepted_picard_update_count": int(accepted_picard_update_count),
                        "omega": float(omega_current),
                        "chebyshev_used": bool(chebyshev_used),
                        "chebyshev_ready": bool(outer_chebyshev_ready),
                        "chebyshev_rejected": bool(chebyshev_rejected),
                        "chebyshev_reset": bool(chebyshev_reset),
                        "trial_measure": float(trial_measure),
                        "trial_rms": float(trial_rms),
                        "previous_measure": float(previous_measure) if np.isfinite(previous_measure) else None,
                        "clipped_update": bool(clipped_update),
                        "accepted_update": bool(inner_usable_for_picard),
                        "transmissivity_relaxation_used": None if np.isnan(T_relax) else float(T_relax),
                        "max_abs_head_change": float(final_max_abs_head_change),
                        "rms_head_change": float(trial_rms),
                        "min_head": float(np.nanmin(h_iter[active_mask])) if np.any(active_mask) else float("nan"),
                        "max_head": float(np.nanmax(h_iter[active_mask])) if np.any(active_mask) else float("nan"),
                        "min_saturated_thickness": float(np.nanmin(sat[active_mask])) if np.any(active_mask) else float("nan"),
                        "max_saturated_thickness": float(np.nanmax(sat[active_mask])) if np.any(active_mask) else float("nan"),
                        "mean_saturated_thickness": float(np.nanmean(sat[free_mask0])) if np.any(free_mask0) else float("nan"),
                        "min_transmissivity": float(np.nanmin(T_pic[active_mask])) if np.any(active_mask) else float("nan"),
                        "max_transmissivity": float(np.nanmax(T_pic[active_mask])) if np.any(active_mask) else float("nan"),
                        "storage_reference": str(storage_reference_mode),
                        "unconfined_storage_mode_2d": storage_mode_2d,
                        "storage_coeff_min": storage_coeff_min,
                        "storage_coeff_max": storage_coeff_max,
                        "storage_coeff_mean": storage_coeff_mean,
                        "storage_diag_min": storage_diag_min,
                        "storage_diag_max": storage_diag_max,
                        "storage_diag_mean": storage_diag_mean,
                        "storage_diag_change_max": storage_diag_change_max,
                        "storage_diag_change_rms": storage_diag_change_rms,
                        "inner_iterations": int(last_linear_info.get("n_cycles_used", 0)),
                        "inner_residual": None if final_residual is None else float(final_residual),
                    }
                )

                previous_storage_diag_arr = None if storage_coeff_inner is None else np.asarray(storage_inner_diag, dtype=np.float64).copy()
                if save_transient_diagnostics_b:
                    last_storage_coeff_array = (
                        None if storage_coeff_inner is None else np.asarray(storage_inner_arr, dtype=np.float64).copy()
                    )
                    last_sy_storage_coeff_array = (
                        None if storage_sy_coeff_arr is None else np.asarray(storage_sy_coeff_arr, dtype=np.float64).copy()
                    )
                    last_ss_storage_coeff_array = (
                        None if storage_ss_coeff_arr is None else np.asarray(storage_ss_coeff_arr, dtype=np.float64).copy()
                    )
                    last_storage_reference_head_array = (
                        None
                        if storage_reference_head_arr is None
                        else np.asarray(storage_reference_head_arr, dtype=np.float64).copy()
                    )

                head_change_converged = final_max_abs_head_change < hclose_f
                strict_picard_convergence_passed = bool(
                    head_change_converged and (inner_usable_for_picard or accept_on_head_change_only)
                )
                practical_picard_acceptance_passed = False
                if secant_sy_practical_mode:
                    practical_picard_acceptance_passed = bool(
                        int(outer_idx + 1) >= min_practical_outer_iterations_i
                        and final_residual is not None
                        and np.isfinite(float(final_residual))
                        and float(final_residual) <= practical_residual_tol_f
                        and np.isfinite(float(trial_rms))
                        and float(trial_rms) <= practical_dh_rms_tol_f
                        and storage_diag_change_rms is not None
                        and np.isfinite(float(storage_diag_change_rms))
                        and float(storage_diag_change_rms) <= practical_storage_diag_change_rms_tol_f
                    )
                production_acceptance_passed = bool(
                    strict_picard_convergence_passed or practical_picard_acceptance_passed
                )
                if outer_history:
                    outer_history[-1]["strict_picard_convergence_passed"] = bool(strict_picard_convergence_passed)
                    outer_history[-1]["practical_picard_acceptance_passed"] = bool(practical_picard_acceptance_passed)
                    outer_history[-1]["production_acceptance_passed"] = bool(production_acceptance_passed)
                # Diagnostic opt-in (default off): accept the Picard update on head
                # change alone, treating the inner-residual / inner_usable_for_picard
                # gate as a guardrail rather than a hard failure criterion. When False
                # this is identical to ``and inner_usable_for_picard``.
                if production_acceptance_passed:
                    break

            final_sat = h_iter.astype(np.float64, copy=False) - zbot_arr
            final_sat = np.maximum(final_sat, min_sat)
            if ztop_arr is not None:
                final_sat = np.minimum(final_sat, np.maximum(ztop_arr - zbot_arr, min_sat))
            final_T = (K_arr * final_sat).astype(NP_FLOAT, copy=False)
            final_T[~active_mask] = NP_FLOAT(0.0)
            self.update_T_in_place(final_T)

            effectively_dry = active_mask & (h_iter.astype(np.float64, copy=False) <= zbot_arr + float(dry_cell_flag_threshold))
            info_out = dict(last_linear_info) if isinstance(last_linear_info, dict) else {}
            info_out.update(
                    {
                        "solver_type": "kcycle_unconfined_picard_chebyshev",
                        "linear_solver_type": str(last_linear_info.get("solver_type", "kcycle")),
                        "unconfined": True,
                    "converged": bool(production_acceptance_passed),
                    "outer_iterations": int(len(outer_history)),
                    "chebyshev_enabled": bool(chebyshev_enabled),
                    "chebyshev_order": int(chebyshev_order),
                    "chebyshev_rejections": int(chebyshev_rejections),
                    "chebyshev_resets": int(chebyshev_resets),
                    "omega_final": float(omega_current),
                    "min_saturated_thickness": float(min_sat),
                    "max_head_change_per_outer_iteration": float(max_update_f),
                    "final_max_abs_head_change": float(final_max_abs_head_change),
                    "final_residual": None if final_residual is None else float(final_residual),
                    "inner_solve_failures": int(inner_solve_failures),
                    "strict_inner_nonconvergence_count": int(strict_inner_nonconvergence_count),
                    "unusable_inner_solve_count": int(unusable_inner_solve_count),
                    "practical_inner_acceptance_count": int(practical_inner_acceptances),
                    "accepted_picard_update_count": int(accepted_picard_update_count),
                    "outer_chebyshev_ready_count": int(outer_chebyshev_ready_count),
                    "outer_chebyshev_used_count": int(outer_chebyshev_used_count),
                    "outer_chebyshev_reset_count": int(outer_chebyshev_reset_count),
                    "effectively_dry_cell_count": int(np.count_nonzero(effectively_dry)),
                    "inner_forcing_eta": float(inner_forcing_eta_f),
                    "inner_head_residual_tol_min": float(inner_head_residual_tol_min_f),
                    "inner_head_residual_tol_max": float(inner_head_residual_tol_max_f),
                    "nonlinear_convergence_basis": (
                        "head_change_only"
                        if bool(accept_on_head_change_only)
                        else "head_change_and_inner_usable_for_picard"
                    ),
                    "accept_on_head_change_only": bool(accept_on_head_change_only),
                    "residual_floor_tol": None if residual_floor_tol_f is None else float(residual_floor_tol_f),
                    "inner_head_residual_tol": float(inner_head_residual_tol_f),
                    "inner_residual_converged": bool(inner_residual_converged),
                    "inner_head_change_converged": bool(inner_head_change_converged),
                    "inner_practically_converged": bool(inner_practically_converged),
                    "inner_usable_for_picard": bool(inner_usable_for_picard),
                    "inner_h_rms_end": float(final_h_rms_end) if np.isfinite(final_h_rms_end) else None,
                    "inner_max_cycles_used": int(final_inner_max_cycles),
                    "outer_history": outer_history,
                    "picard_converged": bool(strict_picard_convergence_passed),
                    "strict_picard_convergence_passed": bool(strict_picard_convergence_passed),
                    "practical_picard_acceptance_passed": bool(practical_picard_acceptance_passed),
                    "production_acceptance_passed": bool(production_acceptance_passed),
                    "practical_picard_acceptance_enabled": bool(secant_sy_practical_mode),
                    "min_practical_outer_iterations": int(min_practical_outer_iterations_i),
                    "practical_residual_tol": float(practical_residual_tol_f),
                    "practical_dh_rms_tol": float(practical_dh_rms_tol_f),
                    "practical_storage_diag_change_rms_tol": float(practical_storage_diag_change_rms_tol_f),
                    "picard_n_iter_used": int(len(outer_history)),
                    "picard_max_iter": int(max_outer),
                    "picard_relax": float(omega_current),
                    "picard_head_tol": float(hclose_f),
                    "picard_dh_max_end": float(final_max_abs_head_change),
                    "unconfined_min_sat": float(min_sat),
                    "unconfined_startup_mode": str(startup_mode),
                    "unconfined_pre_solve_iterations": int(unconfined_pre_solve_iterations_i),
                    "storage_reference": str(storage_reference_mode),
                    "unconfined_storage_mode_2d": storage_mode_2d,
                    "max_storage_diag_change_max": float(max_storage_diag_change_max),
                    "max_storage_diag_change_rms": float(max_storage_diag_change_rms),
                    "save_transient_diagnostics": bool(save_transient_diagnostics_b),
                    "diag_preconditioner_backend": self._diag_backend_env_or_default(),
                    "update_T_profile_last": None if self._last_update_T_profile is None else dict(self._last_update_T_profile),
                    "update_T_profile_totals": None if self._update_T_profile_totals is None else dict(self._update_T_profile_totals),
                }
            )
            if save_transient_diagnostics_b:
                info_out.update(
                    {
                        "storage_coeff_last_linearization_array": last_storage_coeff_array,
                        "sy_storage_coeff_last_linearization_array": last_sy_storage_coeff_array,
                        "ss_storage_coeff_last_linearization_array": last_ss_storage_coeff_array,
                        "storage_reference_head_last_linearization_array": last_storage_reference_head_array,
                    }
                )
            return (h_iter, info_out) if return_info else h_iter


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

    def solve_transient_2d_unconfined(
            self,
            *,
            initial_head: np.ndarray,
            recharge_rates: np.ndarray,
            k_field: np.ndarray,
            zbot_field: np.ndarray,
            ztop_field: np.ndarray,
            sy: float,
            ss: float,
            dt: float,
            active: np.ndarray | None = None,
            bc_mask: np.ndarray | None = None,
            bc_values: np.ndarray | None = None,
            storage_mode: str = "mf6_convertible_secant_sy",
            storage_reference: str = "current_picard",
            solve_controls: dict | None = None,
            min_saturated_thickness: float = 0.1,
            save_diagnostics: bool = False,
            return_info: bool = True,
    ):
        """
        Step a 2D unconfined transient solve through multiple stress periods.

        This is solver infrastructure only: callers remain responsible for MF6
        artifact loading, comparisons, reporting, mass balance, and persistence.

        :param initial_head: Initial/previous head for period 1.
        :param recharge_rates: One recharge value per stress period.
        :param k_field: Hydraulic conductivity field.
        :param zbot_field: Cell bottom field.
        :param ztop_field: Cell top field.
        :param sy: Specific yield.
        :param ss: Specific storage.
        :param dt: Transient time step.
        :param active: Optional active mask; defaults to all active.
        :param bc_mask: Optional Dirichlet mask; defaults to no Dirichlet cells.
        :param bc_values: Optional Dirichlet values; defaults to initial heads.
        :param storage_mode: 2D unconfined storage mode passed to the Picard solver.
        :param storage_reference: ``current_picard`` or ``previous_period``.
        :param solve_controls: Extra controls forwarded to :meth:`solve`.
        :param min_saturated_thickness: Minimum saturated thickness.
        :param save_diagnostics: Save full-grid storage/reference arrays.
        :param return_info: Return ``(heads_per_period, info)`` when true.
        :return: Heads per period, plus diagnostics when ``return_info`` is true.
        """
        h0 = np.asarray(initial_head, dtype=NP_FLOAT)
        if h0.shape != (self.ny, self.nx):
            raise ValueError(f"initial_head shape {h0.shape} expected {(self.ny, self.nx)}")
        k = np.asarray(k_field, dtype=NP_FLOAT)
        bottom = np.asarray(zbot_field, dtype=NP_FLOAT)
        top = np.asarray(ztop_field, dtype=NP_FLOAT)
        for name, arr in (("k_field", k), ("zbot_field", bottom), ("ztop_field", top)):
            if arr.shape != h0.shape:
                raise ValueError(f"{name} shape {arr.shape} expected {h0.shape}")

        if active is None:
            active_i = np.ones(h0.shape, dtype=np.int32)
        else:
            active_i = np.asarray(active, dtype=np.int32)
        if bc_mask is None:
            bc_i = np.zeros(h0.shape, dtype=np.int32)
        else:
            bc_i = np.asarray(bc_mask, dtype=np.int32)
        if bc_values is None:
            bc_v = np.asarray(h0, dtype=NP_FLOAT)
        else:
            bc_v = np.asarray(bc_values, dtype=NP_FLOAT)
        for name, arr in (("active", active_i), ("bc_mask", bc_i), ("bc_values", bc_v)):
            if arr.shape != h0.shape:
                raise ValueError(f"{name} shape {arr.shape} expected {h0.shape}")

        rates = np.asarray(recharge_rates, dtype=NP_FLOAT).reshape(-1)
        if rates.size < 1:
            raise ValueError("recharge_rates must contain at least one period.")
        dt_f = float(dt)
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            raise ValueError("dt must be finite and > 0.")

        controls = {} if solve_controls is None else dict(solve_controls)
        save_diagnostics_b = bool(controls.pop("save_transient_diagnostics", save_diagnostics))
        fast_path_controls = dict(controls)
        # Drop keys consumed by this wrapper or the device fast path so the
        # ``**controls`` spread forwarded to ``solve()`` does not inject unexpected
        # keywords.
        for _control_key in (
            "strict_head_residual_tol",
            "practical_head_residual_tol",
            "unconfined_inner_max_cycles_early",
            "unconfined_inner_max_cycles_middle",
            "unconfined_inner_max_cycles_late",
            "unconfined_inner_middle_dh",
            "unconfined_inner_late_dh",
            "adaptive_unconfined_inner_enabled",
            "adaptive_inner_initial_block_cycles",
            "adaptive_inner_min_block_cycles",
            "adaptive_inner_max_block_cycles",
            "adaptive_inner_min_total_cycles",
            "adaptive_inner_eta_initial",
            "adaptive_inner_eta_min",
            "adaptive_inner_eta_max",
            "adaptive_inner_eta_gamma",
            "adaptive_inner_eta_power",
            "adaptive_inner_good_contraction_ratio",
            "adaptive_inner_weak_contraction_ratio",
            "adaptive_inner_stall_contraction_ratio",
            "adaptive_inner_divergence_contraction_ratio",
            "adaptive_inner_stall_patience",
            "adaptive_inner_minimum_usable_reduction_ratio",
            "adaptive_inner_residual_floor",
            "adaptive_inner_relative_flow_residual_target",
            "adaptive_inner_save_block_history",
            "allow_unaccepted_transient_period",
            "use_device_transient_fast_path",
            "profile_transient_fast_path",
            "use_incremental_picard",
            "adaptive_dt_enabled",
            "adaptive_dt_min_fraction",
            "adaptive_dt_shrink_factor",
            "adaptive_dt_grow_factor",
            "adaptive_dt_strict_max_outer",
            "adaptive_dt_max_growth_steps",
            "adaptive_dt_early_shrink_enabled",
            "adaptive_dt_early_shrink_min_outer",
            "adaptive_dt_early_shrink_patience",
            "adaptive_dt_extension_enabled",
            "adaptive_dt_extension_factor",
            "adaptive_dt_extension_max_outer",
            "adaptive_dt_extension_contraction_ratio",
        ):
            controls.pop(_control_key, None)
        min_sat = float(controls.get("min_saturated_thickness", min_saturated_thickness))
        thickness = np.clip(h0 - bottom, min_sat, np.maximum(top - bottom, min_sat))
        initial_T = np.asarray(k * thickness, dtype=NP_FLOAT)
        initial_T[active_i == 0] = 0.0
        recharge_field = np.zeros(h0.shape, dtype=NP_FLOAT)
        self.build_from_fields(
            T_field=initial_T,
            R_field=recharge_field,
            active=active_i,
            bc_mask=bc_i,
            bc_values=bc_v,
        )

        n_periods = int(rates.size)
        heads_per_period = np.zeros((n_periods, self.ny, self.nx), dtype=np.float64)
        if save_diagnostics_b:
            heads_old_per_period = np.zeros_like(heads_per_period)
            storage_reference_heads = np.zeros_like(heads_per_period)
            storage_coeffs = np.zeros_like(heads_per_period)
            sy_coeffs = np.zeros_like(heads_per_period)
            ss_coeffs = np.zeros_like(heads_per_period)
            storage_terms = np.zeros_like(heads_per_period)
            sy_terms = np.zeros_like(heads_per_period)
            ss_terms = np.zeros_like(heads_per_period)
            sy_crossing_terms = np.zeros_like(heads_per_period)
        else:
            heads_old_per_period = None
            storage_reference_heads = None
            storage_coeffs = None
            sy_coeffs = None
            ss_coeffs = None
            storage_terms = None
            sy_terms = None
            ss_terms = None
            sy_crossing_terms = None
        period_infos: list[dict] = []
        period_times = np.zeros(n_periods, dtype=np.float64)
        counters = {
            "host_to_device_full_grid_copies": 0,
            "device_to_host_full_grid_copies": 0,
            "full_grid_allocations_inside_period_loop": 0,
            "full_grid_allocations_inside_outer_loop": 0,
            "hierarchy_rebuilds": 0,
            "hierarchy_rebuilds_inside_picard": 0,
            "hierarchy_device_coarse_value_refreshes": 0,
            "T_device_updates": 0,
            "storage_device_updates": 0,
            "R_device_updates": 0,
            "rhs_device_updates": 0,
            "scalar_reductions": 0,
            "gpu_scalar_synchronizations": 0,
            "head_downloads": 0,
            "full_head_downloads_inside_picard": 0,
            "host_T_builds_inside_picard": 0,
            "host_storage_builds_inside_picard": 0,
            "host_rhs_builds_inside_picard": 0,
            "host_to_device_T_uploads_inside_picard": 0,
            "host_to_device_storage_uploads_inside_picard": 0,
            "diagnostic_full_grid_arrays_saved": int(save_diagnostics_b),
            "device_side_picard_fast_path_active": 0,
        }
        if save_diagnostics_b:
            counters["full_grid_allocations_inside_period_loop"] += 9

        head_prev = np.asarray(h0, dtype=np.float64).copy()
        total_t0 = time.perf_counter()
        last_info: dict = {}

        use_device_fast_path = bool(fast_path_controls.get("use_device_transient_fast_path", False))
        use_incremental_picard = bool(fast_path_controls.get("use_incremental_picard", False))
        # Per-block h_iter = h^k + delta sync clip: large enough to be a no-op so
        # the residual check sees the true current head iterate (delta is a head
        # correction, bounded by the domain scale; the final relaxed update does
        # the real clipping via max_head_change_per_outer_iteration).
        incremental_picard_sync_max_change = 1.0e9
        fast_path = (
            use_device_fast_path
            and storage_mode == "mf6_convertible_secant_sy"
            and storage_reference == "current_picard"
        )
        if fast_path:
            controls = fast_path_controls
            if self.use_ghb:
                raise NotImplementedError("device transient fast path does not yet support GHB RHS assembly")
            counters["device_side_picard_fast_path_active"] = 1
            device = self.device_str
            n_free = int(np.count_nonzero((active_i != 0) & (bc_i == 0)))

            h_prev_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
            h_iter_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
            h_substep_start_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
            h_snapshot_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
            h_inner_snapshot_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
            bottom_wp = wp.array(bottom, dtype=WP_FLOAT, device=device)
            top_wp = wp.array(top, dtype=WP_FLOAT, device=device)
            k_field_wp = wp.array(k, dtype=WP_FLOAT, device=device)

            storage_diag_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            storage_diag_prev_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            storage_coeff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            sy_coeff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            ss_coeff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            rhs_eff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)

            # Incremental-Picard (correction) scratch buffers. ``delta_wp`` holds
            # the per-outer-iteration correction solved from ``A*delta = r^k``;
            # ``residual_wp`` holds the nonlinear residual field ``b - A*h^k``;
            # ``zero_bc_values_wp`` pins the correction to 0 on Dirichlet cells;
            # ``delta_snapshot_wp`` supports adaptive-block rollback. Allocated
            # unconditionally (cheap) and only touched when ``use_incremental_picard``.
            delta_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            residual_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            zero_bc_values_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
            delta_snapshot_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)

            storage_change_sum_sq_buf = wp.zeros(1, dtype=wp.float64, device=device)
            storage_change_max_buf = wp.zeros(1, dtype=wp.float64, device=device)
            dh_max_buf = wp.zeros(1, dtype=wp.float64, device=device)
            dh_rms_buf = wp.zeros(1, dtype=wp.float64, device=device)
            flow_rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
            head_rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
            rhs_rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
            converged_flag_buf = wp.zeros(1, dtype=wp.int32, device=device)
            head_nonfinite_flag_buf = wp.zeros(1, dtype=wp.int32, device=device)

            self.storage_diag_wp = storage_diag_wp
            self._storage_active = True

            if not hasattr(self, "storage_diag_host") or self.storage_diag_host is None:
                self.storage_diag_host = np.zeros_like(self.T_field_host)

            if self.mg_levels is None:
                self.build_hierarchy(
                    max_levels=int(controls.get("max_levels", 5)),
                    min_coarse_n=4,
                    min_coarse_cells=controls.get("min_coarse_cells", 500),
                )
            if self.mg_levels:
                self.mg_levels[0].T_wp = self.T_wp
                self.mg_levels[0].storage_diag_wp = storage_diag_wp
            fast_path_coarse_operator_mode = "device_refreshed_dynamic_coarse_operator"

            max_outer = int(controls.get("unconfined_max_picard_iter", controls.get("max_outer_iterations", 100)))
            max_cycles_hard_i = int(controls.get("max_cycles", 200))
            hclose = float(controls.get("unconfined_head_tol", controls.get("hclose", 1.0e-4)))
            strict_head_residual_tol_f = float(controls.get("strict_head_residual_tol", hclose))
            min_practical_outer_iterations_i = int(controls.get("min_practical_outer_iterations", 20))
            practical_head_residual_tol_f = float(
                controls.get("practical_head_residual_tol", controls.get("practical_residual_tol", 1.0e-4))
            )
            practical_residual_tol_alias_used = "practical_head_residual_tol" not in controls and "practical_residual_tol" in controls
            practical_dh_rms_tol_f = float(controls.get("practical_dh_rms_tol", 3.0e-3))
            practical_storage_diag_change_rms_tol_f = float(
                controls.get("practical_storage_diag_change_rms_tol", 30.0)
            )
            practical_picard_acceptance_enabled_b = bool(
                controls.get("practical_picard_acceptance_enabled", False)
            )
            omega_current_f = float(controls.get("unconfined_relax", controls.get("omega", 0.8)))
            omega_min_f = float(controls.get("omega_min", 0.05))
            omega_max_f = float(controls.get("omega_max", 0.75))
            if not (0.0 < omega_min_f <= omega_max_f):
                raise ValueError("omega_min and omega_max must satisfy 0 < omega_min <= omega_max.")
            omega_current_f = min(max(omega_current_f, omega_min_f), omega_max_f)
            max_update_f = float(controls.get("max_head_change_per_outer_iteration", 5.0))
            if max_update_f <= 0.0 or not np.isfinite(max_update_f):
                raise ValueError("max_head_change_per_outer_iteration must be positive and finite.")
            inner_max_cycles_early_i = int(controls.get("unconfined_inner_max_cycles_early", 10))
            inner_max_cycles_middle_i = int(controls.get("unconfined_inner_max_cycles_middle", 25))
            inner_max_cycles_late_i = int(controls.get("unconfined_inner_max_cycles_late", 60))
            inner_middle_dh_f = float(controls.get("unconfined_inner_middle_dh", 1.0))
            inner_late_dh_f = float(controls.get("unconfined_inner_late_dh", 1.0e-2))
            inner_head_residual_tol_min_f = float(
                controls.get("inner_head_residual_tol_min", controls.get("inner_head_residual_tol", hclose))
            )
            inner_head_residual_tol_max_f = float(controls.get("inner_head_residual_tol_max", 1.0e-2))
            inner_picard_scale_max_fraction_f = float(controls.get("inner_picard_scale_max_fraction", 0.10))
            allow_unaccepted_transient_period_b = bool(controls.get("allow_unaccepted_transient_period", False))
            startup_mode = str(controls.get("unconfined_startup_mode", "initial_head")).strip().lower()
            if startup_mode not in {"initial_head", "confined_pre_solve"}:
                raise ValueError("device transient fast path supports startup modes 'initial_head' and 'confined_pre_solve'.")
            profile_fast_path_b = bool(controls.get("profile_transient_fast_path", False))
            adaptive_inner_config = _build_adaptive_inner_solve_config_from_controls(
                controls=controls,
                max_cycles=max_cycles_hard_i,
            )
            min_sat_f = float(min_sat)
            sy_f = float(sy)
            ss_f = float(ss)
            dx_f = float(self.dx)
            dt_f_val = float(dt_f)
            adaptive_dt_enabled_b = bool(controls.get("adaptive_dt_enabled", False))
            adaptive_dt_min_fraction_f = float(controls.get("adaptive_dt_min_fraction", 0.0625))
            adaptive_dt_shrink_factor_f = float(controls.get("adaptive_dt_shrink_factor", 0.5))
            adaptive_dt_grow_factor_f = float(controls.get("adaptive_dt_grow_factor", 2.0))
            adaptive_dt_strict_max_outer_i = int(controls.get("adaptive_dt_strict_max_outer", 20))
            adaptive_dt_max_growth_steps_i = int(controls.get("adaptive_dt_max_growth_steps", 2))
            adaptive_dt_early_shrink_enabled_b = bool(controls.get("adaptive_dt_early_shrink_enabled", True))
            adaptive_dt_early_shrink_min_outer_i = int(controls.get("adaptive_dt_early_shrink_min_outer", 6))
            adaptive_dt_early_shrink_patience_i = int(controls.get("adaptive_dt_early_shrink_patience", 3))
            adaptive_dt_extension_enabled_b = bool(controls.get("adaptive_dt_extension_enabled", True))
            adaptive_dt_extension_factor_f = float(controls.get("adaptive_dt_extension_factor", 5.0))
            adaptive_dt_extension_max_outer_i = int(controls.get("adaptive_dt_extension_max_outer", 4))
            adaptive_dt_extension_contraction_ratio_f = float(
                controls.get("adaptive_dt_extension_contraction_ratio", 0.8)
            )
            if adaptive_dt_enabled_b:
                if not (0.0 < adaptive_dt_min_fraction_f <= 1.0):
                    raise ValueError("adaptive_dt_min_fraction must be in (0, 1].")
                if not (0.0 < adaptive_dt_shrink_factor_f < 1.0):
                    raise ValueError("adaptive_dt_shrink_factor must be in (0, 1).")
                if adaptive_dt_grow_factor_f < 1.0:
                    raise ValueError("adaptive_dt_grow_factor must be >= 1.")
                if adaptive_dt_strict_max_outer_i < 1:
                    raise ValueError("adaptive_dt_strict_max_outer must be >= 1.")
                if adaptive_dt_strict_max_outer_i > max_outer:
                    raise ValueError(
                        "adaptive_dt_strict_max_outer must be <= unconfined_max_picard_iter/max_outer."
                    )
                if adaptive_dt_max_growth_steps_i < 0:
                    raise ValueError("adaptive_dt_max_growth_steps must be >= 0.")
                if adaptive_dt_early_shrink_min_outer_i < 1:
                    raise ValueError("adaptive_dt_early_shrink_min_outer must be >= 1.")
                if adaptive_dt_early_shrink_patience_i < 1:
                    raise ValueError("adaptive_dt_early_shrink_patience must be >= 1.")
                if adaptive_dt_extension_factor_f < 1.0:
                    raise ValueError("adaptive_dt_extension_factor must be >= 1.")
                if adaptive_dt_extension_max_outer_i < 1:
                    raise ValueError("adaptive_dt_extension_max_outer must be >= 1.")
                if not (0.0 < adaptive_dt_extension_contraction_ratio_f < 1.0):
                    raise ValueError("adaptive_dt_extension_contraction_ratio must be in (0, 1).")

            if inner_head_residual_tol_min_f < 0.0 or not np.isfinite(inner_head_residual_tol_min_f):
                raise ValueError("inner_head_residual_tol_min must be non-negative and finite.")
            if inner_head_residual_tol_max_f < inner_head_residual_tol_min_f:
                raise ValueError("inner_head_residual_tol_max must be >= inner_head_residual_tol_min.")
            if inner_picard_scale_max_fraction_f < 0.0 or inner_picard_scale_max_fraction_f > 1.0:
                raise ValueError("inner_picard_scale_max_fraction must be in [0, 1].")

            dim2d = (self.ny, self.nx)

            def _fast_path_phase_start() -> float:
                if profile_fast_path_b and str(device).startswith("cuda"):
                    wp.synchronize_device(device)
                return time.perf_counter()

            def _fast_path_phase_elapsed(t_start: float) -> float:
                if profile_fast_path_b and str(device).startswith("cuda"):
                    wp.synchronize_device(device)
                return float(time.perf_counter() - t_start)

            def _fast_path_head_residual_check() -> tuple[float, float, float, bool]:
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[flow_rTr_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[head_rTr_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rhs_rTr_buf], device=device)
                wp.launch(kernel=zero_int_scalar_kernel, dim=1, inputs=[head_nonfinite_flag_buf], device=device)
                wp.launch(
                    kernel=compute_dual_residual_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp,
                        rhs_eff_wp,
                        self.T_wp,
                        self.active_wp,
                        self.bc_mask_wp,
                        self.mg_levels[0].gh_mask_wp,
                        self.mg_levels[0].ghb_factor_wp,
                        storage_diag_wp,
                        flow_rTr_buf,
                        head_rTr_buf,
                        self.nx,
                        self.ny,
                    ],
                    device=device,
                )
                wp.launch(
                    kernel=detect_nonfinite_field_kernel,
                    dim=dim2d,
                    inputs=[h_iter_wp, head_nonfinite_flag_buf, self.nx, self.ny],
                    device=device,
                )
                wp.launch(
                    kernel=compute_active_rhs_l2_kernel,
                    dim=dim2d,
                    inputs=[rhs_eff_wp, self.active_wp, self.bc_mask_wp, rhs_rTr_buf, self.nx, self.ny],
                    device=device,
                )
                counters["scalar_reductions"] += 1
                head_rtr = float(head_rTr_buf.numpy()[0])
                flow_rtr = float(flow_rTr_buf.numpy()[0])
                head_nonfinite = bool(int(head_nonfinite_flag_buf.numpy()[0]) != 0)
                head_rms = float(np.sqrt(max(head_rtr, 0.0) / float(max(n_free, 1))))
                flow_rms = float(np.sqrt(max(flow_rtr, 0.0) / float(max(n_free, 1))))
                rhs_rms = float(np.sqrt(max(float(rhs_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1))))
                relative_flow_rms = flow_rms / max(rhs_rms, float(adaptive_inner_config.residual_floor))
                return head_rms, flow_rms, relative_flow_rms, head_nonfinite

            def evaluate_refreshed_nonlinear_candidate(
                *,
                outer_iteration: int,
                info_lin: dict[str, Any],
                dh_max: float,
                dh_rms: float,
                substep_dt: float,
                require_strict: bool = False,
            ) -> dict[str, Any]:
                """Refresh the nonlinear operator and evaluate authoritative acceptance."""
                wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[storage_diag_wp, storage_diag_prev_wp, self.nx, self.ny], device=device)
                wp.launch(
                    kernel=update_unconfined_transmissivity_from_head_kernel,
                    dim=dim2d,
                    inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                    device=device,
                )
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)
                wp.launch(
                    kernel=update_secant_sy_storage_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                        sy_f, ss_f, dx_f, substep_dt, min_sat_f, 1.0e-12, self.nx, self.ny,
                        storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                        storage_change_sum_sq_buf, storage_change_max_buf,
                    ],
                    device=device,
                )
                wp.launch(
                    kernel=build_transient_rhs_from_storage_kernel,
                    dim=dim2d,
                    inputs=[self.R_wp, storage_diag_wp, h_prev_wp, self.active_wp, self.bc_mask_wp, self.bc_values_wp, dx_f, self.nx, self.ny, rhs_eff_wp],
                    device=device,
                )
                head_rms, flow_rms, relative_flow_rms, _ = _fast_path_head_residual_check()
                storage_change_rms = float(
                    np.sqrt(max(float(storage_change_sum_sq_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                storage_change_max = float(storage_change_max_buf.numpy()[0])
                adaptive_used = bool(
                    adaptive_inner_config.enabled and info_lin.get("adaptive_inner_controller_used", False)
                )
                inner_solved = _adaptive_practical_acceptance_allowed(
                    practical_acceptance_enabled=True,
                    adaptive_controller_used=adaptive_used,
                    inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                )
                strict = bool(
                    inner_solved and dh_max <= hclose and head_rms <= strict_head_residual_tol_f
                )
                practical = bool(
                    (not require_strict)
                    and _adaptive_practical_acceptance_allowed(
                        practical_acceptance_enabled=practical_picard_acceptance_enabled_b,
                        adaptive_controller_used=adaptive_used,
                        inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                    )
                    and int(outer_iteration) >= min_practical_outer_iterations_i
                    and np.isfinite(head_rms) and head_rms <= practical_head_residual_tol_f
                    and np.isfinite(dh_rms) and dh_rms <= practical_dh_rms_tol_f
                    and np.isfinite(storage_change_rms)
                    and storage_change_rms <= practical_storage_diag_change_rms_tol_f
                )
                return {
                    "head_residual_rms": float(head_rms),
                    "flow_residual_rms": float(flow_rms),
                    "relative_flow_residual_rms": float(relative_flow_rms),
                    "storage_diag_change_max": storage_change_max,
                    "storage_diag_change_rms": storage_change_rms,
                    "strict_acceptance_passed": strict,
                    "practical_acceptance_passed": practical,
                    "production_acceptance_passed": bool(strict or practical),
                }

            for period_index in range(n_periods):
                self.update_uniform_recharge_in_place(float(rates[period_index]))
                counters["R_device_updates"] += 1
                wp.launch(
                    kernel=copy_field_kernel,
                    dim=dim2d,
                    inputs=[h_prev_wp, h_substep_start_wp, self.nx, self.ny],
                    device=device,
                )
                if save_diagnostics_b:
                    period_head_old = np.asarray(head_prev, dtype=np.float64).copy()

                period_t0 = time.perf_counter()
                T_update_seconds = 0.0
                storage_kernel_seconds = 0.0
                fine_m_inv_refresh_seconds = 0.0
                dynamic_coarse_refresh_seconds = 0.0
                rhs_assembly_seconds = 0.0
                storage_assembly_seconds = 0.0
                inner_solver_seconds = 0.0
                outer_convergence_check_seconds = 0.0
                final_nonlinear_residual_check_seconds = 0.0
                head_download_seconds = 0.0
                startup_inner_cycles = 0
                startup_converged = None

                if startup_mode == "confined_pre_solve":
                    startup_t0 = _fast_path_phase_start()
                    phase_t0 = _fast_path_phase_start()
                    wp.launch(
                        kernel=update_unconfined_transmissivity_from_head_kernel,
                        dim=dim2d,
                        inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                        device=device,
                    )
                    T_update_seconds += _fast_path_phase_elapsed(phase_t0)
                    counters["T_device_updates"] += 1
                    phase_t0 = _fast_path_phase_start()
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)
                    wp.launch(
                        kernel=update_secant_sy_storage_kernel,
                        dim=dim2d,
                        inputs=[
                            h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                            sy_f, ss_f, dx_f, dt_f_val, min_sat_f, 1.0e-12, self.nx, self.ny,
                            storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                            storage_change_sum_sq_buf, storage_change_max_buf,
                        ],
                        device=device,
                    )
                    storage_kernel_seconds += _fast_path_phase_elapsed(phase_t0)
                    counters["storage_device_updates"] += 1
                    if hasattr(self, "_update_diag_preconditioner_device"):
                        phase_t0 = _fast_path_phase_start()
                        self._update_diag_preconditioner_device(
                            T_wp=self.T_wp,
                            active_wp=self.active_wp,
                            bc_mask_wp=self.bc_mask_wp,
                            gh_mask_wp=self.mg_levels[0].gh_mask_wp,
                            ghb_factor_wp=self.mg_levels[0].ghb_factor_wp,
                            M_inv_wp=self.mg_levels[0].M_inv_wp,
                            nx=self.nx,
                            ny=self.ny,
                            use_ghb=bool(self.use_ghb),
                            storage_diag_wp=storage_diag_wp,
                        )
                        fine_m_inv_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                    phase_t0 = _fast_path_phase_start()
                    self._refresh_transient_device_hierarchy_values(levels=self.mg_levels)
                    dynamic_coarse_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                    counters["hierarchy_device_coarse_value_refreshes"] += 1
                    counters["rhs_device_updates"] += 1
                    phase_t0 = _fast_path_phase_start()
                    wp.launch(
                        kernel=build_transient_rhs_from_storage_kernel,
                        dim=dim2d,
                        inputs=[
                            self.R_wp,
                            storage_diag_wp,
                            h_prev_wp,
                            self.active_wp,
                            self.bc_mask_wp,
                            self.bc_values_wp,
                            dx_f,
                            self.nx,
                            self.ny,
                            rhs_eff_wp,
                        ],
                        device=device,
                    )
                    rhs_assembly_seconds += _fast_path_phase_elapsed(phase_t0)
                    startup_controls = dict(controls)
                    startup_controls["max_cycles"] = int(controls.get("max_cycles", 200))
                    startup_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                    phase_t0 = _fast_path_phase_start()
                    startup_info = self._solve_multigrid_kcycle_device_buffers(
                        x_wp=h_iter_wp,
                        rhs_wp=rhs_eff_wp,
                        T_wp=self.T_wp,
                        storage_diag_wp=storage_diag_wp,
                        active_wp=self.active_wp,
                        bc_mask_wp=self.bc_mask_wp,
                        bc_values_wp=self.bc_values_wp,
                        levels=self.mg_levels,
                        solve_controls=startup_controls,
                        return_scalar_info=True,
                    )
                    inner_solver_seconds += _fast_path_phase_elapsed(phase_t0)
                    startup_inner_cycles = int(startup_info.get("n_cycles_used", 0) or 0)
                    startup_converged = bool(startup_info.get("converged", False))
                    phase_t0 = _fast_path_phase_start()
                    wp.launch(
                        kernel=clamp_unconfined_head_kernel,
                        dim=dim2d,
                        inputs=[
                            h_iter_wp,
                            bottom_wp,
                            top_wp,
                            self.active_wp,
                            self.bc_mask_wp,
                            self.bc_values_wp,
                            min_sat_f,
                            self.nx,
                            self.ny,
                        ],
                        device=device,
                    )
                    storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                    inner_solver_seconds += _fast_path_phase_elapsed(phase_t0)
                last_dh_max = float("nan")
                last_dh_rms = float("nan")
                last_flow_residual_rms = float("nan")
                last_head_residual_rms = float("nan")
                last_storage_diag_change_max = float("nan")
                last_storage_diag_change_rms = float("nan")
                strict_picard_convergence_passed = False
                practical_picard_acceptance_passed = False
                production_acceptance_passed = False
                previous_dh_measure = None
                previous_outer_head_residual_rms = None
                previous_initial_head_residual_rms = None
                previous_outer_dh_rms = None
                total_inner_kcycles = 0
                maximum_inner_kcycles_in_one_outer_iteration = 0
                inner_kcycle_caps: list[int] = []
                inner_kcycle_used: list[int] = []
                inner_block_counts: list[int] = []
                inner_residual_check_count = 0
                adaptive_target_achievement_count = 0
                legacy_dh_fallback_count = 0
                stalled_inner_solve_count = 0
                divergent_inner_solve_count = 0
                rolled_back_block_count = 0
                outer_iteration_summaries: list[dict[str, Any]] = []
                period_gpu_scalar_syncs = 0
                info_lin = {
                    "converged": False,
                    "coarse_operator_mode": fast_path_coarse_operator_mode,
                    "fine_operator_residual_checked": True,
                }

                period_dt_f = dt_f_val
                remaining_dt_f = period_dt_f
                current_dt_f = period_dt_f
                actual_dt_f = period_dt_f
                dt_min_f = period_dt_f * adaptive_dt_min_fraction_f
                adaptive_dt_growth_steps_i = 0
                adaptive_dt_substep_dts: list[float] = []
                adaptive_dt_retry_count = 0
                adaptive_dt_practical_fallback_count = 0
                adaptive_dt_total_outer_iterations_i = 0
                adaptive_dt_practical_at_min_b = not adaptive_dt_enabled_b
                adaptive_dt_dh_history: list[float] = []
                adaptive_dt_extension_used_b = False
                adaptive_dt_early_shrink_streak_i = 0
                adaptive_dt_early_shrink_count = 0
                adaptive_dt_extension_count = 0
                substep_outer_limit_i = (
                    adaptive_dt_strict_max_outer_i if adaptive_dt_enabled_b else max_outer
                )
                outer_iter = 0
                while outer_iter < substep_outer_limit_i:
                    adaptive_dt_total_outer_iterations_i += 1
                    storage_t0 = _fast_path_phase_start()
                    phase_t0 = _fast_path_phase_start()
                    wp.launch(
                        kernel=update_unconfined_transmissivity_from_head_kernel,
                        dim=dim2d,
                        inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                        device=device
                    )
                    T_update_seconds += _fast_path_phase_elapsed(phase_t0)
                    counters["T_device_updates"] += 1

                    phase_t0 = _fast_path_phase_start()
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)

                    wp.launch(
                        kernel=update_secant_sy_storage_kernel,
                        dim=dim2d,
                        inputs=[
                            h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                            sy_f, ss_f, dx_f, actual_dt_f, min_sat_f, 1.0e-12, self.nx, self.ny,
                            storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                            storage_change_sum_sq_buf, storage_change_max_buf
                        ],
                        device=device
                    )
                    wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[storage_diag_wp, storage_diag_prev_wp, self.nx, self.ny], device=device)
                    storage_kernel_seconds += _fast_path_phase_elapsed(phase_t0)
                    counters["storage_device_updates"] += 1
                    storage_assembly_seconds += _fast_path_phase_elapsed(storage_t0)

                    if hasattr(self, "_update_diag_preconditioner_device"):
                        phase_t0 = _fast_path_phase_start()
                        self._update_diag_preconditioner_device(
                            T_wp=self.T_wp,
                            active_wp=self.active_wp,
                            bc_mask_wp=self.bc_mask_wp,
                            gh_mask_wp=self.mg_levels[0].gh_mask_wp,
                            ghb_factor_wp=self.mg_levels[0].ghb_factor_wp,
                            M_inv_wp=self.mg_levels[0].M_inv_wp,
                            nx=self.nx,
                            ny=self.ny,
                            use_ghb=bool(self.use_ghb),
                            storage_diag_wp=storage_diag_wp
                        )
                        fine_m_inv_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                    phase_t0 = _fast_path_phase_start()
                    self._refresh_transient_device_hierarchy_values(levels=self.mg_levels)
                    dynamic_coarse_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                    counters["hierarchy_device_coarse_value_refreshes"] += 1

                    rhs_t0 = _fast_path_phase_start()
                    counters["rhs_device_updates"] += 1
                    wp.launch(
                        kernel=build_transient_rhs_from_storage_kernel,
                        dim=dim2d,
                        inputs=[
                            self.R_wp,
                            storage_diag_wp,
                            h_prev_wp,
                            self.active_wp,
                            self.bc_mask_wp,
                            self.bc_values_wp,
                            dx_f,
                            self.nx,
                            self.ny,
                            rhs_eff_wp,
                        ],
                        device=device
                    )
                    rhs_assembly_seconds += _fast_path_phase_elapsed(rhs_t0)

                    wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[h_iter_wp, h_snapshot_wp, self.nx, self.ny], device=device)
                    # Incremental Picard: materialise the nonlinear residual field
                    # r^k = b - A*h^k (h_snapshot == h^k here) and reset the correction
                    # to zero, so the inner solve targets A*delta = r^k with delta=0 on
                    # Dirichlet cells. rhs_rTr_buf is reused as an unread scratch for the
                    # kernel's rTr reduction; it is re-zeroed before any later read.
                    if use_incremental_picard:
                        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rhs_rTr_buf], device=device)
                        wp.launch(
                            kernel=compute_residual_kernel,
                            dim=dim2d,
                            inputs=[
                                h_snapshot_wp,
                                rhs_eff_wp,
                                self.T_wp,
                                self.active_wp,
                                self.bc_mask_wp,
                                self.mg_levels[0].gh_mask_wp,
                                self.mg_levels[0].ghb_factor_wp,
                                storage_diag_wp,
                                residual_wp,
                                rhs_rTr_buf,
                                self.nx,
                                self.ny,
                            ],
                            device=device,
                        )
                        wp.launch(kernel=zero_field_kernel, dim=dim2d, inputs=[delta_wp, self.nx, self.ny], device=device)
                    adaptive_fallback_reason = ""
                    adaptive_controller_used = bool(adaptive_inner_config.enabled)
                    legacy_dh_fallback_used = False
                    forcing_eta_used = float("nan")
                    inner_initial_head_residual_rms = float("nan")
                    inner_initial_flow_residual_rms = float("nan")
                    inner_initial_relative_flow_residual_rms = float("nan")
                    inner_target_head_residual_rms = float("nan")
                    inner_target_relative_flow_residual_rms = float("nan")
                    inner_final_head_residual_rms = float("nan")
                    adaptive_state: AdaptiveInnerSolveState | None = None
                    adaptive_pre_fallback_cycles = 0
                    adaptive_pre_fallback_blocks = 0

                    if adaptive_inner_config.enabled:
                        residual_check_t0 = _fast_path_phase_start()
                        (
                            initial_head_residual_rms,
                            initial_flow_residual_rms,
                            initial_relative_flow_residual_rms,
                            initial_head_nonfinite,
                        ) = _fast_path_head_residual_check()
                        inner_solver_seconds += _fast_path_phase_elapsed(residual_check_t0)
                        inner_residual_check_count += 1
                        inner_initial_head_residual_rms = float(initial_head_residual_rms)
                        inner_initial_flow_residual_rms = float(initial_flow_residual_rms)
                        inner_initial_relative_flow_residual_rms = float(
                            initial_relative_flow_residual_rms
                        )

                        if initial_head_nonfinite or not np.isfinite(initial_head_residual_rms):
                            adaptive_fallback_reason = "nonfinite_initial_head_residual"
                            adaptive_controller_used = False
                            legacy_dh_fallback_used = True
                        else:
                            forcing_eta_used = _compute_inner_forcing_eta(
                                current_outer_residual_rms=initial_head_residual_rms,
                                previous_outer_residual_rms=previous_initial_head_residual_rms,
                                config=adaptive_inner_config,
                            )
                            inner_target_head_residual_rms = _compute_inner_target_residual(
                                initial_residual_rms=initial_head_residual_rms,
                                forcing_eta=forcing_eta_used,
                                residual_floor=float(adaptive_inner_config.residual_floor),
                                inner_head_residual_tol_min=inner_head_residual_tol_min_f,
                                inner_head_residual_tol_max=inner_head_residual_tol_max_f,
                                inner_picard_scale_max_fraction=inner_picard_scale_max_fraction_f,
                                previous_outer_dh_rms=previous_outer_dh_rms,
                                hclose=hclose,
                            )
                            inner_target_relative_flow_residual_rms = max(
                                float(adaptive_inner_config.residual_floor),
                                float(forcing_eta_used) * float(initial_flow_residual_rms),
                            )

                            def _run_adaptive_block(block_cycles: int) -> dict[str, Any]:
                                nonlocal inner_solver_seconds
                                if use_incremental_picard:
                                    # Snapshot the running correction so a divergent
                                    # block can be rolled back; continue delta in place.
                                    wp.launch(
                                        kernel=copy_field_kernel,
                                        dim=dim2d,
                                        inputs=[delta_wp, delta_snapshot_wp, self.nx, self.ny],
                                        device=device,
                                    )
                                    block_controls = dict(controls)
                                    block_controls["max_cycles"] = int(block_cycles)
                                    block_controls["check_every_no"] = int(block_cycles)
                                    block_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                                    block_t0 = _fast_path_phase_start()
                                    block_info = self._solve_multigrid_kcycle_device_buffers(
                                        x_wp=delta_wp,
                                        rhs_wp=residual_wp,
                                        T_wp=self.T_wp,
                                        storage_diag_wp=storage_diag_wp,
                                        active_wp=self.active_wp,
                                        bc_mask_wp=self.bc_mask_wp,
                                        bc_values_wp=zero_bc_values_wp,
                                        levels=self.mg_levels,
                                        solve_controls=block_controls,
                                        return_scalar_info=False,
                                    )
                                    inner_solver_seconds += _fast_path_phase_elapsed(block_t0)
                                    # Sync h_iter = h^k + delta so the (unchanged) residual
                                    # check measures ||b - A*(h^k + delta)|| = ||r^k - A*delta||.
                                    wp.launch(
                                        kernel=apply_relaxed_correction_kernel,
                                        dim=dim2d,
                                        inputs=[
                                            h_snapshot_wp,
                                            delta_wp,
                                            self.active_wp,
                                            self.bc_mask_wp,
                                            self.bc_values_wp,
                                            WP_FLOAT(1.0),
                                            WP_FLOAT(incremental_picard_sync_max_change),
                                            self.nx,
                                            self.ny,
                                            h_iter_wp,
                                        ],
                                        device=device,
                                    )
                                    residual_t0 = _fast_path_phase_start()
                                    head_rms_after, flow_rms_after, _, head_nonfinite_after = _fast_path_head_residual_check()
                                    inner_solver_seconds += _fast_path_phase_elapsed(residual_t0)
                                    return {
                                        "actual_cycles": int(
                                            block_info["n_cycles_used"]
                                            if block_info.get("n_cycles_used") is not None else block_cycles
                                        ),
                                        "residual_after_rms": float(head_rms_after),
                                        "relative_flow_residual_rms": float(flow_rms_after),
                                        "rollback_required": bool(
                                            head_nonfinite_after or (not np.isfinite(head_rms_after))
                                        ),
                                        "head_nonfinite": bool(head_nonfinite_after),
                                        "numerical_breakdown": False,
                                    }
                                wp.launch(
                                    kernel=copy_field_kernel,
                                    dim=dim2d,
                                    inputs=[h_iter_wp, h_inner_snapshot_wp, self.nx, self.ny],
                                    device=device,
                                )
                                block_controls = dict(controls)
                                block_controls["max_cycles"] = int(block_cycles)
                                block_controls["check_every_no"] = int(block_cycles)
                                block_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                                block_t0 = _fast_path_phase_start()
                                block_info = self._solve_multigrid_kcycle_device_buffers(
                                    x_wp=h_iter_wp,
                                    rhs_wp=rhs_eff_wp,
                                    T_wp=self.T_wp,
                                    storage_diag_wp=storage_diag_wp,
                                    active_wp=self.active_wp,
                                    bc_mask_wp=self.bc_mask_wp,
                                    bc_values_wp=self.bc_values_wp,
                                    levels=self.mg_levels,
                                    solve_controls=block_controls,
                                    return_scalar_info=False,
                                )
                                inner_solver_seconds += _fast_path_phase_elapsed(block_t0)
                                residual_t0 = _fast_path_phase_start()
                                head_rms_after, flow_rms_after, _, head_nonfinite_after = _fast_path_head_residual_check()
                                inner_solver_seconds += _fast_path_phase_elapsed(residual_t0)
                                return {
                                    "actual_cycles": int(
                                        block_info["n_cycles_used"]
                                        if block_info.get("n_cycles_used") is not None else block_cycles
                                    ),
                                    "residual_after_rms": float(head_rms_after),
                                    "relative_flow_residual_rms": float(flow_rms_after),
                                    "rollback_required": bool(
                                        head_nonfinite_after or (not np.isfinite(head_rms_after))
                                    ),
                                    "head_nonfinite": bool(head_nonfinite_after),
                                    "numerical_breakdown": False,
                                }

                            def _rollback_adaptive_block() -> None:
                                if use_incremental_picard:
                                    wp.launch(
                                        kernel=copy_field_kernel,
                                        dim=dim2d,
                                        inputs=[delta_snapshot_wp, delta_wp, self.nx, self.ny],
                                        device=device,
                                    )
                                    return
                                wp.launch(
                                    kernel=copy_field_kernel,
                                    dim=dim2d,
                                    inputs=[h_inner_snapshot_wp, h_iter_wp, self.nx, self.ny],
                                    device=device,
                                )

                            adaptive_state = _run_adaptive_inner_kcycle_blocks(
                                initial_residual_rms=initial_head_residual_rms,
                                target_residual_rms=inner_target_head_residual_rms,
                                forcing_eta=forcing_eta_used,
                                previous_outer_residual_rms=previous_outer_head_residual_rms,
                                previous_outer_dh_rms=previous_outer_dh_rms,
                                max_cycles=max_cycles_hard_i,
                                config=adaptive_inner_config,
                                run_block=_run_adaptive_block,
                                rollback_block=_rollback_adaptive_block,
                                initial_relative_flow_residual_rms=initial_flow_residual_rms,
                                target_relative_flow_residual_rms=inner_target_relative_flow_residual_rms,
                            )
                            previous_initial_head_residual_rms = float(initial_head_residual_rms)
                            inner_residual_check_count += int(adaptive_state.residual_check_count)
                            inner_cycles_used_i = int(adaptive_state.total_cycles)
                            adaptive_pre_fallback_cycles = int(adaptive_state.total_cycles)
                            adaptive_pre_fallback_blocks = int(adaptive_state.block_index)
                            inner_block_counts.append(int(adaptive_state.block_index))
                            if adaptive_state.target_achieved:
                                adaptive_target_achievement_count += 1
                            if adaptive_state.stalled:
                                stalled_inner_solve_count += 1
                            if adaptive_state.diverged:
                                divergent_inner_solve_count += 1
                            if adaptive_state.rollback_count:
                                rolled_back_block_count += int(adaptive_state.rollback_count)
                            inner_final_head_residual_rms = float(adaptive_state.final_residual_rms)
                            info_lin = {
                                "converged": bool(adaptive_state.converged),
                                "n_cycles_used": int(adaptive_state.total_cycles),
                                "h_rms_end": (
                                    float(adaptive_state.final_residual_rms)
                                    if np.isfinite(float(adaptive_state.final_residual_rms))
                                    else None
                                ),
                                "adaptive_inner_residual_check_count": int(adaptive_state.residual_check_count),
                                "coarse_operator_mode": fast_path_coarse_operator_mode,
                                "fine_operator_residual_checked": True,
                                "adaptive_inner_controller_enabled": True,
                                "adaptive_inner_controller_used": True,
                                "adaptive_inner_fallback_to_legacy_dh": False,
                                "adaptive_inner_fallback_reason": "",
                                "inner_target_achieved": bool(adaptive_state.target_achieved),
                                "inner_usable_for_picard": bool(adaptive_state.usable_for_picard),
                                "inner_stalled": bool(adaptive_state.stalled),
                                "inner_diverged": bool(adaptive_state.diverged),
                                "inner_rollback_count": int(adaptive_state.rollback_count),
                                "inner_termination_reason": str(adaptive_state.termination_reason),
                                "initial_head_residual_rms": float(initial_head_residual_rms),
                                "initial_relative_flow_residual_rms": float(initial_relative_flow_residual_rms),
                                "initial_flow_residual_rms": float(initial_flow_residual_rms),
                                "target_head_residual_rms": float(inner_target_head_residual_rms),
                                "target_relative_flow_residual_rms": float(inner_target_relative_flow_residual_rms),
                                "final_flow_residual_rms": float(adaptive_state.final_relative_flow_residual_rms),
                                "head_reduction_ratio": (
                                    float(adaptive_state.final_residual_rms)
                                    / max(float(adaptive_state.initial_residual_rms), float(adaptive_inner_config.residual_floor))
                                ),
                                "flow_reduction_ratio": (
                                    float(adaptive_state.final_relative_flow_residual_rms)
                                    / max(float(adaptive_state.initial_relative_flow_residual_rms), float(adaptive_inner_config.residual_floor))
                                ),
                                "head_q": list(adaptive_state.head_per_cycle_convergence_factors),
                                "flow_q": list(adaptive_state.flow_per_cycle_convergence_factors),
                                "controller_q": list(adaptive_state.controller_per_cycle_convergence_factors),
                                "head_target_gap": (
                                    float(adaptive_state.final_residual_rms)
                                    / max(float(inner_target_head_residual_rms), float(adaptive_inner_config.residual_floor))
                                ),
                                "flow_target_gap": (
                                    float(adaptive_state.final_relative_flow_residual_rms)
                                    / max(float(inner_target_relative_flow_residual_rms), float(adaptive_inner_config.residual_floor))
                                ),
                                "controller_target_gap": max(
                                    float(adaptive_state.final_residual_rms)
                                    / max(float(inner_target_head_residual_rms), float(adaptive_inner_config.residual_floor)),
                                    float(adaptive_state.final_relative_flow_residual_rms)
                                    / max(float(inner_target_relative_flow_residual_rms), float(adaptive_inner_config.residual_floor)),
                                ),
                                "final_head_residual_rms": (
                                    float(adaptive_state.final_residual_rms)
                                    if np.isfinite(float(adaptive_state.final_residual_rms))
                                    else None
                                ),
                                "forcing_eta": float(forcing_eta_used),
                                "controller_mode": "adaptive_residual_blocks",
                                "inner_block_count": int(adaptive_state.block_index),
                                "inner_cycles_per_block": list(adaptive_state.cycles_per_block),
                            }
                            if adaptive_inner_config.save_block_history:
                                info_lin.update(
                                    {
                                        "inner_cycles_per_block": list(adaptive_state.cycles_per_block),
                                        "inner_residuals_per_block": list(adaptive_state.residuals_per_block),
                                        "inner_contraction_ratios": list(adaptive_state.contraction_ratios),
                                        "inner_per_cycle_convergence_factors": list(
                                            adaptive_state.per_cycle_convergence_factors
                                        ),
                                        "inner_predicted_cycles_per_block": list(
                                            adaptive_state.predicted_cycles_per_block
                                        ),
                                    }
                                )

                            if not adaptive_state.target_achieved:
                                adaptive_fallback_reason = (
                                    adaptive_state.fallback_reason or adaptive_state.termination_reason
                                )
                                adaptive_controller_used = False
                                legacy_dh_fallback_used = True

                    if not adaptive_controller_used:
                        legacy_dh_fallback_count += 1 if legacy_dh_fallback_used else 0
                        inner_max_cycles_i = _select_legacy_unconfined_inner_max_cycles_from_dh(
                            previous_dh_measure=previous_dh_measure,
                            early_cycles=inner_max_cycles_early_i,
                            middle_cycles=inner_max_cycles_middle_i,
                            late_cycles=inner_max_cycles_late_i,
                            middle_dh=inner_middle_dh_f,
                            late_dh=inner_late_dh_f,
                        )
                        legacy_cycles_requested_i = int(inner_max_cycles_i)
                        inner_max_cycles_i = _remaining_legacy_fallback_cycles(
                            max_cycles=max_cycles_hard_i,
                            adaptive_cycles_used=adaptive_pre_fallback_cycles,
                            selected_legacy_cycles=inner_max_cycles_i,
                        )
                        inner_controls = dict(controls)
                        inner_controls["max_cycles"] = int(inner_max_cycles_i)
                        inner_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                        inner_kcycle_caps.append(int(inner_max_cycles_i))

                        if inner_max_cycles_i > 0:
                            inner_t0 = _fast_path_phase_start()
                            if use_incremental_picard:
                                _legacy_x_wp = delta_wp
                                _legacy_rhs_wp = residual_wp
                                _legacy_bc_values_wp = zero_bc_values_wp
                            else:
                                _legacy_x_wp = h_iter_wp
                                _legacy_rhs_wp = rhs_eff_wp
                                _legacy_bc_values_wp = self.bc_values_wp
                            info_lin = self._solve_multigrid_kcycle_device_buffers(
                                x_wp=_legacy_x_wp,
                                rhs_wp=_legacy_rhs_wp,
                                T_wp=self.T_wp,
                                storage_diag_wp=storage_diag_wp,
                                active_wp=self.active_wp,
                                bc_mask_wp=self.bc_mask_wp,
                                bc_values_wp=_legacy_bc_values_wp,
                                levels=self.mg_levels,
                                solve_controls=inner_controls,
                                return_scalar_info=False
                            )
                            inner_solver_seconds += _fast_path_phase_elapsed(inner_t0)
                            inner_cycles_used_i = int(
                                info_lin["n_cycles_used"]
                                if info_lin.get("n_cycles_used") is not None else inner_max_cycles_i
                            )
                        else:
                            info_lin = {
                                "n_cycles_used": 0,
                                "converged": False,
                                "inner_termination_reason": "hard_cycle_ceiling_before_legacy_fallback",
                            }
                            inner_cycles_used_i = 0
                        legacy_cycles_used_i = int(inner_cycles_used_i)
                        if adaptive_pre_fallback_blocks > 0 and inner_block_counts:
                            inner_block_counts.pop()
                        inner_cycles_used_i += int(adaptive_pre_fallback_cycles)
                        inner_block_counts.append(int(1 + adaptive_pre_fallback_blocks))
                        inner_final_head_residual_rms = float("nan")
                        info_lin.update(
                            {
                                "adaptive_inner_controller_enabled": bool(adaptive_inner_config.enabled),
                                "adaptive_inner_controller_used": False,
                                "adaptive_inner_fallback_to_legacy_dh": bool(legacy_dh_fallback_used),
                                "adaptive_inner_fallback_reason": str(adaptive_fallback_reason),
                                "controller_mode": "legacy_dh_schedule",
                                "inner_block_count": int(1 + adaptive_pre_fallback_blocks),
                                "inner_cycles_per_block": (
                                    list(adaptive_state.cycles_per_block)
                                    if adaptive_state is not None else []
                                ) + [legacy_cycles_used_i],
                                "adaptive_cycles_before_fallback": int(adaptive_pre_fallback_cycles),
                                "legacy_fallback_cycles": int(legacy_cycles_used_i),
                                "legacy_fallback_cycles_requested": int(legacy_cycles_requested_i),
                                "inner_termination_reason": (
                                    "legacy_fixed_cycle_cap"
                                    if inner_max_cycles_i > 0
                                    else "hard_cycle_ceiling_before_legacy_fallback"
                                ),
                                "initial_head_residual_rms": (
                                    float(inner_initial_head_residual_rms)
                                    if np.isfinite(inner_initial_head_residual_rms)
                                    else None
                                ),
                                "target_head_residual_rms": (
                                    float(inner_target_head_residual_rms)
                                    if np.isfinite(inner_target_head_residual_rms)
                                    else None
                                ),
                                "final_head_residual_rms": (
                                    float(inner_final_head_residual_rms)
                                    if np.isfinite(inner_final_head_residual_rms)
                                    else None
                                ),
                                "forcing_eta": float(forcing_eta_used) if np.isfinite(forcing_eta_used) else None,
                            }
                        )
                    else:
                        inner_kcycle_caps.append(int(max_cycles_hard_i))
                        inner_cycles_used_i = int(info_lin.get("n_cycles_used", 0))

                    inner_kcycle_used.append(inner_cycles_used_i)
                    total_inner_kcycles += inner_cycles_used_i
                    maximum_inner_kcycles_in_one_outer_iteration = max(
                        maximum_inner_kcycles_in_one_outer_iteration,
                        inner_cycles_used_i,
                    )
                    if use_incremental_picard:
                        wp.launch(
                            kernel=apply_relaxed_correction_kernel,
                            dim=dim2d,
                            inputs=[
                                h_snapshot_wp,
                                delta_wp,
                                self.active_wp,
                                self.bc_mask_wp,
                                self.bc_values_wp,
                                WP_FLOAT(omega_current_f),
                                WP_FLOAT(max_update_f),
                                self.nx,
                                self.ny,
                                h_iter_wp,
                            ],
                            device=device,
                        )
                    else:
                        wp.launch(
                            kernel=apply_relaxed_clipped_picard_update_kernel,
                            dim=dim2d,
                            inputs=[
                                h_iter_wp,
                                h_snapshot_wp,
                                self.active_wp,
                                self.bc_mask_wp,
                                self.bc_values_wp,
                                WP_FLOAT(omega_current_f),
                                WP_FLOAT(max_update_f),
                                self.nx,
                                self.ny,
                                h_iter_wp,
                            ],
                            device=device,
                        )

                    outer_check_t0 = _fast_path_phase_start()
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[dh_rms_buf], device=device)
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[dh_max_buf], device=device)
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[flow_rTr_buf], device=device)
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[head_rTr_buf], device=device)
                    wp.launch(
                        kernel=kcycle_check_dh_and_dual_residual_kernel,
                        dim=dim2d,
                        inputs=[
                            h_iter_wp,
                            h_snapshot_wp,
                            rhs_eff_wp,
                            self.T_wp,
                            self.active_wp,
                            self.bc_mask_wp,
                            self.mg_levels[0].gh_mask_wp,
                            self.mg_levels[0].ghb_factor_wp,
                            storage_diag_wp,
                            dh_rms_buf,
                            dh_max_buf,
                            flow_rTr_buf,
                            head_rTr_buf,
                            int(1 if self.use_ghb else 0),
                            self.nx,
                            self.ny,
                        ],
                        device=device
                    )
                    counters["scalar_reductions"] += 1
                    counters["gpu_scalar_synchronizations"] += 6
                    period_gpu_scalar_syncs += 6
                    last_dh_max = float(dh_max_buf.numpy()[0])
                    last_dh_rms = float(np.sqrt(max(float(dh_rms_buf.numpy()[0]), 0.0) / float(max(n_free, 1))))
                    if adaptive_dt_enabled_b:
                        adaptive_dt_dh_history.append(float(last_dh_max))
                    last_flow_residual_rms = float(
                        np.sqrt(max(float(flow_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                    )
                    last_head_residual_rms = float(
                        np.sqrt(max(float(head_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                    )
                    last_storage_diag_change_max = float(storage_change_max_buf.numpy()[0])
                    last_storage_diag_change_rms = float(
                        np.sqrt(max(float(storage_change_sum_sq_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                    )
                    outer_convergence_check_seconds += _fast_path_phase_elapsed(outer_check_t0)
                    previous_outer_head_residual_rms_before = previous_outer_head_residual_rms
                    previous_outer_dh_rms_before = previous_outer_dh_rms
                    previous_dh_measure = float(last_dh_max)
                    previous_outer_head_residual_rms = float(last_head_residual_rms)
                    previous_outer_dh_rms = float(last_dh_rms)

                    adaptive_final_linearisation = bool(
                        adaptive_inner_config.enabled
                        and info_lin.get("adaptive_inner_controller_used", False)
                    )
                    final_linearisation_solved = _adaptive_practical_acceptance_allowed(
                        practical_acceptance_enabled=True,
                        adaptive_controller_used=adaptive_final_linearisation,
                        inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                    )
                    strict_picard_convergence_passed = bool(
                        final_linearisation_solved
                        and last_dh_max <= hclose
                        and last_head_residual_rms <= strict_head_residual_tol_f
                    )
                    practical_picard_acceptance_passed = bool(
                        _adaptive_practical_acceptance_allowed(
                            practical_acceptance_enabled=practical_picard_acceptance_enabled_b,
                            adaptive_controller_used=adaptive_final_linearisation,
                            inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                        )
                        and int(outer_iter + 1) >= min_practical_outer_iterations_i
                        and np.isfinite(last_head_residual_rms)
                        and last_head_residual_rms <= practical_head_residual_tol_f
                        and np.isfinite(last_dh_rms)
                        and last_dh_rms <= practical_dh_rms_tol_f
                        and np.isfinite(last_storage_diag_change_rms)
                        and last_storage_diag_change_rms <= practical_storage_diag_change_rms_tol_f
                    )
                    production_acceptance_passed = bool(
                        strict_picard_convergence_passed or practical_picard_acceptance_passed
                    )
                    outer_summary = {
                        "outer_iteration": int(outer_iter + 1),
                        "controller_mode": str(info_lin.get("controller_mode", "legacy_dh_schedule")),
                        "initial_head_residual_rms": (
                            float(inner_initial_head_residual_rms)
                            if np.isfinite(inner_initial_head_residual_rms)
                            else None
                        ),
                        "target_head_residual_rms": (
                            float(inner_target_head_residual_rms)
                            if np.isfinite(inner_target_head_residual_rms)
                            else None
                        ),
                        "initial_relative_flow_residual_rms": (
                            float(inner_initial_relative_flow_residual_rms)
                            if np.isfinite(inner_initial_relative_flow_residual_rms) else None
                        ),
                        "initial_flow_residual_rms": (
                            float(inner_initial_flow_residual_rms)
                            if np.isfinite(inner_initial_flow_residual_rms) else None
                        ),
                        "target_flow_residual_rms": (
                            float(inner_target_relative_flow_residual_rms)
                            if np.isfinite(inner_target_relative_flow_residual_rms) else None
                        ),
                        "final_flow_residual_rms": info_lin.get("final_flow_residual_rms"),
                        "target_relative_flow_residual_rms": (
                            float(inner_target_relative_flow_residual_rms)
                            if np.isfinite(inner_target_relative_flow_residual_rms) else None
                        ),
                        "final_head_residual_rms": float(last_head_residual_rms),
                        "final_max_abs_head_change": float(last_dh_max),
                        "final_rms_head_change": float(last_dh_rms),
                        "final_relative_flow_residual_rms": info_lin.get(
                            "final_relative_flow_residual_rms"
                        ),
                        "head_reduction_ratio": info_lin.get("head_reduction_ratio"),
                        "flow_reduction_ratio": info_lin.get("flow_reduction_ratio"),
                        "head_q": list(info_lin.get("head_q", [])),
                        "flow_q": list(info_lin.get("flow_q", [])),
                        "controller_q": list(info_lin.get("controller_q", [])),
                        "head_target_gap": info_lin.get("head_target_gap"),
                        "flow_target_gap": info_lin.get("flow_target_gap"),
                        "controller_target_gap": info_lin.get("controller_target_gap"),
                        "adaptive_cycles_before_fallback": int(
                            info_lin.get("adaptive_cycles_before_fallback", adaptive_pre_fallback_cycles)
                        ),
                        "legacy_fallback_cycles": int(info_lin.get("legacy_fallback_cycles", 0)),
                        "total_cycles": int(inner_cycles_used_i),
                        "refreshed_acceptance_passed": None,
                        "refreshed_acceptance_checked": False,
                        "provisional_picard_acceptance_passed": bool(production_acceptance_passed),
                        "outer_iteration_of_acceptance": None,
                        "termination_reason": "continuing_picard",
                        "forcing_eta": float(forcing_eta_used) if np.isfinite(forcing_eta_used) else None,
                        "previous_outer_head_residual_rms": (
                            float(previous_outer_head_residual_rms_before)
                            if previous_outer_head_residual_rms_before is not None
                            and np.isfinite(previous_outer_head_residual_rms_before)
                            else None
                        ),
                        "previous_outer_dh_rms": (
                            float(previous_outer_dh_rms_before)
                            if previous_outer_dh_rms_before is not None and np.isfinite(previous_outer_dh_rms_before)
                            else None
                        ),
                        "total_inner_kcycles": int(inner_cycles_used_i),
                        "inner_block_count": int(info_lin.get("inner_block_count", 1)),
                        "inner_target_achieved": bool(info_lin.get("inner_target_achieved", False)),
                        "inner_usable_for_picard": bool(info_lin.get("inner_usable_for_picard", True)),
                        "inner_stalled": bool(info_lin.get("inner_stalled", False)),
                        "inner_diverged": bool(info_lin.get("inner_diverged", False)),
                        "inner_rollback_count": int(info_lin.get("inner_rollback_count", 0) or 0),
                        "inner_termination_reason": str(
                            info_lin.get("inner_termination_reason", "legacy_fixed_cycle_cap")
                        ),
                        "legacy_dh_fallback_used": bool(info_lin.get("adaptive_inner_fallback_to_legacy_dh", False)),
                        "inner_cycles_per_block": list(info_lin.get("inner_cycles_per_block", [])),
                        "adaptive_dt_substep_index": int(len(adaptive_dt_substep_dts)),
                        "adaptive_dt_substep_dt": float(actual_dt_f),
                        "adaptive_dt_practical_at_min": bool(adaptive_dt_practical_at_min_b),
                    }
                    if adaptive_inner_config.save_block_history:
                        outer_summary["inner_cycles_per_block"] = list(info_lin.get("inner_cycles_per_block", []))
                        outer_summary["inner_residuals_per_block"] = list(info_lin.get("inner_residuals_per_block", []))
                        outer_summary["inner_contraction_ratios"] = list(info_lin.get("inner_contraction_ratios", []))
                        outer_summary["inner_per_cycle_convergence_factors"] = list(
                            info_lin.get("inner_per_cycle_convergence_factors", [])
                        )
                        outer_summary["inner_predicted_cycles_per_block"] = list(
                            info_lin.get("inner_predicted_cycles_per_block", [])
                        )
                    outer_iteration_summaries.append(outer_summary)
                    if production_acceptance_passed and (
                        not adaptive_dt_enabled_b
                        or strict_picard_convergence_passed
                        or adaptive_dt_practical_at_min_b
                    ):
                        refreshed_result = evaluate_refreshed_nonlinear_candidate(
                            outer_iteration=int(outer_iter + 1),
                            info_lin=info_lin,
                            dh_max=last_dh_max,
                            dh_rms=last_dh_rms,
                            substep_dt=actual_dt_f,
                            require_strict=bool(adaptive_dt_enabled_b and not adaptive_dt_practical_at_min_b),
                        )
                        outer_summary["refreshed_acceptance_checked"] = True
                        outer_summary["refreshed_acceptance_passed"] = bool(
                            refreshed_result["production_acceptance_passed"]
                        )
                        if refreshed_result["production_acceptance_passed"]:
                            last_head_residual_rms = float(refreshed_result["head_residual_rms"])
                            last_flow_residual_rms = float(refreshed_result["flow_residual_rms"])
                            last_storage_diag_change_max = float(refreshed_result["storage_diag_change_max"])
                            last_storage_diag_change_rms = float(refreshed_result["storage_diag_change_rms"])
                            strict_picard_convergence_passed = bool(refreshed_result["strict_acceptance_passed"])
                            practical_picard_acceptance_passed = bool(refreshed_result["practical_acceptance_passed"])
                            production_acceptance_passed = True
                            outer_summary["termination_reason"] = (
                                "refreshed_strict_acceptance"
                                if strict_picard_convergence_passed else "refreshed_practical_acceptance"
                            )
                            outer_summary["outer_iteration_of_acceptance"] = int(outer_iter + 1)
                            if not adaptive_dt_enabled_b:
                                break
                            adaptive_dt_substep_dts.append(float(actual_dt_f))
                            remaining_dt_f = max(0.0, remaining_dt_f - actual_dt_f)
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_iter_wp, h_substep_start_wp, self.nx, self.ny],
                                device=device,
                            )
                            if remaining_dt_f <= max(1.0e-12, period_dt_f * 1.0e-12):
                                break
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_substep_start_wp, h_prev_wp, self.nx, self.ny],
                                device=device,
                            )
                            if refreshed_result["strict_acceptance_passed"] and not adaptive_dt_practical_at_min_b and not adaptive_dt_extension_used_b:
                                # Grow only after a clean strict acceptance (no practical
                                # fallback or budget extension touched this sub-step). A
                                # sub-step that needed assistance keeps dt instead of
                                # re-attempting strict at a larger dt and shrinking
                                # straight back (retry storm / grow-shrink oscillation).
                                if adaptive_dt_growth_steps_i < adaptive_dt_max_growth_steps_i:
                                    current_dt_f = min(
                                        period_dt_f,
                                        current_dt_f * adaptive_dt_grow_factor_f,
                                    )
                                    adaptive_dt_growth_steps_i += 1
                            actual_dt_f = min(current_dt_f, remaining_dt_f)
                            if remaining_dt_f - actual_dt_f < dt_min_f:
                                # Absorb a sub-dt_min sliver into the final sub-step.
                                actual_dt_f = remaining_dt_f
                            substep_outer_limit_i = adaptive_dt_strict_max_outer_i
                            adaptive_dt_practical_at_min_b = False
                            adaptive_dt_dh_history = []
                            adaptive_dt_extension_used_b = False
                            adaptive_dt_early_shrink_streak_i = 0
                            previous_dh_measure = None
                            previous_outer_head_residual_rms = None
                            previous_initial_head_residual_rms = None
                            previous_outer_dh_rms = None
                            storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                            outer_iter = 0
                            continue
                        production_acceptance_passed = False

                    adaptive_dt_budget_exhausted_b = bool(
                        adaptive_dt_enabled_b and outer_iter + 1 >= substep_outer_limit_i
                    )
                    adaptive_dt_early_shrink_b = False
                    if (
                        adaptive_dt_enabled_b
                        and not adaptive_dt_budget_exhausted_b
                        and not adaptive_dt_practical_at_min_b
                        and adaptive_dt_early_shrink_enabled_b
                        and actual_dt_f > dt_min_f + max(1.0e-12, period_dt_f * 1.0e-12)
                    ):
                        # Early shrink: the dh contraction projection says strict
                        # cannot reach hclose within the remaining budget — but the
                        # comparison is against budget + available extension, since an
                        # extension at exhaustion finishes a near-miss far cheaper
                        # than a shrink + full retry. Only genuinely hopeless
                        # sub-steps shrink early, and only after the pessimistic
                        # projection persists for early_shrink_patience consecutive
                        # checks (early-iteration contraction is often pessimistic;
                        # it accelerates as the Picard iterate settles).
                        adaptive_dt_effective_budget_i = int(substep_outer_limit_i)
                        if adaptive_dt_extension_enabled_b and not adaptive_dt_extension_used_b:
                            adaptive_dt_effective_budget_i += adaptive_dt_extension_max_outer_i
                        if _adaptive_dt_should_early_shrink(
                            adaptive_dt_dh_history,
                            tol=hclose,
                            outer_iterations_done=int(outer_iter + 1),
                            budget=int(adaptive_dt_effective_budget_i),
                            min_outer=adaptive_dt_early_shrink_min_outer_i,
                        ):
                            adaptive_dt_early_shrink_streak_i += 1
                            adaptive_dt_early_shrink_b = bool(
                                adaptive_dt_early_shrink_streak_i >= adaptive_dt_early_shrink_patience_i
                            )
                        else:
                            adaptive_dt_early_shrink_streak_i = 0
                            adaptive_dt_early_shrink_b = False
                    if adaptive_dt_budget_exhausted_b or adaptive_dt_early_shrink_b:
                        if (
                            adaptive_dt_budget_exhausted_b
                            and not adaptive_dt_early_shrink_b
                            and not adaptive_dt_practical_at_min_b
                            and adaptive_dt_extension_enabled_b
                            and not adaptive_dt_extension_used_b
                            and _adaptive_dt_should_extend_budget(
                                adaptive_dt_dh_history,
                                tol=hclose,
                                extension_factor=adaptive_dt_extension_factor_f,
                                extension_contraction_ratio=adaptive_dt_extension_contraction_ratio_f,
                            )
                        ):
                            # Budget extension: strict is close and still contracting,
                            # so a few extra iterations are cheaper than a shrink and
                            # a full retry of the sub-step. At most one per sub-step.
                            substep_outer_limit_i += adaptive_dt_extension_max_outer_i
                            adaptive_dt_extension_used_b = True
                            adaptive_dt_extension_count += 1
                            outer_iter += 1
                            continue
                        if actual_dt_f > dt_min_f + max(1.0e-12, period_dt_f * 1.0e-12):
                            current_dt_f = max(dt_min_f, actual_dt_f * adaptive_dt_shrink_factor_f)
                            actual_dt_f = min(current_dt_f, remaining_dt_f)
                            if remaining_dt_f - actual_dt_f < dt_min_f:
                                # Absorb a sub-dt_min sliver into the final sub-step.
                                actual_dt_f = remaining_dt_f
                            adaptive_dt_growth_steps_i = 0
                            adaptive_dt_retry_count += 1
                            if adaptive_dt_early_shrink_b:
                                adaptive_dt_early_shrink_count += 1
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_substep_start_wp, h_prev_wp, self.nx, self.ny],
                                device=device,
                            )
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_substep_start_wp, h_iter_wp, self.nx, self.ny],
                                device=device,
                            )
                            storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                            adaptive_dt_dh_history = []
                            adaptive_dt_extension_used_b = False
                            adaptive_dt_early_shrink_streak_i = 0
                            previous_dh_measure = None
                            previous_outer_head_residual_rms = None
                            previous_initial_head_residual_rms = None
                            previous_outer_dh_rms = None
                            outer_iter = 0
                            continue
                        if not adaptive_dt_practical_at_min_b:
                            adaptive_dt_practical_at_min_b = True
                            adaptive_dt_practical_fallback_count += 1
                            substep_outer_limit_i = max_outer
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_substep_start_wp, h_prev_wp, self.nx, self.ny],
                                device=device,
                            )
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_substep_start_wp, h_iter_wp, self.nx, self.ny],
                                device=device,
                            )
                            storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                            adaptive_dt_dh_history = []
                            adaptive_dt_extension_used_b = False
                            adaptive_dt_early_shrink_streak_i = 0
                            previous_dh_measure = None
                            previous_outer_head_residual_rms = None
                            previous_initial_head_residual_rms = None
                            previous_outer_dh_rms = None
                            outer_iter = 0
                            continue

                    outer_iter += 1

                phase_t0 = _fast_path_phase_start()
                wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[storage_diag_wp, storage_diag_prev_wp, self.nx, self.ny], device=device)
                wp.launch(
                    kernel=update_unconfined_transmissivity_from_head_kernel,
                    dim=dim2d,
                    inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                    device=device
                )
                T_update_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["T_device_updates"] += 1
                phase_t0 = _fast_path_phase_start()
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)
                wp.launch(
                    kernel=update_secant_sy_storage_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                        sy_f, ss_f, dx_f, actual_dt_f, min_sat_f, 1.0e-12, self.nx, self.ny,
                        storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                        storage_change_sum_sq_buf, storage_change_max_buf
                    ],
                    device=device
                )
                storage_kernel_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["storage_device_updates"] += 1
                counters["rhs_device_updates"] += 1
                phase_t0 = _fast_path_phase_start()
                wp.launch(
                    kernel=build_transient_rhs_from_storage_kernel,
                    dim=dim2d,
                    inputs=[
                        self.R_wp,
                        storage_diag_wp,
                        h_prev_wp,
                        self.active_wp,
                        self.bc_mask_wp,
                        self.bc_values_wp,
                        dx_f,
                        self.nx,
                        self.ny,
                        rhs_eff_wp,
                    ],
                    device=device
                )
                rhs_assembly_seconds += _fast_path_phase_elapsed(phase_t0)
                phase_t0 = _fast_path_phase_start()
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[flow_rTr_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[head_rTr_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rhs_rTr_buf], device=device)
                wp.launch(
                    kernel=compute_dual_residual_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp,
                        rhs_eff_wp,
                        self.T_wp,
                        self.active_wp,
                        self.bc_mask_wp,
                        self.mg_levels[0].gh_mask_wp,
                        self.mg_levels[0].ghb_factor_wp,
                        storage_diag_wp,
                        flow_rTr_buf,
                        head_rTr_buf,
                        self.nx,
                        self.ny,
                    ],
                    device=device,
                )
                wp.launch(
                    kernel=compute_active_rhs_l2_kernel,
                    dim=dim2d,
                    inputs=[rhs_eff_wp, self.active_wp, self.bc_mask_wp, rhs_rTr_buf, self.nx, self.ny],
                    device=device,
                )
                final_nonlinear_residual_check_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["scalar_reductions"] += 1
                counters["gpu_scalar_synchronizations"] += 4
                period_gpu_scalar_syncs += 4
                last_flow_residual_rms = float(
                    np.sqrt(max(float(flow_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                last_head_residual_rms = float(
                    np.sqrt(max(float(head_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                final_rhs_rms = float(
                    np.sqrt(max(float(rhs_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                final_relative_flow_residual_rms = last_flow_residual_rms / max(
                    final_rhs_rms, float(adaptive_inner_config.residual_floor)
                )
                last_storage_diag_change_max = float(storage_change_max_buf.numpy()[0])
                last_storage_diag_change_rms = float(
                    np.sqrt(max(float(storage_change_sum_sq_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                strict_picard_convergence_passed = bool(
                    final_linearisation_solved
                    and last_dh_max <= hclose
                    and last_head_residual_rms <= strict_head_residual_tol_f
                )
                practical_picard_acceptance_passed = bool(
                    _adaptive_practical_acceptance_allowed(
                        practical_acceptance_enabled=practical_picard_acceptance_enabled_b,
                        adaptive_controller_used=adaptive_final_linearisation,
                        inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                        final_relative_flow_residual_rms=final_relative_flow_residual_rms,
                        relative_flow_target=float(adaptive_inner_config.relative_flow_residual_target),
                    )
                    and int(min(max_outer, outer_iter + 1)) >= min_practical_outer_iterations_i
                    and np.isfinite(last_head_residual_rms)
                    and last_head_residual_rms <= practical_head_residual_tol_f
                    and np.isfinite(last_dh_rms)
                    and last_dh_rms <= practical_dh_rms_tol_f
                    and np.isfinite(last_storage_diag_change_rms)
                    and last_storage_diag_change_rms <= practical_storage_diag_change_rms_tol_f
                )
                production_acceptance_passed = bool(
                    strict_picard_convergence_passed or practical_picard_acceptance_passed
                )
                if outer_iteration_summaries:
                    outer_iteration_summaries[-1]["refreshed_acceptance_passed"] = bool(
                        production_acceptance_passed
                    )

                refreshed_result = evaluate_refreshed_nonlinear_candidate(
                    outer_iteration=int(min(max_outer, outer_iter + 1)),
                    info_lin=info_lin,
                    dh_max=last_dh_max,
                    dh_rms=last_dh_rms,
                    substep_dt=actual_dt_f,
                    require_strict=bool(adaptive_dt_enabled_b and not adaptive_dt_practical_at_min_b),
                )
                strict_picard_convergence_passed = bool(refreshed_result["strict_acceptance_passed"])
                practical_picard_acceptance_passed = bool(refreshed_result["practical_acceptance_passed"])
                production_acceptance_passed = bool(refreshed_result["production_acceptance_passed"])
                last_head_residual_rms = float(refreshed_result["head_residual_rms"])
                last_flow_residual_rms = float(refreshed_result["flow_residual_rms"])
                last_storage_diag_change_max = float(refreshed_result["storage_diag_change_max"])
                last_storage_diag_change_rms = float(refreshed_result["storage_diag_change_rms"])
                if outer_iteration_summaries:
                    outer_iteration_summaries[-1]["refreshed_acceptance_checked"] = True
                    outer_iteration_summaries[-1]["refreshed_acceptance_passed"] = bool(production_acceptance_passed)
                    if not production_acceptance_passed:
                        outer_iteration_summaries[-1]["termination_reason"] = "max_outer_iterations"

                if (not production_acceptance_passed) and (not allow_unaccepted_transient_period_b):
                    if adaptive_dt_enabled_b and actual_dt_f <= dt_min_f + max(
                        1.0e-12,
                        period_dt_f * 1.0e-12,
                    ):
                        raise RuntimeError(f"adaptive dt failed at dt_min={dt_min_f}")
                    raise RuntimeError(
                        _format_unaccepted_transient_period_error(
                            period_index=period_index,
                            outer_iterations=int(min(max_outer, outer_iter + 1)),
                            final_max_abs_head_change=last_dh_max,
                            final_rms_head_change=last_dh_rms,
                            final_head_residual_rms=last_head_residual_rms,
                            final_flow_residual_rms=last_flow_residual_rms,
                            storage_diag_change_max=last_storage_diag_change_max,
                            storage_diag_change_rms=last_storage_diag_change_rms,
                            storage_mode=str(storage_mode),
                            storage_reference=str(storage_reference),
                            coarse_operator_mode=str(fast_path_coarse_operator_mode),
                            coarse_krylov_method="recursive_kcycle_safe_alpha",
                            total_inner_cycles=int(total_inner_kcycles),
                            inner_controller_mode=str(info_lin.get("controller_mode", "legacy_dh_schedule")),
                            last_inner_termination_reason=str(
                                info_lin.get("inner_termination_reason", "legacy_fixed_cycle_cap")
                            ),
                            last_inner_initial_residual=info_lin.get("initial_head_residual_rms"),
                            last_inner_target_residual=info_lin.get("target_head_residual_rms"),
                            last_inner_final_residual=info_lin.get("final_head_residual_rms"),
                            last_inner_block_count=info_lin.get("inner_block_count"),
                            stalled_inner_solve_count=int(stalled_inner_solve_count),
                            divergent_inner_solve_count=int(divergent_inner_solve_count),
                        )
                    )

                wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[h_iter_wp, h_prev_wp, self.nx, self.ny], device=device)

                period_times[period_index] = time.perf_counter() - period_t0
                counters["device_to_host_full_grid_copies"] += 1
                counters["head_downloads"] += 1
                head_download_t0 = _fast_path_phase_start()
                head_arr = np.asarray(h_iter_wp.numpy(), dtype=np.float64)
                head_download_seconds += _fast_path_phase_elapsed(head_download_t0)
                heads_per_period[period_index] = head_arr
                if save_diagnostics_b:
                    counters["device_to_host_full_grid_copies"] += 3
                    storage_ref_arr = head_arr.copy()
                    storage_coeff_arr = np.asarray(storage_coeff_wp.numpy(), dtype=np.float64)
                    sy_coeff_arr = np.asarray(sy_coeff_wp.numpy(), dtype=np.float64)
                    ss_coeff_arr = np.asarray(ss_coeff_wp.numpy(), dtype=np.float64)
                    delta_head = head_arr - period_head_old
                    exact_storage_term, exact_sy_term, exact_ss_term = exact_unconfined_storage_terms(
                        head_new=storage_ref_arr,
                        head_old=period_head_old,
                        bottom=bottom,
                        top=top,
                        specific_yield=float(sy),
                        specific_storage=float(ss),
                        dt=dt_f,
                    )
                    heads_old_per_period[period_index] = period_head_old
                    storage_reference_heads[period_index] = storage_ref_arr
                    storage_coeffs[period_index] = storage_coeff_arr
                    sy_coeffs[period_index] = sy_coeff_arr
                    ss_coeffs[period_index] = ss_coeff_arr
                    storage_terms[period_index] = exact_storage_term
                    sy_terms[period_index] = exact_sy_term
                    ss_terms[period_index] = exact_ss_term
                    sy_crossing_terms[period_index] = exact_sy_term
                info_period = dict(info_lin) if isinstance(info_lin, dict) else {}
                info_period.update(
                    {
                        "solver_type": "kcycle_unconfined_picard_device_fast_path",
                        "converged": bool(production_acceptance_passed),
                        "outer_iterations": int(min(max_outer, outer_iter + 1)),
                        "strict_picard_convergence_passed": bool(strict_picard_convergence_passed),
                        "practical_picard_acceptance_passed": bool(practical_picard_acceptance_passed),
                        "production_acceptance_passed": bool(production_acceptance_passed),
                        "final_max_abs_head_change": float(last_dh_max),
                        "final_rms_head_change": float(last_dh_rms),
                        "final_flow_residual_rms": float(last_flow_residual_rms),
                        "final_rhs_rms": float(final_rhs_rms),
                        "final_relative_flow_residual_rms": float(final_relative_flow_residual_rms),
                        "refreshed_acceptance_passed": bool(production_acceptance_passed),
                        "final_head_residual_rms": float(last_head_residual_rms),
                        "final_residual": float(last_head_residual_rms),
                        "adaptive_inner_controller_enabled": bool(adaptive_inner_config.enabled),
                        "adaptive_inner_controller_used": bool(info_lin.get("adaptive_inner_controller_used", False)),
                        "adaptive_inner_fallback_to_legacy_dh": bool(
                            info_lin.get("adaptive_inner_fallback_to_legacy_dh", False)
                        ),
                        "adaptive_inner_fallback_reason": str(
                            info_lin.get("adaptive_inner_fallback_reason", "")
                        ),
                        "storage_diag_change_max": float(last_storage_diag_change_max),
                        "storage_diag_change_rms": float(last_storage_diag_change_rms),
                        "storage_mode": str(storage_mode),
                        "storage_specific_storage_formulation": "secant_potential",
                        "unconfined_storage_mode_2d": str(storage_mode),
                        "storage_reference": str(storage_reference),
                        "incremental_picard_enabled": bool(use_incremental_picard),
                        "adaptive_dt_enabled": bool(adaptive_dt_enabled_b),
                        "adaptive_dt_min_fraction": float(adaptive_dt_min_fraction_f),
                        "adaptive_dt_retry_count": int(adaptive_dt_retry_count),
                        "adaptive_dt_practical_fallback_count": int(adaptive_dt_practical_fallback_count),
                        "adaptive_dt_early_shrink_count": int(adaptive_dt_early_shrink_count),
                        "adaptive_dt_extension_count": int(adaptive_dt_extension_count),
                        "adaptive_dt_total_outer_iterations": int(adaptive_dt_total_outer_iterations_i),
                        "adaptive_dt_substep_count": int(len(adaptive_dt_substep_dts)),
                        "adaptive_dt_substep_dts": [float(value) for value in adaptive_dt_substep_dts],
                        "device_side_picard_fast_path_active": True,
                        "unconfined_startup_mode": str(startup_mode),
                        "startup_inner_kcycles": int(startup_inner_cycles),
                        "startup_converged": startup_converged,
                        "practical_picard_acceptance_enabled": bool(practical_picard_acceptance_enabled_b),
                        "picard_relax": float(omega_current_f),
                        "max_head_change_per_outer_iteration": float(max_update_f),
                        "strict_head_residual_tol": float(strict_head_residual_tol_f),
                        "min_practical_outer_iterations": int(min_practical_outer_iterations_i),
                        "practical_head_residual_tol": float(practical_head_residual_tol_f),
                        "practical_residual_tol": float(practical_head_residual_tol_f),
                        "practical_residual_tol_deprecated_alias_used": bool(practical_residual_tol_alias_used),
                        "practical_dh_rms_tol": float(practical_dh_rms_tol_f),
                        "practical_storage_diag_change_rms_tol": float(practical_storage_diag_change_rms_tol_f),
                        "total_inner_kcycles": int(total_inner_kcycles),
                        "maximum_inner_kcycles_in_one_outer_iteration": int(maximum_inner_kcycles_in_one_outer_iteration),
                        "mean_inner_kcycles_per_outer_iteration": float(
                            total_inner_kcycles / float(max(int(min(max_outer, outer_iter + 1)), 1))
                        ),
                        "total_inner_blocks": int(sum(inner_block_counts)),
                        "mean_cycles_per_block": float(
                            total_inner_kcycles / float(max(sum(inner_block_counts), 1))
                        ),
                        "stalled_inner_solve_count": int(stalled_inner_solve_count),
                        "divergent_inner_solve_count": int(divergent_inner_solve_count),
                        "rolled_back_block_count": int(rolled_back_block_count),
                        "legacy_dh_fallback_count": int(legacy_dh_fallback_count),
                        "adaptive_target_achievement_count": int(adaptive_target_achievement_count),
                        "adaptive_inner_residual_check_count": int(inner_residual_check_count),
                        "coarse_operator_mode": str(fast_path_coarse_operator_mode),
                        "fine_operator_residual_checked": True,
                        "coarse_krylov_method": "recursive_kcycle_safe_alpha",
                        "gpu_scalar_synchronization_count": int(period_gpu_scalar_syncs),
                        "outer_iteration_summaries": outer_iteration_summaries,
                        "T_update_seconds": float(T_update_seconds),
                        "storage_kernel_seconds": float(storage_kernel_seconds),
                        "fine_m_inv_refresh_seconds": float(fine_m_inv_refresh_seconds),
                        "dynamic_coarse_refresh_seconds": float(dynamic_coarse_refresh_seconds),
                        "rhs_assembly_seconds": float(rhs_assembly_seconds),
                        "storage_assembly_seconds": float(storage_assembly_seconds),
                        "inner_solver_seconds": float(inner_solver_seconds),
                        "outer_convergence_check_seconds": float(outer_convergence_check_seconds),
                        "final_nonlinear_residual_check_seconds": float(final_nonlinear_residual_check_seconds),
                        "head_download_seconds": float(head_download_seconds),
                        "period_total_seconds": float(period_times[period_index]),
                    }
                )
                if save_diagnostics_b:
                    info_period["inner_kcycle_caps"] = [int(v) for v in inner_kcycle_caps]
                    info_period["inner_kcycle_used"] = [int(v) for v in inner_kcycle_used]
                    info_period["inner_block_counts"] = [int(v) for v in inner_block_counts]
                period_infos.append(info_period)
                last_info = info_period
                head_prev = head_arr

        else:
            for period_index in range(n_periods):
                self.update_uniform_recharge_in_place(float(rates[period_index]))
                counters["R_device_updates"] += 1
                if save_diagnostics_b:
                    period_head_old = np.asarray(head_prev, dtype=np.float64).copy()
                else:
                    period_head_old = head_prev
                period_t0 = time.perf_counter()
                head, info = self.solve(
                    formulation="unconfined",
                    initial_head=head_prev,
                    K_field=k,
                    zbot_field=bottom,
                    ztop_field=top,
                    transient=True,
                    storage_coeff=None,
                    dt=dt_f,
                    head_prev=head_prev,
                    return_info=True,
                    storage_reference=storage_reference,
                    unconfined_storage_mode_2d=storage_mode,
                    save_transient_diagnostics=save_diagnostics_b,
                    sy=float(sy),
                    ss=float(ss),
                    **controls,
                )
                period_times[period_index] = time.perf_counter() - period_t0
                counters["device_to_host_full_grid_copies"] += 1
                counters["head_downloads"] += 1
                head_arr = np.asarray(head, dtype=np.float64)
                info_out = dict(info) if isinstance(info, dict) else {}
                info_out.setdefault("incremental_picard_enabled", False)
                info_out.setdefault("adaptive_dt_enabled", False)
                storage_ref = info_out.pop("storage_reference_head_last_linearization_array", None)
                storage_coeff = info_out.pop("storage_coeff_last_linearization_array", None)
                sy_coeff = info_out.pop("sy_storage_coeff_last_linearization_array", None)
                ss_coeff = info_out.pop("ss_storage_coeff_last_linearization_array", None)
                if bool(info_out.get("cuda_graph_built_this_call", False)):
                    counters["hierarchy_rebuilds"] += 1

                heads_per_period[period_index] = head_arr
                if save_diagnostics_b:
                    if storage_ref is None:
                        storage_ref = head_arr if storage_reference == "current_picard" else period_head_old
                    if storage_coeff is None:
                        storage_coeff = np.zeros_like(head_arr)
                    if sy_coeff is None:
                        sy_coeff = np.zeros_like(head_arr)
                    if ss_coeff is None:
                        ss_coeff = np.asarray(storage_coeff, dtype=np.float64) - np.asarray(sy_coeff, dtype=np.float64)

                    storage_ref_arr = np.asarray(storage_ref, dtype=np.float64)
                    storage_coeff_arr = np.asarray(storage_coeff, dtype=np.float64)
                    sy_coeff_arr = np.asarray(sy_coeff, dtype=np.float64)
                    ss_coeff_arr = np.asarray(ss_coeff, dtype=np.float64)
                    delta_head = head_arr - period_head_old

                    exact_storage_term, exact_sy_term, exact_ss_term = exact_unconfined_storage_terms(
                        head_new=storage_ref_arr,
                        head_old=period_head_old,
                        bottom=bottom,
                        top=top,
                        specific_yield=float(sy),
                        specific_storage=float(ss),
                        dt=dt_f,
                    )
                    heads_old_per_period[period_index] = period_head_old
                    storage_reference_heads[period_index] = storage_ref_arr
                    storage_coeffs[period_index] = storage_coeff_arr
                    sy_coeffs[period_index] = sy_coeff_arr
                    ss_coeffs[period_index] = ss_coeff_arr
                    storage_terms[period_index] = exact_storage_term
                    sy_terms[period_index] = exact_sy_term
                    ss_terms[period_index] = exact_ss_term
                    sy_crossing_terms[period_index] = exact_sy_term
                period_infos.append(info_out)
                last_info = info_out
                head_prev = head_arr

        info_all = {
            "heads_per_period": heads_per_period,
            "heads_final": heads_per_period[-1],
            "period_infos": period_infos,
            "last_info": last_info,
            "period_times": period_times,
            "total_time": float(time.perf_counter() - total_t0),
            "n_periods": n_periods,
            "storage_reference": storage_reference,
            "dt": dt_f,
            "solve_controls": controls,
            "save_diagnostics": bool(save_diagnostics_b),
            "transient_replay_counters": counters,
        }
        if save_diagnostics_b:
            info_all.update(
                {
                    "heads_old_per_period": heads_old_per_period,
                    "storage_reference_heads_per_period": storage_reference_heads,
                    "storage_coeffs_per_period": storage_coeffs,
                    "sy_storage_coeffs_per_period": sy_coeffs,
                    "ss_storage_coeffs_per_period": ss_coeffs,
                    "storage_terms_per_period": storage_terms,
                    "sy_storage_terms_per_period": sy_terms,
                    "ss_storage_terms_per_period": ss_terms,
                    "sy_crossing_volume_terms_per_period": sy_crossing_terms,
                }
            )
        self._transient_replay_counters = dict(counters)
        return (heads_per_period, info_all) if return_info else heads_per_period

    def solve(
        self,
        formulation: str = "confined",
        solver: str | None = None,
        initial_head: np.ndarray | None = None,
        K_field: np.ndarray | None = None,
        zbot_field: np.ndarray | None = None,
        ztop_field: np.ndarray | None = None,
        return_info: bool = True,
        transient: bool = False,
        storage_coeff=None,
        dt=None,
        head_prev=None,
        refresh_diag_with_transient_storage: bool = True,
        **kwargs,
    ):
        """
        Solve the assembled 2D groundwater flow problem.

        Parameters
        ----------
        formulation:
            ``"confined"`` for the fixed-transmissivity 5-point operator or
            ``"unconfined"`` for the Picard saturated-thickness update path.
        solver:
            Optional solver selector. ``"kcycle"`` runs the multigrid K-cycle
            path and supports confined and unconfined transient solves.
            ``"pcg"`` is confined steady-state only; transient PCG calls raise
            ``NotImplementedError`` so storage terms cannot be silently ignored.
        initial_head:
            Optional starting head field. For transient solves, this is also
            used as the previous-time head when ``head_prev`` is not supplied.
        K_field, zbot_field, ztop_field:
            Hydraulic conductivity, aquifer bottom, and optional aquifer top
            fields for unconfined solves.
        transient:
            If true, add backward-Euler storage to the K-cycle operator and RHS.
        storage_coeff:
            Scalar or ``(ny, nx)`` storage coefficient. The solver forms
            ``storage_coeff * dx**2 / dt`` on active non-boundary cells.
        dt:
            Positive transient time-step length in the same time unit used by
            the storage coefficient and source terms.
        head_prev:
            Previous-time head field for the storage RHS term. Boundary cells
            are overwritten with fixed boundary values before use.
        refresh_diag_with_transient_storage:
            Rebuilds diagonal/hierarchy state when the transient storage
            diagonal changes. Leave enabled unless the caller has already
            staged a compatible hierarchy.
        return_info:
            If true, return ``(head, info)``; otherwise return only heads.
        **kwargs:
            Additional solver controls forwarded to ``solve_multigrid_kcycle``
            or the steady PCG implementation.
        """
        form_mode = str(formulation).strip().lower()
        if form_mode not in {"confined", "unconfined"}:
            raise ValueError("formulation must be 'confined' or 'unconfined'.")

        solver_mode = self.solver_type if solver is None else str(solver)
        solver_mode = str(solver_mode).strip().lower()
        if solver_mode in {"multigrid", "mg"}:
            solver_mode = "kcycle"
        if solver_mode not in {"pcg", "kcycle"}:
            raise ValueError("solver must be 'pcg' or 'kcycle'.")
        if form_mode == "unconfined" and solver_mode != "kcycle":
            raise ValueError("2D unconfined solves currently require solver='kcycle'.")
        if solver_mode == "pcg" and bool(transient):
            raise NotImplementedError(
                "Transient storage is implemented for solver='kcycle' only; "
                "use solver='kcycle' for transient 2D solves."
            )

        if solver_mode == "pcg":
            head, info = self._solve_pcg_device_loop(
                max_iter=int(kwargs.pop("pcg_max_iter", kwargs.pop("max_iter", 250))),
                rel_tol=float(kwargs.pop("rel_tol", 5.0e-7)),
                abs_tol_min=float(kwargs.pop("abs_tol_min", 5.0e-7)),
                initial_head=initial_head,
                history_every=kwargs.pop("history_every", None),
            )
            if kwargs:
                raise TypeError(f"unused solve kwargs for solver='pcg': {sorted(kwargs.keys())}")
            if return_info:
                info_out = dict(info) if isinstance(info, dict) else {}
                info_out["formulation"] = "confined"
                return head, info_out
            return head

        head_info = self.solve_multigrid_kcycle(
            initial_head=initial_head,
            return_info=return_info,
            unconfined=(form_mode == "unconfined"),
            K_field=K_field,
            zbot_field=zbot_field,
            ztop_field=ztop_field,
            transient=transient,
            storage_coeff=storage_coeff,
            dt=dt,
            head_prev=head_prev,
            refresh_diag_with_transient_storage=refresh_diag_with_transient_storage,
            **kwargs,
        )
        if return_info:
            head, info = head_info
            info_out = dict(info) if isinstance(info, dict) else {}
            info_out["formulation"] = form_mode
            return head, info_out
        return head_info

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
        if hasattr(self, "_stage_G0"):
            self._stage_G0 = None
        if hasattr(self, "_stage_G0_host"):
            self._stage_G0_host = None
        if hasattr(self, "_stage_Gc_2lvl"):
            self._stage_Gc_2lvl = None
        if hasattr(self, "_stage_G_levels"):
            self._stage_G_levels = None
        # Storage-diagonal and transmissivity/M_inv staging buffers (Warp arrays
        # on CPU). These were previously leaked across solves because close()
        # only dropped the R/G staging siblings.
        if hasattr(self, "_stage_Sc_2lvl"):
            self._stage_Sc_2lvl = None
        if hasattr(self, "_stage_S_levels"):
            self._stage_S_levels = None
        if hasattr(self, "_stage_T0"):
            self._stage_T0 = None
        if hasattr(self, "_stage_T0_host"):
            self._stage_T0_host = None
        if hasattr(self, "_stage_M0"):
            self._stage_M0 = None
        if hasattr(self, "_stage_M0_host"):
            self._stage_M0_host = None
        if hasattr(self, "_stage_Tc_2lvl"):
            self._stage_Tc_2lvl = None
        if hasattr(self, "_stage_Mc_2lvl"):
            self._stage_Mc_2lvl = None
        if hasattr(self, "_stage_T_levels"):
            self._stage_T_levels = None
        if hasattr(self, "_stage_M_levels"):
            self._stage_M_levels = None

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
        self._mg_coarsening_diagnostics = []
        self._two_level_coarsening_diag = None

        # 6) Drop all device arrays on the solver itself
        self.T_wp = None
        self.R_wp = None
        self.active_wp = None
        self.bc_mask_wp = None
        self.bc_values_wp = None
        self.gh_mask_wp = None
        self.gh_head_wp = None
        self.gh_width_wp = None
        self.storage_diag_wp = None

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
        self.ghb_factor_c_wp = None
        self.storage_diag_c_wp = None
        self.M_inv_c_wp = None

        # Storage-diagonal host mirrors (numpy). mg_levels already dropped the
        # per-level copies above; release the solver-level mirrors too.
        self.storage_diag_host = None
        self.storage_diag_c_host = None

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
