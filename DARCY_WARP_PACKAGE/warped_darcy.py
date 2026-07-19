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
from DARCY_WARP_PACKAGE.physics.operator_data import (
    BoundaryFields,
    GridSpec,
    OperatorFields,
    StorageState,
    compute_ghb_factor_from_raw_fields as _physics_compute_ghb_factor_from_raw_fields,
    normalize_scalar_or_grid_to_shape as _physics_normalize_scalar_or_grid_to_shape,
)
from DARCY_WARP_PACKAGE.physics.budgets_2d import (
    compute_mass_balance_budget as _physics_compute_mass_balance_budget,
)
from DARCY_WARP_PACKAGE.physics.storage_2d import (
    exact_unconfined_storage_terms as _physics_exact_unconfined_storage_terms,
    secant_specific_storage_coeff as _physics_secant_specific_storage_coeff,
    secant_specific_yield_coeff as _physics_secant_specific_yield_coeff,
    specific_storage_potential as _physics_specific_storage_potential,
)
from DARCY_WARP_PACKAGE.solvers import (
    ConvergenceControls,
    MultigridHierarchy,
    SolverContext,
    SolverResourceOwner,
    SolverWorkspace,
    canonical_solver_name,
    solve_selected,
)
from DARCY_WARP_PACKAGE.solvers.transient_unconfined import solve_transient_unconfined
from DARCY_WARP_PACKAGE.solvers.pcg import solve_pcg_device_loop
from DARCY_WARP_PACKAGE.solvers.multigrid_kcycle import (
    solve_kcycle_device_buffers,
    solve_multigrid_kcycle_backend,
)
from DARCY_WARP_PACKAGE.solvers.picard_unconfined import solve_unconfined_picard
from DARCY_WARP_PACKAGE.solvers.convergence import (
    chebyshev_relaxation_sequence as _solver_chebyshev_relaxation_sequence,
    chebyshev_update_weights as _solver_chebyshev_update_weights,
)
from DARCY_WARP_PACKAGE.solvers.hierarchy import (
    LinearGridLevel,
    MGLevel as SharedMGLevel,
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


_normalize_scalar_or_grid_to_shape = _physics_normalize_scalar_or_grid_to_shape


# Public compatibility aliases for storage relations extracted to physics.
specific_storage_potential = _physics_specific_storage_potential
secant_specific_yield_coeff = _physics_secant_specific_yield_coeff
secant_specific_storage_coeff = _physics_secant_specific_storage_coeff
exact_unconfined_storage_terms = _physics_exact_unconfined_storage_terms


_chebyshev_update_weights = _solver_chebyshev_update_weights
_chebyshev_relaxation_sequence = _solver_chebyshev_relaxation_sequence


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

_compute_ghb_factor_from_raw_fields = _physics_compute_ghb_factor_from_raw_fields


# Public compatibility alias for the extracted solver-level budget evaluator.
compute_mass_balance_budget = _physics_compute_mass_balance_budget


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


class WarpDarcySolver:
    """
    GPU based solver for 2D steady Darcy flow using Warp.
    Supports PCG and a 2-level multigrid V cycle (Jacobi on fine, PCG on coarse).
    """

    # Shared hierarchy data containers are not model state.
    _GridLevel = LinearGridLevel
    # The hierarchy level container is shared infrastructure, not model state.
    _MGLevel = SharedMGLevel

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
        # The model is the sole owner of Warp arrays, the hierarchy, and graph
        # cache. Backends only receive borrowed references through SolverContext.
        self._resource_owner = SolverResourceOwner(device=self.device_str)
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
        """Compatibility wrapper for the extracted PCG device implementation."""
        return solve_pcg_device_loop(
            model=self,
            max_iter=max_iter,
            rel_tol=rel_tol,
            abs_tol_min=abs_tol_min,
            initial_head=initial_head,
            history_every=history_every,
        )

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



    def _solve_multigrid_kcycle_device_buffers(self, *args, **kwargs):
        """Compatibility delegate for the extracted device-buffer K-cycle."""
        return solve_kcycle_device_buffers(model=self, *args, **kwargs)

    def _solve_multigrid_kcycle_backend(
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
        backend_kwargs = locals()
        backend_kwargs.pop("self")
        return solve_multigrid_kcycle_backend(model=self, **backend_kwargs)

    def _make_solver_context(self, *, formulation: str, transient: bool) -> SolverContext:
        """Expose model-owned numerical resources through the solver boundary.

        The callbacks intentionally target the pre-refactor implementations.
        They retain the same Warp arrays, hierarchy, CUDA graph, and execution
        order; the backend boundary therefore has no allocation or transfer
        cost.  Private backend hooks remain only while algorithm bodies are
        progressively separated from this model container.
        """
        self._resource_owner.refresh(
            hierarchy=self.mg_levels,
            work=self._mg_work,
            cuda_graph=self._kcycle_graph,
        )
        return SolverContext(
            grid=GridSpec(
                nx=int(self.nx),
                ny=int(self.ny),
                dx=float(self.dx),
                device=self.device_str,
            ),
            fields=OperatorFields(
                transmissivity=self.T_wp,
                recharge=self.R_wp,
                head=self.x_wp,
                rhs=self.b_wp,
            ),
            boundaries=BoundaryFields(
                active=self.active_wp,
                dirichlet_mask=self.bc_mask_wp,
                dirichlet_values=self.bc_values_wp,
                ghb_mask=self.gh_mask_wp,
                ghb_factor=self.ghb_factor_wp,
            ),
            storage=StorageState(
                diagonal=self.storage_diag_wp,
                active=bool(self._storage_active),
            ),
            hierarchy=MultigridHierarchy(
                levels=self.mg_levels,
                work=self._mg_work,
                coarsening_diagnostics=self._mg_coarsening_diagnostics,
            ),
            workspace=SolverWorkspace(
                pcg_buffers={
                    "x": self.x_wp,
                    "b": self.b_wp,
                    "r": self.r_wp,
                    "z": self.z_wp,
                    "p": self.p_wp,
                    "Ap": self.Ap_wp,
                },
                cuda_graph=self._kcycle_graph,
                transient_replay_counters=self._transient_replay_counters,
            ),
            convergence=ConvergenceControls(
                formulation=str(formulation),
                transient=bool(transient),
            ),
            model=self,
        )

    def solve_multigrid_kcycle(self, *args, **kwargs):
        """Compatibility entry point for the K-cycle family.

        New callers should use :meth:`solve` with ``confined_kcycle`` or
        ``unconfined_picard_kcycle``.  Positional calls retain the exact legacy
        invocation path because their argument mapping predates the registry.
        """
        if args:
            return self._solve_multigrid_kcycle_backend(*args, **kwargs)
        unconfined = bool(kwargs.get("unconfined", False))
        formulation = "unconfined" if unconfined else "confined"
        context = self._make_solver_context(
            formulation=formulation,
            transient=bool(kwargs.get("transient", False)),
        )
        backend_name = (
            "unconfined_picard_kcycle" if unconfined else "confined_kcycle"
        )
        return solve_selected(
            context,
            solver=backend_name,
            default=backend_name,
            **kwargs,
        )

    def solve_transient_2d_unconfined(
        self,
        *args,
        solver: str | None = "unconfined_picard_kcycle",
        **kwargs,
    ):
        """Run the registered production transient-unconfined backend.

        The public keyword-only contract is unchanged.  The extracted driver
        module owns the backend boundary while this compatibility wrapper keeps
        model construction and resource ownership local to the model.
        """
        if args:
            raise TypeError("solve_transient_2d_unconfined accepts keyword arguments only.")
        context = self._make_solver_context(formulation="unconfined", transient=True)
        return solve_transient_unconfined(context, solver=solver, **kwargs)

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
            Optional backend selector. ``"confined_pcg"``,
            ``"confined_kcycle"``, ``"unconfined_picard_kcycle"``, and the
            experimental ``"unconfined_semismooth_newton_kcycle"`` and
            ``"unconfined_fas"`` are
            explicit names. Legacy ``"pcg"``, ``"kcycle"``, ``"multigrid"``,
            and ``"mg"`` remain supported. PCG is confined steady-state only;
            transient PCG calls raise ``NotImplementedError`` so storage terms
            cannot be silently ignored.
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

        backend_name = canonical_solver_name(
            solver,
            formulation=form_mode,
            default=str(self.solver_type),
        )
        context = self._make_solver_context(
            formulation=form_mode,
            transient=bool(transient),
        )
        if backend_name == "confined_pcg":
            head_info = solve_selected(
                context,
                solver=backend_name,
                default=backend_name,
                initial_head=initial_head,
                **kwargs,
            )
        else:
            head_info = solve_selected(
                context,
                solver=backend_name,
                default=backend_name,
                initial_head=initial_head,
                return_info=return_info,
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
            info_out["solver_backend"] = backend_name
            return head, info_out
        if backend_name == "confined_pcg":
            return head_info[0]
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

        # Keep ownership accounting in step with the released model fields.
        # Backends never own these resources and cannot release them directly.
        if getattr(self, "_resource_owner", None) is not None:
            self._resource_owner.release()

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
