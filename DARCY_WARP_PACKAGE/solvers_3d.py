# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import gc

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.kernels_3d import (
    add_correction_3d_kernel,
    apply_A_and_pAp_7point_kernel,
    axpy_active_scalar_3d_kernel,
    build_diag_preconditioner_7point_kernel,
    copy_field_3d_kernel,
    compute_residual_7point_kernel,
    compute_head_residual_7point_kernel,
    dh_change_reduce_3d_kernel,
    dot_active_3d_kernel,
    jacobi_applyA_fused_7point_kernel,
    prolong_bilinear_axes_3d_kernel,
    prolong_bilinear_xy_3d_kernel,
    restrict_blockavg_axes_3d_kernel,
    restrict_blockavg_xy_3d_kernel,
    zero_scalar_kernel,
)
from DARCY_WARP_PACKAGE.config import NP_FLOAT, WP_FLOAT
from DARCY_WARP_PACKAGE.warped_darcy import (
    _chebyshev_relaxation_sequence,
    _chebyshev_update_weights,
)


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


def _resolve_diag_backend(backend: str, device: str) -> str:
    """
    Resolve a diag_preconditioner_backend selection to a concrete 'host'/'device'.

    Mirrors WarpDarcySolver._diag_backend_env_or_default /
    _select_diag_preconditioner_backend: 'auto' selects 'device' on CUDA,
    'host' otherwise. Honours the DARCY_M_INV_BACKEND env override.
    """
    mode = str(backend).strip().lower()
    if mode not in {"auto", "host", "device"}:
        env = str(__import__("os").environ.get("DARCY_M_INV_BACKEND", "")).strip().lower()
        if env in {"auto", "host", "device"}:
            mode = env
        else:
            mode = "auto"
    if mode == "host":
        return "host"
    if mode == "device":
        return "device"
    return "device" if str(device).startswith("cuda") else "host"


def _fill_m_inv_wp_7point(
    level_wp_arrays: dict,
    m_inv_wp,
    dim,
    nx: int,
    ny: int,
    nz: int,
    device: str,
) -> None:
    """
    Populate a device M_inv array via the device-side diag kernel, using the
    already-uploaded conductance/active/bc/storage Warp arrays in ``level_wp_arrays``.
    """
    wp.launch(
        kernel=build_diag_preconditioner_7point_kernel,
        dim=dim,
        inputs=[
            level_wp_arrays["tx_p_wp"],
            level_wp_arrays["tx_m_wp"],
            level_wp_arrays["ty_p_wp"],
            level_wp_arrays["ty_m_wp"],
            level_wp_arrays["tz_p_wp"],
            level_wp_arrays["tz_m_wp"],
            level_wp_arrays["active_wp"],
            level_wp_arrays["bc_mask_wp"],
            level_wp_arrays["storage_wp"],
            m_inv_wp,
            int(nx),
            int(ny),
            int(nz),
        ],
        device=device,
    )


def _release_mg_levels_3d(levels: list[dict]) -> None:
    """
    Release the device arrays held by a 3D multigrid level list and clear the
    list, so the unconfined Picard loop (which rebuilds levels every outer
    iteration) does not accumulate GPU memory. Mirrors the 2D MG-hierarchy
    leak fix.
    """
    if not levels:
        return
    for lvl in levels:
        if not isinstance(lvl, dict):
            continue
        for key, val in list(lvl.items()):
            if isinstance(val, wp.array):
                try:
                    val.release()
                except Exception:
                    pass
                lvl[key] = None
    levels.clear()
    gc.collect()


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
    Prepare RHS and storage diagonal for an optional confined 3D transient step.

    The transient term is backward Euler on active non-boundary cells:
    ``storage_diag += storage_coeff * dx * dy * dz / dt`` and
    ``rhs += storage_diag_add * head_prev``. Fixed-head boundary cells use
    boundary values in the previous-head field and receive zero storage
    contribution; inactive cells also receive zero storage contribution.

    Existing ``storage_diag`` values are preserved and incremented by the
    transient storage term. The returned arrays are host-side copies suitable
    for upload to the Warp kernels.
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


def _coarsen_mean_edge_axes(field_f: np.ndarray, *, coarsen_y: int, coarsen_x: int) -> np.ndarray:
    """
    Horizontally semi-coarsen a 3D field while preserving layer index.

    This is used for layered groundwater multigrid levels where the vertical
    axis represents model layers, not a smooth geometric continuum.  Values are
    averaged over 2x2 horizontal blocks independently within each layer.  For
    vertical face conductances this is a transparent first-order aggregation:
    the coarse vertical coupling is the arithmetic mean of the fine vertical
    couplings in the horizontal block, rather than a hidden full 2x2x2 merge.
    """
    if int(coarsen_y) not in {1, 2} or int(coarsen_x) not in {1, 2}:
        raise ValueError("coarsen_y and coarsen_x must be 1 or 2.")
    arr_f = np.asarray(field_f, dtype=NP_FLOAT)
    nz_f, ny_f, nx_f = arr_f.shape
    nz_c = nz_f
    fy = int(coarsen_y)
    fx = int(coarsen_x)
    ny_c = (ny_f + fy - 1) // fy
    nx_c = (nx_f + fx - 1) // fx

    pad_z = 0
    pad_y = int(fy * ny_c - ny_f)
    pad_x = int(fx * nx_c - nx_f)

    arr_p = np.pad(arr_f, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="edge")
    arr_c = arr_p.reshape(nz_c, ny_c, fy, nx_c, fx).mean(axis=(2, 4), dtype=np.float64)
    return arr_c.astype(NP_FLOAT, copy=False)


def _coarsen_max_edge_axes(mask_f: np.ndarray, *, coarsen_y: int, coarsen_x: int) -> np.ndarray:
    """
    Horizontally semi-coarsen a 3D mask while preserving layer index.
    """
    if int(coarsen_y) not in {1, 2} or int(coarsen_x) not in {1, 2}:
        raise ValueError("coarsen_y and coarsen_x must be 1 or 2.")
    arr_f = np.asarray(mask_f, dtype=np.int32)
    nz_f, ny_f, nx_f = arr_f.shape
    nz_c = nz_f
    fy = int(coarsen_y)
    fx = int(coarsen_x)
    ny_c = (ny_f + fy - 1) // fy
    nx_c = (nx_f + fx - 1) // fx

    pad_z = 0
    pad_y = int(fy * ny_c - ny_f)
    pad_x = int(fx * nx_c - nx_f)

    arr_p = np.pad(arr_f, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="edge")
    arr_c = arr_p.reshape(nz_c, ny_c, fy, nx_c, fx).max(axis=(2, 4))
    return arr_c.astype(np.int32, copy=False)


def _coarsen_mean_edge_1x2x2(field_f: np.ndarray) -> np.ndarray:
    return _coarsen_mean_edge_axes(field_f, coarsen_y=2, coarsen_x=2)


def _coarsen_max_edge_1x2x2(mask_f: np.ndarray) -> np.ndarray:
    return _coarsen_max_edge_axes(mask_f, coarsen_y=2, coarsen_x=2)


def _choose_horizontal_coarsening(ny: int, nx: int, min_coarse_n: int) -> tuple[int, int]:
    """Choose independent y/x factors, preserving any dimension that cannot coarsen."""
    minimum = int(min_coarse_n)
    if minimum < 1:
        raise ValueError("min_coarse_n must be >= 1.")
    fy = 2 if ((int(ny) + 1) // 2) >= minimum and int(ny) > 1 else 1
    fx = 2 if ((int(nx) + 1) // 2) >= minimum and int(nx) > 1 else 1
    return fy, fx


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
    diag_preconditioner_backend: str = "auto",
    return_info: bool = True,
):
    txp = np.asarray(tx_p, dtype=NP_FLOAT)
    txm = np.asarray(tx_m, dtype=NP_FLOAT)
    typ = np.asarray(ty_p, dtype=NP_FLOAT)
    tym = np.asarray(ty_m, dtype=NP_FLOAT)
    tzp = np.asarray(tz_p, dtype=NP_FLOAT)
    tzm = np.asarray(tz_m, dtype=NP_FLOAT)
    b = np.asarray(rhs, dtype=NP_FLOAT)
    act = np.asarray(active, dtype=np.int32)
    bcm = np.asarray(bc_mask, dtype=np.int32)
    bcv = np.asarray(bc_values, dtype=NP_FLOAT)

    shape = txp.shape
    if txp.ndim != 3:
        raise ValueError("7-point arrays must be 3D with shape (nz, ny, nx).")
    for name, arr in (
        ("tx_m", txm),
        ("ty_p", typ),
        ("ty_m", tym),
        ("tz_p", tzp),
        ("tz_m", tzm),
        ("rhs", b),
        ("active", act),
        ("bc_mask", bcm),
        ("bc_values", bcv),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} expected {shape}")

    free = (act != 0) & (bcm == 0)
    n_free = int(np.count_nonzero(free))

    if n_free <= 0:
        if initial_head is None:
            h0 = np.zeros(shape, dtype=NP_FLOAT)
        else:
            h0 = np.asarray(initial_head, dtype=NP_FLOAT).copy()
            if h0.shape != shape:
                raise ValueError(f"initial_head shape {h0.shape} expected {shape}")
        h0[bcm != 0] = bcv[bcm != 0]
        h0[act == 0] = NP_FLOAT(0.0)
        info0 = {
            "solver_type": "chebyshev_7point_3d",
            "n_iter_used": 0,
            "converged": True,
            "r_rms0": 0.0,
            "r_rms_end": 0.0,
            "tol_abs": float(abs_tol_min),
            "transient": bool(transient),
            "transient_formulation": "confined" if bool(transient) else "steady",
            "dt": float(dt) if bool(transient) and dt is not None else float("nan"),
            "unconfined": False,
        }
        return (h0, info0) if return_info else h0

    b_eff, sdiag, h_prev_used, dt_used = _prepare_7point_transient_terms(
        rhs=b,
        storage_diag=storage_diag,
        active=act,
        bc_mask=bcm,
        bc_values=bcv,
        transient=bool(transient),
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        initial_head=initial_head,
        dx=float(dx),
        dy=dy,
        dz=float(dz),
    )

    if initial_head is None:
        if bool(transient) and h_prev_used is not None:
            x0 = np.asarray(h_prev_used, dtype=NP_FLOAT).copy()
        else:
            x0 = np.zeros(shape, dtype=NP_FLOAT)
    else:
        x0 = np.asarray(initial_head, dtype=NP_FLOAT).copy()
        if x0.shape != shape:
            raise ValueError(f"initial_head shape {x0.shape} expected {shape}")
        if not np.all(np.isfinite(x0)):
            raise ValueError("initial_head must be finite.")

    x0[bcm != 0] = bcv[bcm != 0]
    x0[act == 0] = NP_FLOAT(0.0)

    M_inv = build_diag_preconditioner_7point(
        tx_p=txp,
        tx_m=txm,
        ty_p=typ,
        ty_m=tym,
        tz_p=tzp,
        tz_m=tzm,
        active=act,
        bc_mask=bcm,
        storage_diag=sdiag,
    )

    omegas = _chebyshev_relaxation_sequence(
        order=int(cheby_order),
        lambda_min=float(cheby_lambda_min),
        lambda_max=float(cheby_lambda_max),
    )
    if len(omegas) == 0:
        omegas = (1.0,)

    nz, ny, nx = shape
    dim = (nz, ny, nx)
    diag_mode = _resolve_diag_backend(diag_preconditioner_backend, device)

    txp_wp = wp.array(txp, dtype=WP_FLOAT, device=device)
    txm_wp = wp.array(txm, dtype=WP_FLOAT, device=device)
    typ_wp = wp.array(typ, dtype=WP_FLOAT, device=device)
    tym_wp = wp.array(tym, dtype=WP_FLOAT, device=device)
    tzp_wp = wp.array(tzp, dtype=WP_FLOAT, device=device)
    tzm_wp = wp.array(tzm, dtype=WP_FLOAT, device=device)
    b_wp = wp.array(b_eff, dtype=WP_FLOAT, device=device)
    act_wp = wp.array(act, dtype=wp.int32, device=device)
    bcm_wp = wp.array(bcm, dtype=wp.int32, device=device)
    bcv_wp = wp.array(bcv, dtype=WP_FLOAT, device=device)
    sdiag_wp = wp.array(sdiag, dtype=WP_FLOAT, device=device)
    if diag_mode == "device":
        M_inv_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
        _fill_m_inv_wp_7point(
            {
                "tx_p_wp": txp_wp,
                "tx_m_wp": txm_wp,
                "ty_p_wp": typ_wp,
                "ty_m_wp": tym_wp,
                "tz_p_wp": tzp_wp,
                "tz_m_wp": tzm_wp,
                "active_wp": act_wp,
                "bc_mask_wp": bcm_wp,
                "storage_wp": sdiag_wp,
            },
            M_inv_wp,
            dim,
            int(nx),
            int(ny),
            int(nz),
            device,
        )
    else:
        M_inv_wp = wp.array(M_inv, dtype=WP_FLOAT, device=device)

    x_wp = wp.array(x0, dtype=WP_FLOAT, device=device)
    x_tmp_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
    r_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
    rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)

    def _compute_rms(x_arr_wp) -> tuple[float, float]:
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rTr_buf], device=device)
        wp.launch(
            kernel=compute_residual_7point_kernel,
            dim=dim,
            inputs=[
                x_arr_wp,
                b_wp,
                txp_wp,
                txm_wp,
                typ_wp,
                tym_wp,
                tzp_wp,
                tzm_wp,
                act_wp,
                bcm_wp,
                sdiag_wp,
                r_wp,
                rTr_buf,
                int(nx),
                int(ny),
                int(nz),
            ],
            device=device,
        )
        rtr = float(rTr_buf.numpy()[0])
        return rtr, float(np.sqrt(max(rtr, 0.0) / float(n_free)))

    _rTr0, r_rms0 = _compute_rms(x_wp)
    tol_abs = float(max(abs_tol_min, rel_tol * r_rms0))

    converged = r_rms0 <= tol_abs
    n_iter_used = 0
    r_rms_end = float(r_rms0)

    for it in range(int(max_iter)):
        if converged:
            break

        n_iter_used = it + 1
        x_in = x_wp
        x_out = x_tmp_wp

        for omega_step in omegas:
            wp.launch(
                kernel=jacobi_applyA_fused_7point_kernel,
                dim=dim,
                inputs=[
                    txp_wp,
                    txm_wp,
                    typ_wp,
                    tym_wp,
                    tzp_wp,
                    tzm_wp,
                    act_wp,
                    bcm_wp,
                    sdiag_wp,
                    b_wp,
                    x_in,
                    M_inv_wp,
                    bcv_wp,
                    float(omega_step),
                    int(nx),
                    int(ny),
                    int(nz),
                    x_out,
                ],
                device=device,
            )
            tmp = x_in
            x_in = x_out
            x_out = tmp

        x_wp = x_in
        x_tmp_wp = x_out

        _, r_rms_end = _compute_rms(x_wp)
        if r_rms_end <= tol_abs:
            converged = True
            break

    head_out = np.asarray(x_wp.numpy(), dtype=NP_FLOAT)

    # Head-equivalent (Jacobi-preconditioned) residual RMS for reporting / inner-usable checks.
    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rTr_buf], device=device)
    wp.launch(
        kernel=compute_head_residual_7point_kernel,
        dim=dim,
        inputs=[
            x_wp,
            b_wp,
            txp_wp,
            txm_wp,
            typ_wp,
            tym_wp,
            tzp_wp,
            tzm_wp,
            act_wp,
            bcm_wp,
            sdiag_wp,
            M_inv_wp,
            r_wp,
            rTr_buf,
            int(nx),
            int(ny),
            int(nz),
        ],
        device=device,
    )
    hrTr_end = float(rTr_buf.numpy()[0])
    h_rms_end = float(np.sqrt(max(hrTr_end, 0.0) / float(n_free)))

    info = {
        "solver_type": "chebyshev_7point_3d",
        "n_iter_used": int(n_iter_used),
        "n_cycles_used": int(n_iter_used),
        "max_iter": int(max_iter),
        "cheby_order": int(len(omegas)),
        "cheby_omegas": [float(v) for v in omegas],
        "r_rms0": float(r_rms0),
        "r_rms_start": float(r_rms0),
        "r_rms_end": float(r_rms_end),
        "h_rms_end": float(h_rms_end),
        "tol_abs": float(tol_abs),
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "dh_rms_lastcheck": None,
        "dh_max_lastcheck": None,
        "transient": bool(transient),
        "transient_formulation": "confined" if bool(transient) else "steady",
        "dt": float(dt_used) if bool(transient) else float("nan"),
        "unconfined": False,
        "converged": bool(converged),
        "diag_preconditioner_backend": diag_mode,
    }

    return (head_out, info) if return_info else head_out


def _picard_unconfined_7point_3d(
    inner_solve,
    *,
    shape,
    active,
    bc_mask,
    bc_values,
    rhs,
    kx,
    ky,
    kz,
    zbot,
    ztop,
    initial_head,
    storage_diag,
    min_sat,
    max_outer,
    pic_relax,
    pic_tol,
    omega_min,
    omega_max,
    dx,
    dy,
    dz,
    device,
    transient=False,
    dt=None,
    sy=None,
    ss=None,
    head_prev=None,
    unconfined_storage_mode="phreatic_sy",
    unconfined_startup_mode="initial_head",
    transmissivity_relaxation_enabled=False,
    transmissivity_relaxation_early=0.25,
    transmissivity_relaxation_middle=0.50,
    transmissivity_relaxation_late=1.00,
    transmissivity_relaxation_middle_iteration=5,
    transmissivity_relaxation_late_iteration=15,
    inner_forcing_eta=0.10,
    inner_head_residual_tol_min=None,
    inner_head_residual_tol_max=1.0e-2,
    inner_picard_scale_max_fraction=0.10,
    chebyshev_enabled=True,
    chebyshev_order=3,
    chebyshev_lambda_min_fraction=0.1,
    chebyshev_reset_on_residual_increase=True,
    chebyshev_reset_factor=1.2,
    chebyshev_minor_increase_patience=2,
    chebyshev_rejection_factor=1.2,
    unconfined_inner_max_cycles_early=10,
    unconfined_inner_max_cycles_middle=25,
    unconfined_inner_max_cycles_late=60,
    unconfined_inner_late_dh=1.0e-2,
    unconfined_inner_middle_dh=1.0,
    max_head_change_per_outer_iteration=10.0,
    residual_floor_tol=1.0e-4,
    dh_rms_tol=1.0e-4,
    diag_preconditioner_backend="auto",
    linear_solver_type_label="kcycle_7point_3d",
    solver_type_label="kcycle_7point_3d_unconfined_picard",
    dry_cell_flag_threshold=0.1,
    return_info=True,
):
    """
    Shared unconfined Picard driver for the 3D 7-point solvers.

    This is a faithful 3D port of the 2D unconfined nonlinear loop in
    ``warped_darcy.py`` (``solve_multigrid_kcycle`` unconfined block). It accepts
    an ``inner_solve(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m, initial_head, max_cycles)``
    closure so both the K-cycle and standalone-Chebyshev backends share one
    implementation of the speed/convergence controls:

      - ``unconfined_startup_mode`` ("initial_head" / "confined_pre_solve"),
      - adaptive inner ``max_cycles`` (early/middle/late),
      - ``transmissivity_relaxation_enabled`` (saturation under-relaxation),
      - dynamic inexact inner tolerance (``inner_forcing_eta`` + min/max bounds),
      - outer Chebyshev acceleration (``chebyshev_enabled``) with reset logic
        (``chebyshev_reset_factor``),
      - ``diag_preconditioner_backend`` (forwarded to the inner solve).

    Returns ``(h_iter, info_out)`` where ``info_out`` carries both the nonlinear
    (Picard) and the last inner-solve (K-cycle) reporting fields.
    """
    act = np.asarray(active, dtype=np.int32)
    bcm = np.asarray(bc_mask, dtype=np.int32)
    bcv = np.asarray(bc_values, dtype=NP_FLOAT)
    b = np.asarray(rhs, dtype=NP_FLOAT)
    active_mask = act != 0
    bc_mask0 = bcm != 0
    free_mask = active_mask & (~bc_mask0)
    n_free = int(np.count_nonzero(free_mask))

    kx64 = np.asarray(kx, dtype=np.float64)
    ky64 = np.asarray(ky, dtype=np.float64)
    kz64 = np.asarray(kz, dtype=np.float64)
    zbot64 = np.asarray(zbot, dtype=np.float64)
    ztop64 = None if ztop is None else np.asarray(ztop, dtype=np.float64)

    hclose_f = float(pic_tol)
    if hclose_f < 0.0 or not np.isfinite(hclose_f):
        raise ValueError("unconfined_head_tol must be non-negative and finite.")

    omega_min_f = float(omega_min)
    omega_max_f = float(omega_max)
    if not (0.0 < omega_min_f <= omega_max_f):
        raise ValueError("omega_min and omega_max must satisfy 0 < omega_min <= omega_max.")
    omega_current = min(max(float(pic_relax), omega_min_f), omega_max_f)

    residual_floor_tol_f = None if residual_floor_tol is None else float(residual_floor_tol)
    if residual_floor_tol_f is not None and residual_floor_tol_f < 0.0:
        raise ValueError("residual_floor_tol must be non-negative.")
    dh_rms_tol_f = None if dh_rms_tol is None else float(dh_rms_tol)

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
    rejection_factor_f = float(chebyshev_rejection_factor)
    if rejection_factor_f <= 1.0 or not np.isfinite(rejection_factor_f):
        raise ValueError("chebyshev_rejection_factor must be finite and > 1.")

    max_update_f = float(max_head_change_per_outer_iteration)
    if max_update_f <= 0.0 or not np.isfinite(max_update_f):
        raise ValueError("max_head_change_per_outer_iteration must be positive and finite.")

    inner_max_cycles_early = int(unconfined_inner_max_cycles_early)
    inner_max_cycles_middle = int(unconfined_inner_max_cycles_middle)
    inner_max_cycles_late = int(unconfined_inner_max_cycles_late)
    if min(inner_max_cycles_early, inner_max_cycles_middle, inner_max_cycles_late) < 1:
        raise ValueError("unconfined inner max cycles must be >= 1.")
    inner_late_dh_f = float(unconfined_inner_late_dh)
    inner_middle_dh_f = float(unconfined_inner_middle_dh)
    if inner_late_dh_f < 0.0 or inner_middle_dh_f < 0.0:
        raise ValueError("unconfined inner dh thresholds must be non-negative.")

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
    if startup_mode not in {"initial_head", "confined_pre_solve"}:
        raise ValueError("unconfined_startup_mode must be 'initial_head' or 'confined_pre_solve'.")

    # --- Transient unconfined storage configuration -----------------------
    # Two explicit storage modes for transient unconfined 3D:
    #   * "phreatic_sy" - specific-yield (water-table) storage coupled to the
    #     Picard saturated-thickness update: Sy*dx*dy/dt on the per-column
    #     water-table cell plus Ss*sat*dx*dy/dt on saturated cells. This is
    #     area-based (NOT full-cell-volume) and is the physically meaningful
    #     unconfined term, recomputed every outer iteration from h_iter.
    #   * "confined_volume" - legacy first-order approximation: the inner
    #     confined-transient solve applies storage_coeff*dx*dy*dz/dt over the
    #     full cell volume. Kept for backward compatibility / comparison only.
    storage_mode = str(unconfined_storage_mode).strip().lower()
    if storage_mode not in {"phreatic_sy", "confined_volume"}:
        raise ValueError(
            "unconfined_storage_mode must be 'phreatic_sy' or 'confined_volume'."
        )

    def _broadcast_storage_field(value, name: str) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape == ():
            return np.full(shape, float(arr.reshape(()).item()), dtype=np.float64)
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} expected {shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite.")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be >= 0.")
        return arr.astype(np.float64, copy=True)

    def _storage_param_summary(value) -> float | str | None:
        if value is None:
            return None
        arr = np.asarray(value)
        return float(arr.reshape(()).item()) if arr.shape == () else "field"

    sy_field = _broadcast_storage_field(sy, "sy")
    ss_field = _broadcast_storage_field(ss, "ss")
    sy_summary = _storage_param_summary(sy)
    ss_summary = _storage_param_summary(ss)
    # Phreatic Sy storage is active only for a transient phreatic_sy solve with
    # a supplied specific yield. Otherwise the inner solve uses the legacy
    # confined-volume path (or is steady).
    phreatic_active = (
        bool(transient)
        and storage_mode == "phreatic_sy"
        and sy_field is not None
    )

    h_prev_storage: np.ndarray | None = None
    if phreatic_active:
        if dt is None or not np.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("phreatic_sy transient storage requires dt > 0.")
        if head_prev is not None:
            h_prev_storage = np.asarray(head_prev, dtype=NP_FLOAT).copy()
        elif initial_head is not None:
            h_prev_storage = np.asarray(initial_head, dtype=NP_FLOAT).copy()
        else:
            h_prev_storage = np.full(
                shape, float(zbot64.min()) + float(min_sat), dtype=NP_FLOAT
            )
        if h_prev_storage.shape != shape:
            raise ValueError(f"head_prev shape {h_prev_storage.shape} expected {shape}.")
        h_prev_storage[bc_mask0] = bcv[bc_mask0]
        h_prev_storage[~active_mask] = NP_FLOAT(0.0)
        if not np.all(np.isfinite(h_prev_storage)):
            raise ValueError("head_prev contains non-finite values.")

    if initial_head is None:
        h_iter = (zbot64 + float(min_sat)).astype(NP_FLOAT, copy=False)
    else:
        h_iter = np.asarray(initial_head, dtype=NP_FLOAT).copy()
        if h_iter.shape != shape:
            raise ValueError(f"initial_head shape {h_iter.shape} expected {shape}.")
        if not np.all(np.isfinite(h_iter)):
            raise ValueError("initial_head must be finite.")
    h_iter[bc_mask0] = bcv[bc_mask0]
    h_iter[~active_mask] = NP_FLOAT(0.0)

    def _conductances_from_sat(sat_arr):
        ks = sat_arr.astype(np.float64, copy=False)
        return build_7point_face_conductance_from_k(
            kx_field=(kx64 * ks).astype(NP_FLOAT, copy=False),
            ky_field=(ky64 * ks).astype(NP_FLOAT, copy=False),
            kz_field=(kz64 * ks).astype(NP_FLOAT, copy=False),
            active=act,
            dx=float(dx),
            dy=float(dx) if dy is None else float(dy),
            dz=float(dz),
        )

    if startup_mode == "confined_pre_solve":
        sat_startup = np.maximum(h_iter.astype(np.float64, copy=False) - zbot64, float(min_sat))
        if ztop64 is not None:
            sat_startup = np.minimum(sat_startup, np.maximum(ztop64 - zbot64, float(min_sat)))
        txp_s, txm_s, typ_s, tym_s, tzp_s, tzm_s = _conductances_from_sat(sat_startup)
        h_startup, _ = inner_solve(txp_s, txm_s, typ_s, tym_s, tzp_s, tzm_s, h_iter, inner_max_cycles_late)
        h_startup = np.asarray(h_startup, dtype=np.float64)
        h_startup = np.maximum(h_startup, zbot64 + float(min_sat))
        if ztop64 is not None:
            h_startup = np.minimum(h_startup, ztop64)
        h_startup[~active_mask] = 0.0
        h_startup[bc_mask0] = bcv[bc_mask0]
        if not np.all(np.isfinite(h_startup)):
            raise FloatingPointError("confined pre-solve produced non-finite heads.")
        h_iter = h_startup.astype(NP_FLOAT, copy=False)

    cheb_weights = _chebyshev_update_weights(
        order=int(chebyshev_order),
        lambda_min_fraction=float(chebyshev_lambda_min_fraction),
    )
    previous_update = np.zeros(shape, dtype=np.float64)
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
    final_picard_dh_rms = float("nan")
    last_linear_info: dict = {}
    outer_history: list[dict] = []
    converged_nonlinear = False
    sat_prev: np.ndarray | None = None
    T_relax_used = float("nan")

    def _to_finite(value):
        try:
            f = float(value)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    for outer_idx in range(int(max_outer)):
        if not np.isfinite(previous_measure):
            inner_max_cycles = inner_max_cycles_early
        elif previous_measure > inner_middle_dh_f:
            inner_max_cycles = inner_max_cycles_early
        elif previous_measure > inner_late_dh_f:
            inner_max_cycles = inner_max_cycles_middle
        else:
            inner_max_cycles = inner_max_cycles_late

        sat = h_iter.astype(np.float64, copy=False) - zbot64
        sat = np.maximum(sat, float(min_sat))
        if ztop64 is not None:
            sat = np.minimum(sat, np.maximum(ztop64 - zbot64, float(min_sat)))
        if not np.all(np.isfinite(sat)) or np.any(sat <= 0.0):
            raise FloatingPointError("unconfined saturated thickness became invalid.")
        sat_candidate = sat

        if transmissivity_relaxation_enabled_b and outer_idx > 0 and sat_prev is not None:
            if outer_idx < T_relax_middle_iter:
                T_relax_used = T_relax_early_f
            elif outer_idx < T_relax_late_iter:
                T_relax_used = T_relax_middle_f
            else:
                T_relax_used = T_relax_late_f
            sat_use = (1.0 - T_relax_used) * sat_prev + T_relax_used * sat_candidate
        else:
            sat_use = sat_candidate
            T_relax_used = float("nan")
        sat_prev = sat_candidate.copy()

        txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i = _conductances_from_sat(sat_use)

        iter_storage_kwargs: dict = {}
        if phreatic_active:
            # Physical saturated thickness from the current iterate (0 if dry,
            # capped at the cell thickness). This couples the storage capacity
            # to the Picard saturated-thickness update.
            phys_sat = np.maximum(h_iter.astype(np.float64, copy=False) - zbot64, 0.0)
            if ztop64 is not None:
                phys_sat = np.minimum(phys_sat, np.maximum(ztop64 - zbot64, 0.0))
            saturated = active_mask & (phys_sat > 0.0)
            # Water-table cell per (j,i) column: the topmost (smallest layer
            # index) active cell with positive saturation. Layers are ordered
            # top->bottom, so this is the cell containing the phreatic surface.
            has_sat_col = saturated.any(axis=0)
            wt_layer = saturated.argmax(axis=0)
            lay_idx = np.arange(shape[0])[:, None, None]
            is_wt = has_sat_col[None, :, :] & (lay_idx == wt_layer[None, :, :])

            area_f = float(dx) * (float(dx) if dy is None else float(dy))
            dt_f = float(dt)
            ss_term = (
                ss_field * phys_sat * area_f / dt_f
                if ss_field is not None
                else np.zeros(shape, dtype=np.float64)
            )
            sy_term = np.where(
                is_wt,
                sy_field * area_f / dt_f,
                0.0,
            )
            storage_diag_iter = (ss_term + sy_term).astype(NP_FLOAT, copy=False)
            storage_diag_iter[~free_mask] = NP_FLOAT(0.0)
            # Backward-Euler RHS contribution from the previous time step.
            rhs_eff_iter = (
                b.astype(np.float64, copy=False)
                + storage_diag_iter.astype(np.float64, copy=False) * h_prev_storage
            ).astype(NP_FLOAT, copy=False)
            iter_storage_kwargs = {
                "storage_diag_iter": storage_diag_iter,
                "rhs_eff_iter": rhs_eff_iter,
            }

        head_lin, info_lin = inner_solve(
            txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i, h_iter, inner_max_cycles,
            **iter_storage_kwargs,
        )
        last_linear_info = dict(info_lin) if isinstance(info_lin, dict) else {}
        inner_converged = bool(last_linear_info.get("converged", False))

        h_lin = np.asarray(head_lin, dtype=np.float64)
        if h_lin.shape != shape:
            raise RuntimeError(f"inner linear solve returned shape {h_lin.shape}, expected {shape}.")
        picard_update = h_lin - h_iter.astype(np.float64, copy=False)

        if np.any(free_mask):
            picard_update_free_raw = picard_update[free_mask]
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
        h_trial[bc_mask0] = bcv[bc_mask0]
        h_trial[~active_mask] = 0.0

        if np.any(free_mask):
            trial_dh = (h_trial - h_iter.astype(np.float64, copy=False))[free_mask]
            trial_measure = float(np.max(np.abs(trial_dh)))
            trial_measure_rms = float(np.sqrt(np.mean(trial_dh * trial_dh)))
        else:
            trial_measure = 0.0
            trial_measure_rms = 0.0

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
            h_trial[bc_mask0] = bcv[bc_mask0]
            h_trial[~active_mask] = 0.0
            if np.any(free_mask):
                trial_dh = (h_trial - h_iter.astype(np.float64, copy=False))[free_mask]
                trial_measure = float(np.max(np.abs(trial_dh)))
                trial_measure_rms = float(np.sqrt(np.mean(trial_dh * trial_dh)))
            else:
                trial_measure = 0.0
                trial_measure_rms = 0.0

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

        previous_update[:, :, :] = clipped
        h_iter = h_trial.astype(NP_FLOAT, copy=False)
        final_max_abs_head_change = float(trial_measure)
        final_picard_dh_rms = float(trial_measure_rms)
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
                "previous_measure": float(previous_measure) if np.isfinite(previous_measure) else None,
                "clipped_update": bool(clipped_update),
                "transmissivity_relaxation_used": None if np.isnan(T_relax_used) else float(T_relax_used),
                "max_abs_head_change": float(final_max_abs_head_change),
                "inner_iterations": int(last_linear_info.get("n_cycles_used", last_linear_info.get("n_iter_used", 0))),
                "inner_residual": None if final_residual is None else float(final_residual),
            }
        )

        head_change_converged = final_max_abs_head_change < hclose_f
        if head_change_converged and inner_usable_for_picard:
            converged_nonlinear = True
            break

    effectively_dry = active_mask & (
        h_iter.astype(np.float64, copy=False) <= zbot64 + float(dry_cell_flag_threshold)
    )
    info_out = dict(last_linear_info) if isinstance(last_linear_info, dict) else {}
    # Carry the last inner-solve (K-cycle) linear fields to the top level so the
    # runner can report both nonlinear (Picard) and linear (K-cycle) convergence.
    for _ck in (
        "n_cycles_used",
        "r_rms_end",
        "h_rms_end",
        "tol_abs",
        "dh_rms_lastcheck",
        "dh_max_lastcheck",
    ):
        if _ck in last_linear_info:
            info_out[_ck] = last_linear_info[_ck]
    info_out.update(
        {
            "solver_type": str(solver_type_label),
            "linear_solver_type": str(last_linear_info.get("solver_type", linear_solver_type_label)),
            "unconfined": True,
            "transient": bool(transient),
            "transient_formulation": "unconfined" if bool(transient) else "steady",
            "dt": (
                float(dt)
                if bool(transient) and dt is not None and np.isfinite(float(dt))
                else float("nan")
            ),
            "unconfined_storage_mode": str(storage_mode),
            "phreatic_storage_active": bool(phreatic_active),
            "sy": sy_summary,
            "ss": ss_summary,
            "converged": bool(converged_nonlinear),
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
            "nonlinear_convergence_basis": "head_change_and_inner_usable_for_picard",
            "residual_floor_tol": None if residual_floor_tol_f is None else float(residual_floor_tol_f),
            "inner_residual_converged": bool(inner_residual_converged),
            "inner_head_change_converged": bool(inner_head_change_converged),
            "inner_practically_converged": bool(inner_practically_converged),
            "inner_usable_for_picard": bool(inner_usable_for_picard),
            "inner_h_rms_end": float(final_h_rms_end) if np.isfinite(final_h_rms_end) else None,
            "inner_max_cycles_used": int(final_inner_max_cycles),
            "outer_history": outer_history,
            "picard_converged": bool(converged_nonlinear),
            "picard_n_iter_used": int(len(outer_history)),
            "picard_max_iter": int(max_outer),
            "picard_relax": float(omega_current),
            "picard_head_tol": float(hclose_f),
            "picard_dh_rms_end": float(final_picard_dh_rms),
            "picard_dh_max_end": float(final_max_abs_head_change),
            "unconfined_min_sat": float(min_sat),
            "unconfined_startup_mode": str(startup_mode),
            "transmissivity_relaxation_enabled": bool(transmissivity_relaxation_enabled_b),
            "diag_preconditioner_backend": str(diag_preconditioner_backend),
            "r_rms_start": _to_finite(last_linear_info.get("r_rms_start", last_linear_info.get("r_rms0"))),
        }
    )
    return (h_iter, info_out) if return_info else h_iter


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
    sy: np.ndarray | float | None = None,
    ss: np.ndarray | float | None = None,
    unconfined_storage_mode: str = "phreatic_sy",
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
    ztop_field: np.ndarray | None = None,
    omega_min: float = 0.1,
    omega_max: float = 0.9,
    residual_floor_tol: float | None = 1.0e-4,
    dh_rms_tol: float | None = 1.0e-4,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float | None = None,
    inner_head_residual_tol_max: float = 1.0e-2,
    inner_picard_scale_max_fraction: float = 0.10,
    chebyshev_enabled: bool = True,
    chebyshev_order: int = 3,
    chebyshev_lambda_min_fraction: float = 0.1,
    chebyshev_reset_on_residual_increase: bool = True,
    chebyshev_reset_factor: float = 1.2,
    chebyshev_minor_increase_patience: int = 2,
    chebyshev_rejection_factor: float = 1.2,
    unconfined_inner_max_cycles_early: int = 10,
    unconfined_inner_max_cycles_middle: int = 25,
    unconfined_inner_max_cycles_late: int = 60,
    unconfined_inner_late_dh: float = 1.0e-2,
    unconfined_inner_middle_dh: float = 1.0,
    transmissivity_relaxation_enabled: bool = False,
    transmissivity_relaxation_early: float = 0.25,
    transmissivity_relaxation_middle: float = 0.50,
    transmissivity_relaxation_late: float = 1.00,
    transmissivity_relaxation_middle_iteration: int = 5,
    transmissivity_relaxation_late_iteration: int = 15,
    unconfined_startup_mode: str = "initial_head",
    max_head_change_per_outer_iteration: float = 10.0,
    dry_cell_flag_threshold: float = 0.1,
    diag_preconditioner_backend: str = "auto",
    device: str = "cuda:0",
    return_info: bool = True,
):
    txp = np.asarray(tx_p, dtype=NP_FLOAT)
    txm = np.asarray(tx_m, dtype=NP_FLOAT)
    typ = np.asarray(ty_p, dtype=NP_FLOAT)
    tym = np.asarray(ty_m, dtype=NP_FLOAT)
    tzp = np.asarray(tz_p, dtype=NP_FLOAT)
    tzm = np.asarray(tz_m, dtype=NP_FLOAT)
    b = np.asarray(rhs, dtype=NP_FLOAT)
    act = np.asarray(active, dtype=np.int32)
    bcm = np.asarray(bc_mask, dtype=np.int32)
    bcv = np.asarray(bc_values, dtype=NP_FLOAT)

    shape = txp.shape
    if txp.ndim != 3:
        raise ValueError("7-point arrays must be 3D with shape (nz, ny, nx).")
    for name, arr in (
        ("tx_m", txm),
        ("ty_p", typ),
        ("ty_m", tym),
        ("tz_p", tzp),
        ("tz_m", tzm),
        ("rhs", b),
        ("active", act),
        ("bc_mask", bcm),
        ("bc_values", bcv),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} expected {shape}")

    free = (act != 0) & (bcm == 0)
    n_free = int(np.count_nonzero(free))

    if bool(unconfined) and bool(transient):
        # Resolve the transient unconfined storage mode. When a specific yield
        # (sy) is supplied the phreatic water-table storage path is used
        # (Sy*dx*dy/dt on the per-column water-table cell + Ss*sat*dx*dy/dt on
        # saturated cells, recomputed every Picard iteration from h_iter).
        # When only storage_coeff is supplied the legacy confined-volume
        # approximation (storage_coeff*dx*dy*dz/dt over the full cell volume)
        # is retained for backward compatibility. Either way a clean storage
        # diagonal is forced so the inner solve rebuilds / replaces it on every
        # outer iteration (no double-counting of a caller-supplied diagonal).
        _storage_mode = str(unconfined_storage_mode).strip().lower()
        if _storage_mode not in {"phreatic_sy", "confined_volume"}:
            raise ValueError("unconfined_storage_mode must be 'phreatic_sy' or 'confined_volume'.")
        if _storage_mode == "phreatic_sy" and sy is None and storage_coeff is not None:
            # Caller used the legacy storage_coeff argument: keep the confined-
            # volume approximation so existing callers/tests behave as before.
            _storage_mode = "confined_volume"
        unconfined_storage_mode = _storage_mode
        storage_diag = None

    if bool(unconfined):
        if kx_field is None or ky_field is None or kz_field is None or zbot_field is None:
            raise ValueError("unconfined=True requires kx_field, ky_field, kz_field, and zbot_field.")

        kx = np.asarray(kx_field, dtype=NP_FLOAT)
        ky = np.asarray(ky_field, dtype=NP_FLOAT)
        kz = np.asarray(kz_field, dtype=NP_FLOAT)
        zbot = np.asarray(zbot_field, dtype=NP_FLOAT)
        for name, arr in (("kx_field", kx), ("ky_field", ky), ("kz_field", kz), ("zbot_field", zbot)):
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} expected {shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite.")
        if np.any(kx < NP_FLOAT(0.0)) or np.any(ky < NP_FLOAT(0.0)) or np.any(kz < NP_FLOAT(0.0)):
            raise ValueError("Unconfined K fields must be >= 0.")

        min_sat = float(unconfined_min_sat)
        if min_sat <= 0.0:
            raise ValueError("unconfined_min_sat must be positive.")
        n_pic = int(unconfined_max_picard_iter)
        if n_pic < 1:
            raise ValueError("unconfined_max_picard_iter must be >= 1.")
        pic_relax = float(unconfined_relax)
        if pic_relax <= 0.0 or pic_relax > 1.0:
            raise ValueError("unconfined_relax must be in (0, 1].")
        pic_tol = float(unconfined_head_tol)
        if pic_tol < 0.0:
            raise ValueError("unconfined_head_tol must be >= 0.")

        def _inner_solve_cheb(txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i, h0, mcycles,
                              storage_diag_iter=None, rhs_eff_iter=None):
            if storage_diag_iter is not None:
                # Phreatic Sy path: the storage diagonal and head_prev RHS were
                # built by the Picard loop from the current saturated thickness.
                # Use them directly with transient=False so the inner solve does
                # not re-derive a confined-volume storage term.
                return _solve_chebyshev_7point_3d_linear(
                    tx_p=txp_i,
                    tx_m=txm_i,
                    ty_p=typ_i,
                    ty_m=tym_i,
                    tz_p=tzp_i,
                    tz_m=tzm_i,
                    rhs=rhs_eff_iter,
                    active=act,
                    bc_mask=bcm,
                    bc_values=bcv,
                    initial_head=h0,
                    storage_diag=storage_diag_iter,
                    max_iter=int(max_iter),
                    cheby_order=int(cheby_order),
                    cheby_lambda_min=float(cheby_lambda_min),
                    cheby_lambda_max=float(cheby_lambda_max),
                    rel_tol=float(rel_tol),
                    abs_tol_min=float(abs_tol_min),
                    transient=False,
                    storage_coeff=None,
                    dt=None,
                    head_prev=None,
                    dx=float(dx),
                    dy=dy,
                    dz=float(dz),
                    device=str(device),
                    diag_preconditioner_backend=str(diag_preconditioner_backend),
                    return_info=True,
                )
            # Legacy / steady path. Only enable the inner confined-transient
            # storage term when storage_coeff is available (confined_volume
            # mode); otherwise solve steady.
            legacy_transient = bool(transient) and storage_coeff is not None
            return _solve_chebyshev_7point_3d_linear(
                tx_p=txp_i,
                tx_m=txm_i,
                ty_p=typ_i,
                ty_m=tym_i,
                tz_p=tzp_i,
                tz_m=tzm_i,
                rhs=b,
                active=act,
                bc_mask=bcm,
                bc_values=bcv,
                initial_head=h0,
                storage_diag=storage_diag,
                max_iter=int(max_iter),
                cheby_order=int(cheby_order),
                cheby_lambda_min=float(cheby_lambda_min),
                cheby_lambda_max=float(cheby_lambda_max),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                transient=legacy_transient,
                storage_coeff=(storage_coeff if legacy_transient else None),
                dt=(dt if legacy_transient else None),
                head_prev=(head_prev if legacy_transient else None),
                dx=float(dx),
                dy=dy,
                dz=float(dz),
                device=str(device),
                diag_preconditioner_backend=str(diag_preconditioner_backend),
                return_info=True,
            )

        return _picard_unconfined_7point_3d(
            _inner_solve_cheb,
            shape=shape,
            active=act,
            bc_mask=bcm,
            bc_values=bcv,
            rhs=b,
            kx=kx,
            ky=ky,
            kz=kz,
            zbot=zbot,
            ztop=ztop_field,
            initial_head=initial_head,
            storage_diag=storage_diag,
            min_sat=min_sat,
            max_outer=n_pic,
            pic_relax=pic_relax,
            pic_tol=pic_tol,
            omega_min=omega_min,
            omega_max=omega_max,
            dx=float(dx),
            dy=dy,
            dz=float(dz),
            device=str(device),
            unconfined_startup_mode=unconfined_startup_mode,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            transmissivity_relaxation_early=transmissivity_relaxation_early,
            transmissivity_relaxation_middle=transmissivity_relaxation_middle,
            transmissivity_relaxation_late=transmissivity_relaxation_late,
            transmissivity_relaxation_middle_iteration=transmissivity_relaxation_middle_iteration,
            transmissivity_relaxation_late_iteration=transmissivity_relaxation_late_iteration,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            inner_picard_scale_max_fraction=inner_picard_scale_max_fraction,
            chebyshev_enabled=chebyshev_enabled,
            chebyshev_order=chebyshev_order,
            chebyshev_lambda_min_fraction=chebyshev_lambda_min_fraction,
            chebyshev_reset_on_residual_increase=chebyshev_reset_on_residual_increase,
            chebyshev_reset_factor=chebyshev_reset_factor,
            chebyshev_minor_increase_patience=chebyshev_minor_increase_patience,
            chebyshev_rejection_factor=chebyshev_rejection_factor,
            unconfined_inner_max_cycles_early=unconfined_inner_max_cycles_early,
            unconfined_inner_max_cycles_middle=unconfined_inner_max_cycles_middle,
            unconfined_inner_max_cycles_late=unconfined_inner_max_cycles_late,
            unconfined_inner_late_dh=unconfined_inner_late_dh,
            unconfined_inner_middle_dh=unconfined_inner_middle_dh,
            max_head_change_per_outer_iteration=max_head_change_per_outer_iteration,
            residual_floor_tol=residual_floor_tol,
            dh_rms_tol=dh_rms_tol,
            diag_preconditioner_backend=diag_preconditioner_backend,
            linear_solver_type_label="chebyshev_7point_3d",
            solver_type_label="chebyshev_7point_3d_unconfined_picard",
            transient=transient,
            dt=dt,
            sy=sy,
            ss=ss,
            head_prev=head_prev,
            unconfined_storage_mode=unconfined_storage_mode,
            dry_cell_flag_threshold=dry_cell_flag_threshold,
            return_info=return_info,
        )

    return _solve_chebyshev_7point_3d_linear(
        tx_p=txp,
        tx_m=txm,
        ty_p=typ,
        ty_m=tym,
        tz_p=tzp,
        tz_m=tzm,
        rhs=b,
        active=act,
        bc_mask=bcm,
        bc_values=bcv,
        initial_head=initial_head,
        storage_diag=storage_diag,
        max_iter=int(max_iter),
        cheby_order=int(cheby_order),
        cheby_lambda_min=float(cheby_lambda_min),
        cheby_lambda_max=float(cheby_lambda_max),
        rel_tol=float(rel_tol),
        abs_tol_min=float(abs_tol_min),
        transient=bool(transient),
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        dx=float(dx),
        dy=dy,
        dz=float(dz),
        device=str(device),
        diag_preconditioner_backend=str(diag_preconditioner_backend),
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
    line_omega: float = 0.8,
    line_sweeps_pre: int = 1,
    line_sweeps_post: int = 1,
    line_sweeps_coarse: int = 1,
    vertical_line_max_nz: int = 64,
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
    sy: np.ndarray | float | None = None,
    ss: np.ndarray | float | None = None,
    unconfined_storage_mode: str = "phreatic_sy",
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
    ztop_field: np.ndarray | None = None,
    omega_min: float = 0.1,
    omega_max: float = 0.9,
    residual_floor_tol: float | None = 1.0e-4,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float | None = None,
    inner_head_residual_tol_max: float = 1.0e-2,
    inner_picard_scale_max_fraction: float = 0.10,
    chebyshev_enabled: bool = True,
    chebyshev_order: int = 3,
    chebyshev_lambda_min_fraction: float = 0.1,
    chebyshev_reset_on_residual_increase: bool = True,
    chebyshev_reset_factor: float = 1.2,
    chebyshev_minor_increase_patience: int = 2,
    chebyshev_rejection_factor: float = 1.2,
    unconfined_inner_max_cycles_early: int = 10,
    unconfined_inner_max_cycles_middle: int = 25,
    unconfined_inner_max_cycles_late: int = 60,
    unconfined_inner_late_dh: float = 1.0e-2,
    unconfined_inner_middle_dh: float = 1.0,
    transmissivity_relaxation_enabled: bool = False,
    transmissivity_relaxation_early: float = 0.25,
    transmissivity_relaxation_middle: float = 0.50,
    transmissivity_relaxation_late: float = 1.00,
    transmissivity_relaxation_middle_iteration: int = 5,
    transmissivity_relaxation_late_iteration: int = 15,
    unconfined_startup_mode: str = "initial_head",
    max_head_change_per_outer_iteration: float = 10.0,
    dry_cell_flag_threshold: float = 0.1,
    diag_preconditioner_backend: str = "auto",
    device: str = "cuda:0",
    return_info: bool = True,
):
    """
    3D multilevel K-cycle prototype for a fixed 7-point operator.

    Notes:
      - Coarse operators are built by 1x2x2 horizontal semi-coarsening of face
        conductances: layers are preserved and only horizontal dimensions are
        coarsened.  Vertical face conductances use the same horizontal block
        mean as the other operator arrays as a transparent first implementation.
      - Correction scheme uses homogeneous coarse Dirichlet values (bc_values=0 on coarse).
      - Optional confined transient stream adds storage diagonal and RHS term.
      - Optional unconfined stream uses Picard outer iterations with K*sat updates.
    """
    txp0 = np.asarray(tx_p, dtype=NP_FLOAT)
    txm0 = np.asarray(tx_m, dtype=NP_FLOAT)
    typ0 = np.asarray(ty_p, dtype=NP_FLOAT)
    tym0 = np.asarray(ty_m, dtype=NP_FLOAT)
    tzp0 = np.asarray(tz_p, dtype=NP_FLOAT)
    tzm0 = np.asarray(tz_m, dtype=NP_FLOAT)
    b0 = np.asarray(rhs, dtype=NP_FLOAT)
    act0 = np.asarray(active, dtype=np.int32)
    bcm0 = np.asarray(bc_mask, dtype=np.int32)
    bcv0 = np.asarray(bc_values, dtype=NP_FLOAT)

    shape0 = txp0.shape
    if txp0.ndim != 3:
        raise ValueError("7-point arrays must be 3D with shape (nz, ny, nx).")

    for name, arr in (
        ("tx_m", txm0),
        ("ty_p", typ0),
        ("ty_m", tym0),
        ("tz_p", tzp0),
        ("tz_m", tzm0),
        ("rhs", b0),
        ("active", act0),
        ("bc_mask", bcm0),
        ("bc_values", bcv0),
    ):
        if arr.shape != shape0:
            raise ValueError(f"{name} shape {arr.shape} expected {shape0}")

    free0 = (act0 != 0) & (bcm0 == 0)
    n_free0 = int(np.count_nonzero(free0))

    if bool(unconfined) and bool(transient):
        # Resolve the transient unconfined storage mode. When a specific yield
        # (sy) is supplied the phreatic water-table storage path is used
        # (Sy*dx*dy/dt on the per-column water-table cell + Ss*sat*dx*dy/dt on
        # saturated cells, recomputed every Picard iteration from h_iter).
        # When only storage_coeff is supplied the legacy confined-volume
        # approximation (storage_coeff*dx*dy*dz/dt over the full cell volume)
        # is retained for backward compatibility. Either way a clean storage
        # diagonal is forced so the inner solve rebuilds / replaces it on every
        # outer iteration (no double-counting of a caller-supplied diagonal).
        _storage_mode = str(unconfined_storage_mode).strip().lower()
        if _storage_mode not in {"phreatic_sy", "confined_volume"}:
            raise ValueError("unconfined_storage_mode must be 'phreatic_sy' or 'confined_volume'.")
        if _storage_mode == "phreatic_sy" and sy is None and storage_coeff is not None:
            # Caller used the legacy storage_coeff argument: keep the confined-
            # volume approximation so existing callers/tests behave as before.
            _storage_mode = "confined_volume"
        unconfined_storage_mode = _storage_mode
        storage_diag = None

    if bool(unconfined):
        if kx_field is None or ky_field is None or kz_field is None or zbot_field is None:
            raise ValueError("unconfined=True requires kx_field, ky_field, kz_field, and zbot_field.")

        kx = np.asarray(kx_field, dtype=NP_FLOAT)
        ky = np.asarray(ky_field, dtype=NP_FLOAT)
        kz = np.asarray(kz_field, dtype=NP_FLOAT)
        zbot = np.asarray(zbot_field, dtype=NP_FLOAT)
        for name, arr in (("kx_field", kx), ("ky_field", ky), ("kz_field", kz), ("zbot_field", zbot)):
            if arr.shape != shape0:
                raise ValueError(f"{name} shape {arr.shape} expected {shape0}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite.")
        if np.any(kx < NP_FLOAT(0.0)) or np.any(ky < NP_FLOAT(0.0)) or np.any(kz < NP_FLOAT(0.0)):
            raise ValueError("Unconfined K fields must be >= 0.")

        min_sat = float(unconfined_min_sat)
        if min_sat <= 0.0:
            raise ValueError("unconfined_min_sat must be positive.")
        n_pic = int(unconfined_max_picard_iter)
        if n_pic < 1:
            raise ValueError("unconfined_max_picard_iter must be >= 1.")
        pic_relax = float(unconfined_relax)
        if pic_relax <= 0.0 or pic_relax > 1.0:
            raise ValueError("unconfined_relax must be in (0, 1].")
        pic_tol = float(unconfined_head_tol)
        if pic_tol < 0.0:
            raise ValueError("unconfined_head_tol must be >= 0.")

        def _inner_solve_kc(txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i, h0, mcycles,
                            storage_diag_iter=None, rhs_eff_iter=None):
            common_kwargs = dict(
                tx_p=txp_i,
                tx_m=txm_i,
                ty_p=typ_i,
                ty_m=tym_i,
                tz_p=tzp_i,
                tz_m=tzm_i,
                active=act0,
                bc_mask=bcm0,
                bc_values=bcv0,
                initial_head=h0,
                max_cycles=int(mcycles),
                nu_pre=int(nu_pre),
                nu_post=int(nu_post),
                nu_coarse=int(nu_coarse),
                max_levels=int(max_levels),
                min_coarse_n=int(min_coarse_n),
                smoother=str(smoother),
                omega=float(omega),
                line_omega=float(line_omega),
                line_sweeps_pre=int(line_sweeps_pre),
                line_sweeps_post=int(line_sweeps_post),
                line_sweeps_coarse=int(line_sweeps_coarse),
                vertical_line_max_nz=int(vertical_line_max_nz),
                cheby_lambda_min=float(cheby_lambda_min),
                cheby_lambda_max=float(cheby_lambda_max),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                check_every_no=int(check_every_no),
                dh_rms_tol=dh_rms_tol,
                dh_max_tol=dh_max_tol,
                dh_max_factor=float(dh_max_factor),
                dx=float(dx),
                dy=dy,
                dz=float(dz),
                unconfined=False,
                diag_preconditioner_backend=str(diag_preconditioner_backend),
                device=str(device),
                return_info=True,
            )
            if storage_diag_iter is not None:
                # Phreatic Sy path: use the Picard-built storage diagonal and
                # head_prev RHS directly (no inner confined-transient term).
                return solve_multigrid_kcycle_7point_3d(
                    rhs=rhs_eff_iter,
                    storage_diag=storage_diag_iter,
                    transient=False,
                    storage_coeff=None,
                    dt=None,
                    head_prev=None,
                    **common_kwargs,
                )
            # Legacy / steady path. Only enable the inner confined-transient
            # storage term when storage_coeff is available (confined_volume
            # mode); otherwise solve steady.
            legacy_transient = bool(transient) and storage_coeff is not None
            return solve_multigrid_kcycle_7point_3d(
                rhs=b0,
                storage_diag=storage_diag,
                transient=legacy_transient,
                storage_coeff=(storage_coeff if legacy_transient else None),
                dt=(dt if legacy_transient else None),
                head_prev=(head_prev if legacy_transient else None),
                **common_kwargs,
            )

        return _picard_unconfined_7point_3d(
            _inner_solve_kc,
            shape=shape0,
            active=act0,
            bc_mask=bcm0,
            bc_values=bcv0,
            rhs=b0,
            kx=kx,
            ky=ky,
            kz=kz,
            zbot=zbot,
            ztop=ztop_field,
            initial_head=initial_head,
            storage_diag=storage_diag,
            min_sat=min_sat,
            max_outer=n_pic,
            pic_relax=pic_relax,
            pic_tol=pic_tol,
            omega_min=omega_min,
            omega_max=omega_max,
            dx=float(dx),
            dy=dy,
            dz=float(dz),
            device=str(device),
            unconfined_startup_mode=unconfined_startup_mode,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            transmissivity_relaxation_early=transmissivity_relaxation_early,
            transmissivity_relaxation_middle=transmissivity_relaxation_middle,
            transmissivity_relaxation_late=transmissivity_relaxation_late,
            transmissivity_relaxation_middle_iteration=transmissivity_relaxation_middle_iteration,
            transmissivity_relaxation_late_iteration=transmissivity_relaxation_late_iteration,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            inner_picard_scale_max_fraction=inner_picard_scale_max_fraction,
            chebyshev_enabled=chebyshev_enabled,
            chebyshev_order=chebyshev_order,
            chebyshev_lambda_min_fraction=chebyshev_lambda_min_fraction,
            chebyshev_reset_on_residual_increase=chebyshev_reset_on_residual_increase,
            chebyshev_reset_factor=chebyshev_reset_factor,
            chebyshev_minor_increase_patience=chebyshev_minor_increase_patience,
            chebyshev_rejection_factor=chebyshev_rejection_factor,
            unconfined_inner_max_cycles_early=unconfined_inner_max_cycles_early,
            unconfined_inner_max_cycles_middle=unconfined_inner_max_cycles_middle,
            unconfined_inner_max_cycles_late=unconfined_inner_max_cycles_late,
            unconfined_inner_late_dh=unconfined_inner_late_dh,
            unconfined_inner_middle_dh=unconfined_inner_middle_dh,
            max_head_change_per_outer_iteration=max_head_change_per_outer_iteration,
            residual_floor_tol=residual_floor_tol,
            dh_rms_tol=dh_rms_tol,
            diag_preconditioner_backend=diag_preconditioner_backend,
            linear_solver_type_label="kcycle_7point_3d",
            solver_type_label="kcycle_7point_3d_unconfined_picard",
            transient=transient,
            dt=dt,
            sy=sy,
            ss=ss,
            head_prev=head_prev,
            unconfined_storage_mode=unconfined_storage_mode,
            dry_cell_flag_threshold=dry_cell_flag_threshold,
            return_info=return_info,
        )

    if n_free0 <= 0:
        if initial_head is None:
            h0 = np.zeros(shape0, dtype=NP_FLOAT)
        else:
            h0 = np.asarray(initial_head, dtype=NP_FLOAT).copy()
            if h0.shape != shape0:
                raise ValueError(f"initial_head shape {h0.shape} expected {shape0}")
        h0[bcm0 != 0] = bcv0[bcm0 != 0]
        h0[act0 == 0] = NP_FLOAT(0.0)
        info0 = {
            "solver_type": "kcycle_7point_3d",
            "n_cycles_used": 0,
            "converged": True,
            "r_rms0": 0.0,
            "r_rms_end": 0.0,
            "tol_abs": float(abs_tol_min),
            "transient": bool(transient),
            "transient_formulation": "confined" if bool(transient) else "steady",
            "dt": float(dt) if bool(transient) and dt is not None else float("nan"),
            "unconfined": False,
        }
        return (h0, info0) if return_info else h0

    b_eff0, sdiag0, h_prev_used, dt_used = _prepare_7point_transient_terms(
        rhs=b0,
        storage_diag=storage_diag,
        active=act0,
        bc_mask=bcm0,
        bc_values=bcv0,
        transient=bool(transient),
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        initial_head=initial_head,
        dx=float(dx),
        dy=dy,
        dz=float(dz),
    )

    if initial_head is None:
        if bool(transient) and h_prev_used is not None:
            x0 = np.asarray(h_prev_used, dtype=NP_FLOAT).copy()
        else:
            x0 = np.zeros(shape0, dtype=NP_FLOAT)
    else:
        x0 = np.asarray(initial_head, dtype=NP_FLOAT).copy()
        if x0.shape != shape0:
            raise ValueError(f"initial_head shape {x0.shape} expected {shape0}")

    x0[bcm0 != 0] = bcv0[bcm0 != 0]
    x0[act0 == 0] = NP_FLOAT(0.0)

    dh_rms_tol_f = None if dh_rms_tol is None else float(dh_rms_tol)
    if dh_max_tol is None:
        dh_max_tol_f = None if dh_rms_tol_f is None else float(dh_max_factor) * dh_rms_tol_f
    else:
        dh_max_tol_f = float(dh_max_tol)

    smooth_mode = str(smoother).strip().lower()
    if smooth_mode not in {"chebyshev", "jacobi", "vertical_line", "chebyshev_vertical_line"}:
        raise ValueError("smoother must be 'chebyshev', 'jacobi', 'vertical_line', or 'chebyshev_vertical_line'.")
    uses_vertical_line = smooth_mode in {"vertical_line", "chebyshev_vertical_line"}
    if uses_vertical_line:
        if int(shape0[0]) > int(vertical_line_max_nz):
            raise ValueError(f"vertical_line smoother currently supports nz <= {vertical_line_max_nz}")
    line_sweeps_pre_i = int(line_sweeps_pre)
    line_sweeps_post_i = int(line_sweeps_post)
    line_sweeps_coarse_i = int(line_sweeps_coarse)
    if line_sweeps_pre_i < 0 or line_sweeps_post_i < 0 or line_sweeps_coarse_i < 0:
        raise ValueError("line_sweeps_pre, line_sweeps_post, and line_sweeps_coarse must be >= 0.")

    if smooth_mode == "vertical_line":
        omg = float(line_omega)
        pre_omegas = tuple(omg for _ in range(int(nu_pre)))
        post_omegas = tuple(omg for _ in range(int(nu_post)))
        point_pre_omegas: tuple[float, ...] = ()
        point_post_omegas: tuple[float, ...] = ()
    elif smooth_mode in {"chebyshev", "chebyshev_vertical_line"}:
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
        point_pre_omegas = pre_omegas
        point_post_omegas = post_omegas
    else:
        omg = float(omega)
        pre_omegas = tuple(omg for _ in range(int(nu_pre)))
        post_omegas = tuple(omg for _ in range(int(nu_post)))
        point_pre_omegas = pre_omegas
        point_post_omegas = post_omegas
    if len(pre_omegas) == 0:
        pre_omegas = (1.0,)
    if len(post_omegas) == 0:
        post_omegas = (1.0,)
    if smooth_mode != "vertical_line":
        point_pre_omegas = pre_omegas
        point_post_omegas = post_omegas

    max_cycles_i = int(max_cycles)
    if max_cycles_i < 1:
        raise ValueError("max_cycles must be >= 1.")
    check_every = int(check_every_no)
    if check_every < 1:
        raise ValueError("check_every_no must be >= 1.")

    levels_host: list[dict] = []
    levels_host.append(
        {
            "tx_p": txp0,
            "tx_m": txm0,
            "ty_p": typ0,
            "ty_m": tym0,
            "tz_p": tzp0,
            "tz_m": tzm0,
            "active": act0,
            "bc_mask": bcm0,
            "bc_values": bcv0,
            "storage_diag": sdiag0,
            "b0": b_eff0,
            "coarsen_y": 1,
            "coarsen_x": 1,
        }
    )

    for _ in range(1, int(max_levels)):
        prev = levels_host[-1]
        nz_f, ny_f, nx_f = prev["active"].shape

        coarsen_y, coarsen_x = _choose_horizontal_coarsening(
            ny=int(ny_f),
            nx=int(nx_f),
            min_coarse_n=int(min_coarse_n),
        )
        if coarsen_y == 1 and coarsen_x == 1:
            break
        nz_c = int(nz_f)
        ny_c = (int(ny_f) + coarsen_y - 1) // coarsen_y
        nx_c = (int(nx_f) + coarsen_x - 1) // coarsen_x

        active_c = _coarsen_max_edge_axes(
            prev["active"], coarsen_y=coarsen_y, coarsen_x=coarsen_x
        )
        bc_mask_c = _coarsen_max_edge_axes(
            prev["bc_mask"], coarsen_y=coarsen_y, coarsen_x=coarsen_x
        )
        bc_values_c = np.zeros((nz_c, ny_c, nx_c), dtype=NP_FLOAT)

        storage_c = _coarsen_mean_edge_axes(
            prev["storage_diag"], coarsen_y=coarsen_y, coarsen_x=coarsen_x
        )
        free_c = (active_c != 0) & (bc_mask_c == 0)
        storage_c[~free_c] = NP_FLOAT(0.0)

        levels_host.append(
            {
                "tx_p": _coarsen_mean_edge_axes(prev["tx_p"], coarsen_y=coarsen_y, coarsen_x=coarsen_x),
                "tx_m": _coarsen_mean_edge_axes(prev["tx_m"], coarsen_y=coarsen_y, coarsen_x=coarsen_x),
                "ty_p": _coarsen_mean_edge_axes(prev["ty_p"], coarsen_y=coarsen_y, coarsen_x=coarsen_x),
                "ty_m": _coarsen_mean_edge_axes(prev["ty_m"], coarsen_y=coarsen_y, coarsen_x=coarsen_x),
                "tz_p": _coarsen_mean_edge_axes(prev["tz_p"], coarsen_y=coarsen_y, coarsen_x=coarsen_x),
                "tz_m": _coarsen_mean_edge_axes(prev["tz_m"], coarsen_y=coarsen_y, coarsen_x=coarsen_x),
                "active": active_c,
                "bc_mask": bc_mask_c,
                "bc_values": bc_values_c,
                "storage_diag": storage_c,
                "b0": np.zeros((nz_c, ny_c, nx_c), dtype=NP_FLOAT),
                "coarsen_y": 1,
                "coarsen_x": 1,
            }
        )
        prev["coarsen_y"] = int(coarsen_y)
        prev["coarsen_x"] = int(coarsen_x)

    levels: list[dict] = []
    diag_mode = _resolve_diag_backend(diag_preconditioner_backend, device)
    for lid, lh in enumerate(levels_host):
        nz, ny, nx = lh["active"].shape
        shapeL = (nz, ny, nx)
        dimL = (nz, ny, nx)

        lvl = {
            "level_id": int(lid),
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "shape": shapeL,
            "dim": dimL,
            "coarsen_y": int(lh.get("coarsen_y", 1)),
            "coarsen_x": int(lh.get("coarsen_x", 1)),
            "tx_p_wp": wp.array(lh["tx_p"], dtype=WP_FLOAT, device=device),
            "tx_m_wp": wp.array(lh["tx_m"], dtype=WP_FLOAT, device=device),
            "ty_p_wp": wp.array(lh["ty_p"], dtype=WP_FLOAT, device=device),
            "ty_m_wp": wp.array(lh["ty_m"], dtype=WP_FLOAT, device=device),
            "tz_p_wp": wp.array(lh["tz_p"], dtype=WP_FLOAT, device=device),
            "tz_m_wp": wp.array(lh["tz_m"], dtype=WP_FLOAT, device=device),
            "active_wp": wp.array(lh["active"], dtype=wp.int32, device=device),
            "bc_mask_wp": wp.array(lh["bc_mask"], dtype=wp.int32, device=device),
            "bc_values_wp": wp.array(lh["bc_values"], dtype=WP_FLOAT, device=device),
            "storage_wp": wp.array(lh["storage_diag"], dtype=WP_FLOAT, device=device),
            "x_wp": wp.zeros(shapeL, dtype=WP_FLOAT, device=device),
            "b_wp": wp.array(lh["b0"], dtype=WP_FLOAT, device=device),
            "r_wp": wp.zeros(shapeL, dtype=WP_FLOAT, device=device),
            "Ax_wp": wp.zeros(shapeL, dtype=WP_FLOAT, device=device),
            "e_wp": wp.zeros(shapeL, dtype=WP_FLOAT, device=device),
            "z1_wp": wp.zeros(shapeL, dtype=WP_FLOAT, device=device),
            "r1_wp": wp.zeros(shapeL, dtype=WP_FLOAT, device=device),
            "rTr_buf": wp.zeros(1, dtype=wp.float64, device=device),
            "dot_buf": wp.zeros(1, dtype=wp.float64, device=device),
            "pAp_buf": wp.zeros(1, dtype=wp.float64, device=device),
            "dh_max_buf": wp.zeros(1, dtype=wp.float64, device=device),
        }
        if diag_mode == "device":
            lvl["M_inv_wp"] = wp.zeros(shapeL, dtype=WP_FLOAT, device=device)
            _fill_m_inv_wp_7point(
                lvl,
                lvl["M_inv_wp"],
                dimL,
                int(nx),
                int(ny),
                int(nz),
                device,
            )
        else:
            lvl["M_inv_wp"] = wp.array(
                build_diag_preconditioner_7point(
                    tx_p=lh["tx_p"],
                    tx_m=lh["tx_m"],
                    ty_p=lh["ty_p"],
                    ty_m=lh["ty_m"],
                    tz_p=lh["tz_p"],
                    tz_m=lh["tz_m"],
                    active=lh["active"],
                    bc_mask=lh["bc_mask"],
                    storage_diag=lh["storage_diag"],
                ),
                dtype=WP_FLOAT,
                device=device,
            )
        if uses_vertical_line:
            lvl["c_prime_wp"] = wp.zeros(shapeL, dtype=WP_FLOAT, device=device)
            lvl["d_prime_wp"] = wp.zeros(shapeL, dtype=WP_FLOAT, device=device)
        levels.append(lvl)

    lvl0 = levels[0]
    lvl0["x_wp"] = wp.array(x0, dtype=WP_FLOAT, device=device)
    lvl0["x_prev_check_wp"] = wp.array(x0, dtype=WP_FLOAT, device=device)

    for lid in range(1, len(levels)):
        levels[lid]["x_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["b_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["r_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["Ax_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["e_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["z1_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["r1_wp"].fill_(WP_FLOAT(0.0))

    def smooth_point_level(level: dict, omegas: tuple[float, ...]) -> None:
        x_in = level["x_wp"]
        x_out = level["Ax_wp"]
        for om in omegas:
            wp.launch(
                kernel=jacobi_applyA_fused_7point_kernel,
                dim=level["dim"],
                inputs=[
                    level["tx_p_wp"],
                    level["tx_m_wp"],
                    level["ty_p_wp"],
                    level["ty_m_wp"],
                    level["tz_p_wp"],
                    level["tz_m_wp"],
                    level["active_wp"],
                    level["bc_mask_wp"],
                    level["storage_wp"],
                    level["b_wp"],
                    x_in,
                    level["M_inv_wp"],
                    level["bc_values_wp"],
                    float(om),
                    int(level["nx"]),
                    int(level["ny"]),
                    int(level["nz"]),
                    x_out,
                ],
                device=device,
            )
            tmp = x_in
            x_in = x_out
            x_out = tmp

        if x_in is not level["x_wp"]:
            wp.launch(
                kernel=copy_field_3d_kernel,
                dim=level["dim"],
                inputs=[x_in, level["x_wp"], int(level["nx"]), int(level["ny"]), int(level["nz"])],
                device=device,
            )

    def smooth_vertical_line_level(level: dict, n_sweeps: int) -> None:
        if int(n_sweeps) <= 0:
            return
        from DARCY_WARP_PACKAGE.kernels_3d import vertical_line_relaxation_7point_kernel

        x_in = level["x_wp"]
        x_out = level["Ax_wp"]
        for _ in range(int(n_sweeps)):
            wp.launch(
                kernel=vertical_line_relaxation_7point_kernel,
                dim=(level["ny"] * level["nx"],),
                inputs=[
                    level["tx_p_wp"],
                    level["tx_m_wp"],
                    level["ty_p_wp"],
                    level["ty_m_wp"],
                    level["tz_p_wp"],
                    level["tz_m_wp"],
                    level["active_wp"],
                    level["bc_mask_wp"],
                    level["storage_wp"],
                    level["b_wp"],
                    x_in,
                    level["bc_values_wp"],
                    float(line_omega),
                    level["c_prime_wp"],
                    level["d_prime_wp"],
                    int(level["nx"]),
                    int(level["ny"]),
                    int(level["nz"]),
                    x_out,
                ],
                device=device,
            )
            tmp = x_in
            x_in = x_out
            x_out = tmp

        if x_in is not level["x_wp"]:
            wp.launch(
                kernel=copy_field_3d_kernel,
                dim=level["dim"],
                inputs=[x_in, level["x_wp"], int(level["nx"]), int(level["ny"]), int(level["nz"])],
                device=device,
            )

    def smooth_level(level: dict, point_omegas: tuple[float, ...], line_sweeps: int) -> None:
        if smooth_mode == "vertical_line":
            smooth_vertical_line_level(level, len(point_omegas))
        elif smooth_mode == "chebyshev_vertical_line":
            smooth_point_level(level, point_omegas)
            smooth_vertical_line_level(level, line_sweeps)
        else:
            smooth_point_level(level, point_omegas)

    def compute_residual_norm(level: dict, x_in_wp, r_out_wp, rtr_buf) -> float:
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rtr_buf], device=device)
        wp.launch(
            kernel=compute_residual_7point_kernel,
            dim=level["dim"],
            inputs=[
                x_in_wp,
                level["b_wp"],
                level["tx_p_wp"],
                level["tx_m_wp"],
                level["ty_p_wp"],
                level["ty_m_wp"],
                level["tz_p_wp"],
                level["tz_m_wp"],
                level["active_wp"],
                level["bc_mask_wp"],
                level["storage_wp"],
                r_out_wp,
                rtr_buf,
                int(level["nx"]),
                int(level["ny"]),
                int(level["nz"]),
            ],
            device=device,
        )
        return float(rtr_buf.numpy()[0])

    def coarsest_relax(level: dict) -> None:
        if smooth_mode == "chebyshev_vertical_line":
            for _ in range(int(nu_coarse)):
                smooth_point_level(level, point_pre_omegas)
                smooth_vertical_line_level(level, line_sweeps_coarse_i)
        else:
            for _ in range(int(nu_coarse)):
                smooth_level(level, pre_omegas, len(pre_omegas))

    def kcycle(level_id: int) -> None:
        level = levels[level_id]
        smooth_level(level, point_pre_omegas if smooth_mode != "vertical_line" else pre_omegas, line_sweeps_pre_i)
        compute_residual_norm(level, level["x_wp"], level["r_wp"], level["rTr_buf"])

        if level_id == (len(levels) - 1):
            coarsest_relax(level)
            return

        coarse = levels[level_id + 1]

        wp.launch(
            kernel=restrict_blockavg_axes_3d_kernel,
            dim=coarse["dim"],
            inputs=[
                level["r_wp"],
                level["active_wp"],
                level["bc_mask_wp"],
                coarse["b_wp"],
                int(level["nx"]),
                int(level["ny"]),
                int(level["nz"]),
                int(coarse["nx"]),
                int(coarse["ny"]),
                int(coarse["nz"]),
                int(level.get("coarsen_y", 2)),
                int(level.get("coarsen_x", 2)),
            ],
            device=device,
        )

        coarse["x_wp"].fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)  # z1
        wp.launch(
            kernel=copy_field_3d_kernel,
            dim=coarse["dim"],
            inputs=[coarse["x_wp"], coarse["z1_wp"], int(coarse["nx"]), int(coarse["ny"]), int(coarse["nz"])],
            device=device,
        )

        # r1 = b - A z1
        compute_residual_norm(coarse, coarse["z1_wp"], coarse["r_wp"], coarse["rTr_buf"])
        wp.launch(
            kernel=copy_field_3d_kernel,
            dim=coarse["dim"],
            inputs=[coarse["r_wp"], coarse["r1_wp"], int(coarse["nx"]), int(coarse["ny"]), int(coarse["nz"])],
            device=device,
        )
        wp.launch(
            kernel=copy_field_3d_kernel,
            dim=coarse["dim"],
            inputs=[coarse["r1_wp"], coarse["b_wp"], int(coarse["nx"]), int(coarse["ny"]), int(coarse["nz"])],
            device=device,
        )

        coarse["x_wp"].fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)  # z2 in coarse.x_wp

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[coarse["dot_buf"]], device=device)
        wp.launch(
            kernel=dot_active_3d_kernel,
            dim=coarse["dim"],
            inputs=[
                coarse["r1_wp"],
                coarse["x_wp"],
                coarse["active_wp"],
                coarse["bc_mask_wp"],
                coarse["dot_buf"],
                int(coarse["nx"]),
                int(coarse["ny"]),
                int(coarse["nz"]),
            ],
            device=device,
        )
        num = float(coarse["dot_buf"].numpy()[0])

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[coarse["pAp_buf"]], device=device)
        wp.launch(
            kernel=apply_A_and_pAp_7point_kernel,
            dim=coarse["dim"],
            inputs=[
                coarse["tx_p_wp"],
                coarse["tx_m_wp"],
                coarse["ty_p_wp"],
                coarse["ty_m_wp"],
                coarse["tz_p_wp"],
                coarse["tz_m_wp"],
                coarse["active_wp"],
                coarse["bc_mask_wp"],
                coarse["storage_wp"],
                coarse["x_wp"],
                coarse["Ax_wp"],
                coarse["pAp_buf"],
                int(coarse["nx"]),
                int(coarse["ny"]),
                int(coarse["nz"]),
            ],
            device=device,
        )
        den = float(coarse["pAp_buf"].numpy()[0])
        alpha = 0.0
        if np.isfinite(num) and np.isfinite(den) and abs(den) > 1.0e-30:
            alpha = num / den

        wp.launch(
            kernel=axpy_active_scalar_3d_kernel,
            dim=coarse["dim"],
            inputs=[
                coarse["z1_wp"],
                coarse["x_wp"],
                coarse["active_wp"],
                coarse["bc_mask_wp"],
                float(alpha),
                int(coarse["nx"]),
                int(coarse["ny"]),
                int(coarse["nz"]),
            ],
            device=device,
        )

        wp.launch(
            kernel=prolong_bilinear_axes_3d_kernel,
            dim=level["dim"],
            inputs=[
                coarse["z1_wp"],
                level["e_wp"],
                int(level["nx"]),
                int(level["ny"]),
                int(level["nz"]),
                int(coarse["nx"]),
                int(coarse["ny"]),
                int(coarse["nz"]),
                int(level.get("coarsen_y", 2)),
                int(level.get("coarsen_x", 2)),
            ],
            device=device,
        )
        wp.launch(
            kernel=add_correction_3d_kernel,
            dim=level["dim"],
            inputs=[
                level["x_wp"],
                level["e_wp"],
                level["active_wp"],
                level["bc_mask_wp"],
                level["bc_values_wp"],
                int(level["nx"]),
                int(level["ny"]),
                int(level["nz"]),
            ],
            device=device,
        )

        smooth_level(level, point_post_omegas if smooth_mode != "vertical_line" else post_omegas, line_sweeps_post_i)

    rTr0 = compute_residual_norm(lvl0, lvl0["x_wp"], lvl0["r_wp"], lvl0["rTr_buf"])
    r_rms0 = float(np.sqrt(max(rTr0, 0.0) / float(n_free0)))
    tol_abs = float(max(abs_tol_min, rel_tol * r_rms0))

    converged = r_rms0 <= tol_abs
    n_cycles_used = 0
    dh_rms_lastcheck = float("nan")
    dh_max_lastcheck = float("nan")

    for cyc in range(max_cycles_i):
        if converged:
            break
        n_cycles_used = cyc + 1
        kcycle(0)

        should_check = ((cyc % check_every) == (check_every - 1)) or (cyc == (max_cycles_i - 1))
        if not should_check:
            continue

        rTr_now = compute_residual_norm(lvl0, lvl0["x_wp"], lvl0["r_wp"], lvl0["rTr_buf"])
        r_rms_now = float(np.sqrt(max(rTr_now, 0.0) / float(n_free0)))

        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0["dot_buf"]], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0["dh_max_buf"]], device=device)
        wp.launch(
            kernel=dh_change_reduce_3d_kernel,
            dim=lvl0["dim"],
            inputs=[
                lvl0["x_wp"],
                lvl0["x_prev_check_wp"],
                lvl0["active_wp"],
                lvl0["bc_mask_wp"],
                lvl0["dot_buf"],
                lvl0["dh_max_buf"],
                lvl0["nx"],
                lvl0["ny"],
                lvl0["nz"],
            ],
            device=device,
        )
        dh2 = float(lvl0["dot_buf"].numpy()[0])
        dh_rms_lastcheck = float(np.sqrt(max(dh2, 0.0) / float(n_free0)))
        dh_max_lastcheck = float(lvl0["dh_max_buf"].numpy()[0])

        dh_ok = True
        if dh_rms_tol_f is not None:
            dh_ok = dh_ok and (dh_rms_lastcheck <= float(dh_rms_tol_f))
        if dh_max_tol_f is not None:
            dh_ok = dh_ok and (dh_max_lastcheck <= float(dh_max_tol_f))

        if (r_rms_now <= tol_abs) and dh_ok:
            converged = True
            break

    head_out = np.asarray(lvl0["x_wp"].numpy(), dtype=NP_FLOAT)
    rTr_end = compute_residual_norm(lvl0, lvl0["x_wp"], lvl0["r_wp"], lvl0["rTr_buf"])
    r_rms_end = float(np.sqrt(max(rTr_end, 0.0) / float(n_free0)))

    # Head-equivalent (Jacobi-preconditioned) residual RMS for reporting / inner-usable checks.
    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0["rTr_buf"]], device=device)
    wp.launch(
        kernel=compute_head_residual_7point_kernel,
        dim=lvl0["dim"],
        inputs=[
            lvl0["x_wp"],
            lvl0["b_wp"],
            lvl0["tx_p_wp"],
            lvl0["tx_m_wp"],
            lvl0["ty_p_wp"],
            lvl0["ty_m_wp"],
            lvl0["tz_p_wp"],
            lvl0["tz_m_wp"],
            lvl0["active_wp"],
            lvl0["bc_mask_wp"],
            lvl0["storage_wp"],
            lvl0["M_inv_wp"],
            lvl0["r_wp"],
            lvl0["rTr_buf"],
            int(lvl0["nx"]),
            int(lvl0["ny"]),
            int(lvl0["nz"]),
        ],
        device=device,
    )
    hrTr_end = float(lvl0["rTr_buf"].numpy()[0])
    h_rms_end = float(np.sqrt(max(hrTr_end, 0.0) / float(n_free0)))

    info = {
        "solver_type": "kcycle_7point_3d",
        "n_levels": int(len(levels)),
        "coarsening_mode": "horizontal",
        "level_shapes": [tuple(lh["active"].shape) for lh in levels_host],
        "coarsening_factors": [
            (int(lh.get("coarsen_y", 1)), int(lh.get("coarsen_x", 1)))
            for lh in levels_host[:-1]
        ],
        "max_cycles": int(max_cycles_i),
        "n_cycles_used": int(n_cycles_used),
        "nu_pre": int(nu_pre),
        "nu_post": int(nu_post),
        "nu_coarse": int(nu_coarse),
        "smoother": str(smooth_mode),
        "omega": float(omega),
        "cheby_lambda_min": (
            float(cheby_lambda_min) if smooth_mode in {"chebyshev", "chebyshev_vertical_line"} else float("nan")
        ),
        "cheby_lambda_max": (
            float(cheby_lambda_max) if smooth_mode in {"chebyshev", "chebyshev_vertical_line"} else float("nan")
        ),
        "cheby_pre_omegas": [float(v) for v in pre_omegas],
        "cheby_post_omegas": [float(v) for v in post_omegas],
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "tol_abs": float(tol_abs),
        "r_rms0": float(r_rms0),
        "r_rms_start": float(r_rms0),
        "r_rms_end": float(r_rms_end),
        "h_rms_end": float(h_rms_end),
        "dh_rms_lastcheck": float(dh_rms_lastcheck),
        "dh_max_lastcheck": float(dh_max_lastcheck),
        "converged": bool(converged),
        "transient": bool(transient),
        "transient_formulation": "confined" if bool(transient) else "steady",
        "dt": float(dt_used) if bool(transient) else float("nan"),
        "unconfined": False,
        "line_omega": float(line_omega),
        "line_sweeps_pre": int(line_sweeps_pre_i),
        "line_sweeps_post": int(line_sweeps_post_i),
        "line_sweeps_coarse": int(line_sweeps_coarse_i),
        "check_every_no": int(check_every),
        "vertical_line_max_nz": int(vertical_line_max_nz),
        "diag_preconditioner_backend": diag_mode,
    }

    # Release the multigrid device arrays so the unconfined Picard loop (which
    # rebuilds levels every outer iteration) does not accumulate GPU memory.
    _release_mg_levels_3d(levels)

    return (head_out, info) if return_info else head_out


__all__ = [
    "_solve_chebyshev_7point_3d_linear",
    "solve_chebyshev_7point_3d",
    "solve_multigrid_kcycle_7point_3d",
]
