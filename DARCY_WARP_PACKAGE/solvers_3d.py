# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.kernels_3d import (
    add_correction_3d_kernel,
    apply_A_and_pAp_7point_kernel,
    axpy_active_scalar_3d_kernel,
    copy_field_3d_kernel,
    compute_residual_7point_kernel,
    dot_active_3d_kernel,
    jacobi_applyA_fused_7point_kernel,
    prolong_bilinear_xy_3d_kernel,
    restrict_blockavg_xy_3d_kernel,
    zero_scalar_kernel,
)
from DARCY_WARP_PACKAGE.warped_darcy_chebyshev import (
    NP_FLOAT,
    WP_FLOAT,
    _chebyshev_relaxation_sequence,
    _prepare_7point_transient_terms,
    build_7point_face_conductance_from_k,
    build_diag_preconditioner_7point,
)


def _coarsen_mean_edge_1x2x2(field_f: np.ndarray) -> np.ndarray:
    """
    Horizontally semi-coarsen a 3D field while preserving layer index.

    This is used for layered groundwater multigrid levels where the vertical
    axis represents model layers, not a smooth geometric continuum.  Values are
    averaged over 2x2 horizontal blocks independently within each layer.  For
    vertical face conductances this is a transparent first-order aggregation:
    the coarse vertical coupling is the arithmetic mean of the fine vertical
    couplings in the horizontal block, rather than a hidden full 2x2x2 merge.
    """
    arr_f = np.asarray(field_f, dtype=NP_FLOAT)
    nz_f, ny_f, nx_f = arr_f.shape
    nz_c = nz_f
    ny_c = (ny_f + 1) // 2
    nx_c = (nx_f + 1) // 2

    pad_z = 0
    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    arr_p = np.pad(arr_f, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="edge")
    arr_c = arr_p.reshape(nz_c, 1, ny_c, 2, nx_c, 2).mean(axis=(1, 3, 5), dtype=np.float64)
    return arr_c.astype(NP_FLOAT, copy=False)


def _coarsen_max_edge_1x2x2(mask_f: np.ndarray) -> np.ndarray:
    """
    Horizontally semi-coarsen a 3D mask while preserving layer index.
    """
    arr_f = np.asarray(mask_f, dtype=np.int32)
    nz_f, ny_f, nx_f = arr_f.shape
    nz_c = nz_f
    ny_c = (ny_f + 1) // 2
    nx_c = (nx_f + 1) // 2

    pad_z = 0
    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    arr_p = np.pad(arr_f, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="edge")
    arr_c = arr_p.reshape(nz_c, 1, ny_c, 2, nx_c, 2).max(axis=(1, 3, 5))
    return arr_c.astype(np.int32, copy=False)


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

    info = {
        "solver_type": "chebyshev_7point_3d",
        "n_iter_used": int(n_iter_used),
        "max_iter": int(max_iter),
        "cheby_order": int(len(omegas)),
        "cheby_omegas": [float(v) for v in omegas],
        "r_rms0": float(r_rms0),
        "r_rms_end": float(r_rms_end),
        "tol_abs": float(tol_abs),
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "transient": bool(transient),
        "transient_formulation": "confined" if bool(transient) else "steady",
        "dt": float(dt_used) if bool(transient) else float("nan"),
        "unconfined": False,
        "converged": bool(converged),
    }

    return (head_out, info) if return_info else head_out


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
        raise NotImplementedError(
            "Transient unconfined 3D solves are scaffolded but not implemented yet. "
            "Use transient confined or steady unconfined mode."
        )

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

        if initial_head is None:
            h_iter = (
                zbot.astype(np.float64, copy=False) + float(min_sat)
            ).astype(NP_FLOAT, copy=False)
        else:
            h_iter = np.asarray(initial_head, dtype=NP_FLOAT).copy()
            if h_iter.shape != shape:
                raise ValueError(f"initial_head shape {h_iter.shape} expected {shape}")
            if not np.all(np.isfinite(h_iter)):
                raise ValueError("initial_head must be finite.")

        h_iter[bcm != 0] = bcv[bcm != 0]
        h_iter[act == 0] = NP_FLOAT(0.0)

        nonlinear_converged = False
        pic_used = 0
        dh_rms_last = float("nan")
        dh_max_last = float("nan")
        last_linear_info = {}

        for pic_it in range(n_pic):
            pic_used = pic_it + 1

            sat = np.maximum(
                h_iter.astype(np.float64, copy=False) - zbot.astype(np.float64, copy=False),
                float(min_sat),
            ).astype(NP_FLOAT, copy=False)

            txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i = build_7point_face_conductance_from_k(
                kx_field=(kx.astype(np.float64, copy=False) * sat.astype(np.float64, copy=False)).astype(
                    NP_FLOAT, copy=False
                ),
                ky_field=(ky.astype(np.float64, copy=False) * sat.astype(np.float64, copy=False)).astype(
                    NP_FLOAT, copy=False
                ),
                kz_field=(kz.astype(np.float64, copy=False) * sat.astype(np.float64, copy=False)).astype(
                    NP_FLOAT, copy=False
                ),
                active=act,
                dx=float(dx),
                dy=float(dx) if dy is None else float(dy),
                dz=float(dz),
            )

            h_lin, info_lin = _solve_chebyshev_7point_3d_linear(
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
                initial_head=h_iter,
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
                return_info=True,
            )

            last_linear_info = info_lin if isinstance(info_lin, dict) else {}
            h_lin = np.asarray(h_lin, dtype=NP_FLOAT)

            h_next = (
                h_iter.astype(np.float64, copy=False)
                + pic_relax * (h_lin.astype(np.float64, copy=False) - h_iter.astype(np.float64, copy=False))
            ).astype(NP_FLOAT, copy=False)
            h_next[bcm != 0] = bcv[bcm != 0]
            h_next[act == 0] = NP_FLOAT(0.0)

            if n_free > 0:
                dh = (
                    h_next.astype(np.float64, copy=False) - h_iter.astype(np.float64, copy=False)
                )[free]
                dh_rms_last = float(np.sqrt(np.mean(dh * dh)))
                dh_max_last = float(np.max(np.abs(dh)))
            else:
                dh_rms_last = 0.0
                dh_max_last = 0.0

            h_iter = h_next
            if dh_rms_last <= pic_tol:
                nonlinear_converged = True
                break

        info_out = dict(last_linear_info) if isinstance(last_linear_info, dict) else {}
        info_out["solver_type"] = "chebyshev_7point_3d_unconfined_picard"
        info_out["linear_solver_type"] = str(last_linear_info.get("solver_type", "chebyshev_7point_3d"))
        info_out["unconfined"] = True
        info_out["picard_converged"] = bool(nonlinear_converged)
        info_out["picard_n_iter_used"] = int(pic_used)
        info_out["picard_max_iter"] = int(n_pic)
        info_out["picard_relax"] = float(pic_relax)
        info_out["picard_head_tol"] = float(pic_tol)
        info_out["picard_dh_rms_end"] = float(dh_rms_last)
        info_out["picard_dh_max_end"] = float(dh_max_last)
        info_out["unconfined_min_sat"] = float(min_sat)
        return (h_iter, info_out) if return_info else h_iter

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
        raise NotImplementedError(
            "Transient unconfined 3D solves are scaffolded but not implemented yet. "
            "Use transient confined or steady unconfined mode."
        )

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

        if initial_head is None:
            h_iter = (
                zbot.astype(np.float64, copy=False) + float(min_sat)
            ).astype(NP_FLOAT, copy=False)
        else:
            h_iter = np.asarray(initial_head, dtype=NP_FLOAT).copy()
            if h_iter.shape != shape0:
                raise ValueError(f"initial_head shape {h_iter.shape} expected {shape0}")

        h_iter[bcm0 != 0] = bcv0[bcm0 != 0]
        h_iter[act0 == 0] = NP_FLOAT(0.0)

        nonlinear_converged = False
        pic_used = 0
        dh_rms_last = float("nan")
        dh_max_last = float("nan")
        last_linear_info = {}

        for pic_it in range(n_pic):
            pic_used = pic_it + 1

            sat = np.maximum(
                h_iter.astype(np.float64, copy=False) - zbot.astype(np.float64, copy=False),
                float(min_sat),
            ).astype(NP_FLOAT, copy=False)

            txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i = build_7point_face_conductance_from_k(
                kx_field=(kx.astype(np.float64, copy=False) * sat.astype(np.float64, copy=False)).astype(
                    NP_FLOAT, copy=False
                ),
                ky_field=(ky.astype(np.float64, copy=False) * sat.astype(np.float64, copy=False)).astype(
                    NP_FLOAT, copy=False
                ),
                kz_field=(kz.astype(np.float64, copy=False) * sat.astype(np.float64, copy=False)).astype(
                    NP_FLOAT, copy=False
                ),
                active=act0,
                dx=float(dx),
                dy=float(dx) if dy is None else float(dy),
                dz=float(dz),
            )

            h_lin, info_lin = solve_multigrid_kcycle_7point_3d(
                tx_p=txp_i,
                tx_m=txm_i,
                ty_p=typ_i,
                ty_m=tym_i,
                tz_p=tzp_i,
                tz_m=tzm_i,
                rhs=b0,
                active=act0,
                bc_mask=bcm0,
                bc_values=bcv0,
                initial_head=h_iter,
                storage_diag=storage_diag,
                max_cycles=int(max_cycles),
                nu_pre=int(nu_pre),
                nu_post=int(nu_post),
                nu_coarse=int(nu_coarse),
                max_levels=int(max_levels),
                min_coarse_n=int(min_coarse_n),
                smoother=str(smoother),
                omega=float(omega),
                cheby_lambda_min=float(cheby_lambda_min),
                cheby_lambda_max=float(cheby_lambda_max),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                check_every_no=int(check_every_no),
                dh_rms_tol=dh_rms_tol,
                dh_max_tol=dh_max_tol,
                dh_max_factor=float(dh_max_factor),
                transient=bool(transient),
                storage_coeff=storage_coeff,
                dt=dt,
                head_prev=head_prev,
                dx=float(dx),
                dy=dy,
                dz=float(dz),
                unconfined=False,
                device=str(device),
                return_info=True,
            )

            last_linear_info = info_lin if isinstance(info_lin, dict) else {}
            h_lin = np.asarray(h_lin, dtype=NP_FLOAT)

            h_next = (
                h_iter.astype(np.float64, copy=False)
                + pic_relax * (h_lin.astype(np.float64, copy=False) - h_iter.astype(np.float64, copy=False))
            ).astype(NP_FLOAT, copy=False)
            h_next[bcm0 != 0] = bcv0[bcm0 != 0]
            h_next[act0 == 0] = NP_FLOAT(0.0)

            if n_free0 > 0:
                dh = (
                    h_next.astype(np.float64, copy=False) - h_iter.astype(np.float64, copy=False)
                )[free0]
                dh_rms_last = float(np.sqrt(np.mean(dh * dh)))
                dh_max_last = float(np.max(np.abs(dh)))
            else:
                dh_rms_last = 0.0
                dh_max_last = 0.0

            h_iter = h_next
            if dh_rms_last <= pic_tol:
                nonlinear_converged = True
                break

        info_out = dict(last_linear_info) if isinstance(last_linear_info, dict) else {}
        info_out["solver_type"] = "kcycle_7point_3d_unconfined_picard"
        info_out["linear_solver_type"] = str(last_linear_info.get("solver_type", "kcycle_7point_3d"))
        info_out["unconfined"] = True
        info_out["picard_converged"] = bool(nonlinear_converged)
        info_out["picard_n_iter_used"] = int(pic_used)
        info_out["picard_max_iter"] = int(n_pic)
        info_out["picard_relax"] = float(pic_relax)
        info_out["picard_head_tol"] = float(pic_tol)
        info_out["picard_dh_rms_end"] = float(dh_rms_last)
        info_out["picard_dh_max_end"] = float(dh_max_last)
        info_out["unconfined_min_sat"] = float(min_sat)
        return (h_iter, info_out) if return_info else h_iter

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
    if smooth_mode not in {"chebyshev", "jacobi"}:
        raise ValueError("smoother must be 'chebyshev' or 'jacobi'.")
    if smooth_mode == "chebyshev":
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
        omg = float(omega)
        pre_omegas = tuple(omg for _ in range(int(nu_pre)))
        post_omegas = tuple(omg for _ in range(int(nu_post)))
    if len(pre_omegas) == 0:
        pre_omegas = (1.0,)
    if len(post_omegas) == 0:
        post_omegas = (1.0,)

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
        }
    )

    for _ in range(1, int(max_levels)):
        prev = levels_host[-1]
        nz_f, ny_f, nx_f = prev["active"].shape

        nz_c = int(nz_f)
        ny_c = (int(ny_f) + 1) // 2
        nx_c = (int(nx_f) + 1) // 2
        if (ny_c, nx_c) == (ny_f, nx_f):
            break
        if nx_c < int(min_coarse_n) or ny_c < int(min_coarse_n):
            break

        active_c = _coarsen_max_edge_1x2x2(prev["active"])
        bc_mask_c = _coarsen_max_edge_1x2x2(prev["bc_mask"])
        bc_values_c = np.zeros((nz_c, ny_c, nx_c), dtype=NP_FLOAT)

        storage_c = _coarsen_mean_edge_1x2x2(prev["storage_diag"])
        free_c = (active_c != 0) & (bc_mask_c == 0)
        storage_c[~free_c] = NP_FLOAT(0.0)

        levels_host.append(
            {
                "tx_p": _coarsen_mean_edge_1x2x2(prev["tx_p"]),
                "tx_m": _coarsen_mean_edge_1x2x2(prev["tx_m"]),
                "ty_p": _coarsen_mean_edge_1x2x2(prev["ty_p"]),
                "ty_m": _coarsen_mean_edge_1x2x2(prev["ty_m"]),
                "tz_p": _coarsen_mean_edge_1x2x2(prev["tz_p"]),
                "tz_m": _coarsen_mean_edge_1x2x2(prev["tz_m"]),
                "active": active_c,
                "bc_mask": bc_mask_c,
                "bc_values": bc_values_c,
                "storage_diag": storage_c,
                "b0": np.zeros((nz_c, ny_c, nx_c), dtype=NP_FLOAT),
            }
        )

    levels: list[dict] = []
    for lid, lh in enumerate(levels_host):
        nz, ny, nx = lh["active"].shape
        shapeL = (nz, ny, nx)
        dimL = (nz, ny, nx)

        M_inv_l = build_diag_preconditioner_7point(
            tx_p=lh["tx_p"],
            tx_m=lh["tx_m"],
            ty_p=lh["ty_p"],
            ty_m=lh["ty_m"],
            tz_p=lh["tz_p"],
            tz_m=lh["tz_m"],
            active=lh["active"],
            bc_mask=lh["bc_mask"],
            storage_diag=lh["storage_diag"],
        )

        lvl = {
            "level_id": int(lid),
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "shape": shapeL,
            "dim": dimL,
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
            "M_inv_wp": wp.array(M_inv_l, dtype=WP_FLOAT, device=device),
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
        }
        levels.append(lvl)

    lvl0 = levels[0]
    lvl0["x_wp"] = wp.array(x0, dtype=WP_FLOAT, device=device)
    x_prev_check = np.asarray(x0, dtype=np.float64).copy()

    for lid in range(1, len(levels)):
        levels[lid]["x_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["b_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["r_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["Ax_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["e_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["z1_wp"].fill_(WP_FLOAT(0.0))
        levels[lid]["r1_wp"].fill_(WP_FLOAT(0.0))

    def smooth_level(level: dict, omegas: tuple[float, ...]) -> None:
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
        for _ in range(int(nu_coarse)):
            smooth_level(level, pre_omegas)

    def kcycle(level_id: int) -> None:
        level = levels[level_id]
        smooth_level(level, pre_omegas)
        compute_residual_norm(level, level["x_wp"], level["r_wp"], level["rTr_buf"])

        if level_id == (len(levels) - 1):
            coarsest_relax(level)
            return

        coarse = levels[level_id + 1]

        wp.launch(
            kernel=restrict_blockavg_xy_3d_kernel,
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
            kernel=prolong_bilinear_xy_3d_kernel,
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

        smooth_level(level, post_omegas)

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

        x_now = np.asarray(lvl0["x_wp"].numpy(), dtype=np.float64)
        dh = (x_now - x_prev_check)[free0]
        if dh.size > 0:
            dh_rms_lastcheck = float(np.sqrt(np.mean(dh * dh)))
            dh_max_lastcheck = float(np.max(np.abs(dh)))
        else:
            dh_rms_lastcheck = 0.0
            dh_max_lastcheck = 0.0
        x_prev_check = x_now

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

    info = {
        "solver_type": "kcycle_7point_3d",
        "n_levels": int(len(levels)),
        "coarsening_mode": "horizontal",
        "level_shapes": [tuple(lh["active"].shape) for lh in levels_host],
        "max_cycles": int(max_cycles_i),
        "n_cycles_used": int(n_cycles_used),
        "nu_pre": int(nu_pre),
        "nu_post": int(nu_post),
        "nu_coarse": int(nu_coarse),
        "smoother": str(smooth_mode),
        "omega": float(omega),
        "cheby_lambda_min": float(cheby_lambda_min) if smooth_mode == "chebyshev" else float("nan"),
        "cheby_lambda_max": float(cheby_lambda_max) if smooth_mode == "chebyshev" else float("nan"),
        "cheby_pre_omegas": [float(v) for v in pre_omegas],
        "cheby_post_omegas": [float(v) for v in post_omegas],
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "tol_abs": float(tol_abs),
        "r_rms0": float(r_rms0),
        "r_rms_end": float(r_rms_end),
        "dh_rms_lastcheck": float(dh_rms_lastcheck),
        "dh_max_lastcheck": float(dh_max_lastcheck),
        "converged": bool(converged),
        "transient": bool(transient),
        "transient_formulation": "confined" if bool(transient) else "steady",
        "dt": float(dt_used) if bool(transient) else float("nan"),
        "unconfined": False,
    }

    return (head_out, info) if return_info else head_out


__all__ = [
    "_solve_chebyshev_7point_3d_linear",
    "solve_chebyshev_7point_3d",
    "solve_multigrid_kcycle_7point_3d",
]
