# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import warp as wp

from DARCY_WARP_PACKAGE.config import WP_FLOAT

__all__ = [
    "add_correction_3d_kernel",
    "apply_A_7point_kernel",
    "apply_A_and_pAp_7point_kernel",
    "axpy_active_scalar_3d_kernel",
    "build_diag_preconditioner_7point_kernel",
    "copy_field_3d_kernel",
    "compute_residual_7point_kernel",
    "compute_head_residual_7point_kernel",
    "dh_change_reduce_3d_kernel",
    "dot_active_3d_kernel",
    "jacobi_applyA_fused_7point_kernel",
    "prolong_trilinear_any_3d_kernel",
    "prolong_bilinear_xy_3d_kernel",
    "restrict_blockavg_3d_kernel",
    "restrict_blockavg_xy_3d_kernel",
    "vertical_line_relaxation_7point_kernel",
    "zero_scalar_kernel",
]


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
def apply_A_7point_kernel(
    tx_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_m: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_p: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_m: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_m: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=3),
    h: wp.array(dtype=WP_FLOAT, ndim=3),
    Ah: wp.array(dtype=WP_FLOAT, ndim=3),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return

    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        Ah[k, j, i] = h[k, j, i]
        return

    tiny = wp.float64(1.0e-12)
    hC = wp.float64(h[k, j, i])

    cxp = wp.float64(tx_p[k, j, i])
    cxm = wp.float64(tx_m[k, j, i])
    cyp = wp.float64(ty_p[k, j, i])
    cym = wp.float64(ty_m[k, j, i])
    czp = wp.float64(tz_p[k, j, i])
    czm = wp.float64(tz_m[k, j, i])
    sdiag = wp.float64(storage_diag[k, j, i])

    if cxp < wp.float64(0.0):
        cxp = wp.float64(0.0)
    if cxm < wp.float64(0.0):
        cxm = wp.float64(0.0)
    if cyp < wp.float64(0.0):
        cyp = wp.float64(0.0)
    if cym < wp.float64(0.0):
        cym = wp.float64(0.0)
    if czp < wp.float64(0.0):
        czp = wp.float64(0.0)
    if czm < wp.float64(0.0):
        czm = wp.float64(0.0)
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)

    # Drop couplings that leave the domain or target inactive cells.
    if i + 1 >= nx or active[k, j, i + 1] == 0:
        cxp = wp.float64(0.0)
    if i - 1 < 0 or active[k, j, i - 1] == 0:
        cxm = wp.float64(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0:
        cyp = wp.float64(0.0)
    if j - 1 < 0 or active[k, j - 1, i] == 0:
        cym = wp.float64(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0:
        czp = wp.float64(0.0)
    if k - 1 < 0 or active[k - 1, j, i] == 0:
        czm = wp.float64(0.0)

    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    if diag < tiny:
        Ah[k, j, i] = WP_FLOAT(hC)
        return

    val = diag * hC
    if cxp > wp.float64(0.0):
        val = val - cxp * wp.float64(h[k, j, i + 1])
    if cxm > wp.float64(0.0):
        val = val - cxm * wp.float64(h[k, j, i - 1])
    if cyp > wp.float64(0.0):
        val = val - cyp * wp.float64(h[k, j + 1, i])
    if cym > wp.float64(0.0):
        val = val - cym * wp.float64(h[k, j - 1, i])
    if czp > wp.float64(0.0):
        val = val - czp * wp.float64(h[k + 1, j, i])
    if czm > wp.float64(0.0):
        val = val - czm * wp.float64(h[k - 1, j, i])

    Ah[k, j, i] = WP_FLOAT(val)


@wp.kernel
def vertical_line_relaxation_7point_kernel(
    tx_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_m: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_p: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_m: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_m: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=3),
    rhs: wp.array(dtype=WP_FLOAT, ndim=3),
    x_old: wp.array(dtype=WP_FLOAT, ndim=3),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=3),
    omega: float,
    c_prime: wp.array(dtype=WP_FLOAT, ndim=3),
    d_prime: wp.array(dtype=WP_FLOAT, ndim=3),
    nx: int,
    ny: int,
    nz: int,
    x_new: wp.array(dtype=WP_FLOAT, ndim=3),
):
    xy_idx = wp.tid()
    i = xy_idx % nx
    j = xy_idx // nx

    # Forward sweep
    for k in range(nz):
        act = active[k, j, i]
        bcm = bc_mask[k, j, i]

        if act > 0 and bcm == 0:
            diag = wp.float64(storage_diag[k, j, i])
            if diag < wp.float64(0.0):
                diag = wp.float64(0.0)
            v = wp.float64(rhs[k, j, i])

            # x+
            cxp = wp.float64(tx_p[k, j, i])
            if cxp < wp.float64(0.0):
                cxp = wp.float64(0.0)
            if i + 1 >= nx or active[k, j, i + 1] == 0:
                cxp = wp.float64(0.0)
            if cxp > wp.float64(0.0):
                diag += cxp
                if bc_mask[k, j, i + 1] > 0:
                    v += cxp * wp.float64(bc_values[k, j, i + 1])
                else:
                    v += cxp * wp.float64(x_old[k, j, i + 1])

            # x-
            cxm = wp.float64(tx_m[k, j, i])
            if cxm < wp.float64(0.0):
                cxm = wp.float64(0.0)
            if i - 1 < 0 or active[k, j, i - 1] == 0:
                cxm = wp.float64(0.0)
            if cxm > wp.float64(0.0):
                diag += cxm
                if bc_mask[k, j, i - 1] > 0:
                    v += cxm * wp.float64(bc_values[k, j, i - 1])
                else:
                    v += cxm * wp.float64(x_old[k, j, i - 1])

            # y+
            cyp = wp.float64(ty_p[k, j, i])
            if cyp < wp.float64(0.0):
                cyp = wp.float64(0.0)
            if j + 1 >= ny or active[k, j + 1, i] == 0:
                cyp = wp.float64(0.0)
            if cyp > wp.float64(0.0):
                diag += cyp
                if bc_mask[k, j + 1, i] > 0:
                    v += cyp * wp.float64(bc_values[k, j + 1, i])
                else:
                    v += cyp * wp.float64(x_old[k, j + 1, i])

            # y-
            cym = wp.float64(ty_m[k, j, i])
            if cym < wp.float64(0.0):
                cym = wp.float64(0.0)
            if j - 1 < 0 or active[k, j - 1, i] == 0:
                cym = wp.float64(0.0)
            if cym > wp.float64(0.0):
                diag += cym
                if bc_mask[k, j - 1, i] > 0:
                    v += cym * wp.float64(bc_values[k, j - 1, i])
                else:
                    v += cym * wp.float64(x_old[k, j - 1, i])

            a_k = wp.float64(0.0)
            c_k = wp.float64(0.0)

            # z- (a_k)
            czm = wp.float64(tz_m[k, j, i])
            if czm < wp.float64(0.0):
                czm = wp.float64(0.0)
            if k - 1 < 0 or active[k - 1, j, i] == 0:
                czm = wp.float64(0.0)
            if czm > wp.float64(0.0):
                diag += czm
                if bc_mask[k - 1, j, i] > 0:
                    v += czm * wp.float64(bc_values[k - 1, j, i])
                else:
                    a_k = -czm

            # z+ (c_k)
            czp = wp.float64(tz_p[k, j, i])
            if czp < wp.float64(0.0):
                czp = wp.float64(0.0)
            if k + 1 >= nz or active[k + 1, j, i] == 0:
                czp = wp.float64(0.0)
            if czp > wp.float64(0.0):
                diag += czp
                if bc_mask[k + 1, j, i] > 0:
                    v += czp * wp.float64(bc_values[k + 1, j, i])
                else:
                    c_k = -czp

            if diag < wp.float64(1.0e-12):
                b_k = wp.float64(1.0)
                d_k = v
            else:
                b_k = diag
                d_k = v

        elif act > 0 and bcm > 0:
            a_k = wp.float64(0.0)
            b_k = wp.float64(1.0)
            c_k = wp.float64(0.0)
            d_k = wp.float64(bc_values[k, j, i])
        else:
            a_k = wp.float64(0.0)
            b_k = wp.float64(1.0)
            c_k = wp.float64(0.0)
            d_k = wp.float64(x_old[k, j, i])

        if k == 0:
            c_prime[k, j, i] = WP_FLOAT(c_k / b_k)
            d_prime[k, j, i] = WP_FLOAT(d_k / b_k)
        else:
            c_prev = wp.float64(c_prime[k - 1, j, i])
            d_prev = wp.float64(d_prime[k - 1, j, i])
            denom = b_k - a_k * c_prev
            if wp.abs(denom) > wp.float64(1e-30):
                c_prime[k, j, i] = WP_FLOAT(c_k / denom)
                d_prime[k, j, i] = WP_FLOAT((d_k - a_k * d_prev) / denom)
            else:
                c_prime[k, j, i] = WP_FLOAT(0.0)
                d_prime[k, j, i] = WP_FLOAT(0.0)

    # Backward sweep
    x_new_val_prev = wp.float64(0.0)
    for k_inv in range(nz):
        k = nz - 1 - k_inv
        
        c_p = wp.float64(c_prime[k, j, i])
        d_p = wp.float64(d_prime[k, j, i])
        
        if k == nz - 1:
            x_new_val = d_p
        else:
            x_new_val = d_p - c_p * x_new_val_prev
            
        x_new_val_prev = x_new_val
        
        act = active[k, j, i]
        bcm = bc_mask[k, j, i]
        if act > 0 and bcm == 0:
            x_old_val = wp.float64(x_old[k, j, i])
            x_new[k, j, i] = WP_FLOAT(x_old_val + wp.float64(omega) * (x_new_val - x_old_val))
        elif act > 0 and bcm > 0:
            x_new[k, j, i] = bc_values[k, j, i]
        else:
            x_new[k, j, i] = WP_FLOAT(0.0)


@wp.kernel
def jacobi_applyA_fused_7point_kernel(
    tx_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_m: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_p: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_m: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_m: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=3),
    b: wp.array(dtype=WP_FLOAT, ndim=3),
    x_in: wp.array(dtype=WP_FLOAT, ndim=3),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=3),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=3),
    omega: float,
    nx: int,
    ny: int,
    nz: int,
    x_out: wp.array(dtype=WP_FLOAT, ndim=3),
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return

    if active[k, j, i] == 0:
        x_out[k, j, i] = WP_FLOAT(0.0)
        return
    if bc_mask[k, j, i] != 0:
        x_out[k, j, i] = bc_values[k, j, i]
        return

    tiny = wp.float64(1.0e-12)
    hC = wp.float64(x_in[k, j, i])

    cxp = wp.float64(tx_p[k, j, i])
    cxm = wp.float64(tx_m[k, j, i])
    cyp = wp.float64(ty_p[k, j, i])
    cym = wp.float64(ty_m[k, j, i])
    czp = wp.float64(tz_p[k, j, i])
    czm = wp.float64(tz_m[k, j, i])
    sdiag = wp.float64(storage_diag[k, j, i])

    if cxp < wp.float64(0.0):
        cxp = wp.float64(0.0)
    if cxm < wp.float64(0.0):
        cxm = wp.float64(0.0)
    if cyp < wp.float64(0.0):
        cyp = wp.float64(0.0)
    if cym < wp.float64(0.0):
        cym = wp.float64(0.0)
    if czp < wp.float64(0.0):
        czp = wp.float64(0.0)
    if czm < wp.float64(0.0):
        czm = wp.float64(0.0)
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)

    if i + 1 >= nx or active[k, j, i + 1] == 0:
        cxp = wp.float64(0.0)
    if i - 1 < 0 or active[k, j, i - 1] == 0:
        cxm = wp.float64(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0:
        cyp = wp.float64(0.0)
    if j - 1 < 0 or active[k, j - 1, i] == 0:
        cym = wp.float64(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0:
        czp = wp.float64(0.0)
    if k - 1 < 0 or active[k - 1, j, i] == 0:
        czm = wp.float64(0.0)

    diag = cxp + cxm + cyp + cym + czp + czm + sdiag

    Ah = wp.float64(0.0)
    if diag < tiny:
        Ah = hC
    else:
        Ah = diag * hC
        if cxp > wp.float64(0.0):
            Ah = Ah - cxp * wp.float64(x_in[k, j, i + 1])
        if cxm > wp.float64(0.0):
            Ah = Ah - cxm * wp.float64(x_in[k, j, i - 1])
        if cyp > wp.float64(0.0):
            Ah = Ah - cyp * wp.float64(x_in[k, j + 1, i])
        if cym > wp.float64(0.0):
            Ah = Ah - cym * wp.float64(x_in[k, j - 1, i])
        if czp > wp.float64(0.0):
            Ah = Ah - czp * wp.float64(x_in[k + 1, j, i])
        if czm > wp.float64(0.0):
            Ah = Ah - czm * wp.float64(x_in[k - 1, j, i])

    r_ijk = wp.float64(b[k, j, i]) - Ah
    x_out[k, j, i] = WP_FLOAT(hC + wp.float64(omega) * wp.float64(M_inv[k, j, i]) * r_ijk)


@wp.kernel
def compute_residual_7point_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=3),
    b: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_m: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_p: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_m: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_m: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=3),
    r: wp.array(dtype=WP_FLOAT, ndim=3),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return

    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        r[k, j, i] = WP_FLOAT(0.0)
        return

    tiny = wp.float64(1.0e-12)
    hC = wp.float64(x[k, j, i])

    cxp = wp.float64(tx_p[k, j, i])
    cxm = wp.float64(tx_m[k, j, i])
    cyp = wp.float64(ty_p[k, j, i])
    cym = wp.float64(ty_m[k, j, i])
    czp = wp.float64(tz_p[k, j, i])
    czm = wp.float64(tz_m[k, j, i])
    sdiag = wp.float64(storage_diag[k, j, i])

    if cxp < wp.float64(0.0):
        cxp = wp.float64(0.0)
    if cxm < wp.float64(0.0):
        cxm = wp.float64(0.0)
    if cyp < wp.float64(0.0):
        cyp = wp.float64(0.0)
    if cym < wp.float64(0.0):
        cym = wp.float64(0.0)
    if czp < wp.float64(0.0):
        czp = wp.float64(0.0)
    if czm < wp.float64(0.0):
        czm = wp.float64(0.0)
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)

    if i + 1 >= nx or active[k, j, i + 1] == 0:
        cxp = wp.float64(0.0)
    if i - 1 < 0 or active[k, j, i - 1] == 0:
        cxm = wp.float64(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0:
        cyp = wp.float64(0.0)
    if j - 1 < 0 or active[k, j - 1, i] == 0:
        cym = wp.float64(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0:
        czp = wp.float64(0.0)
    if k - 1 < 0 or active[k - 1, j, i] == 0:
        czm = wp.float64(0.0)

    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    Ax = wp.float64(0.0)
    if diag < tiny:
        Ax = hC
    else:
        Ax = diag * hC
        if cxp > wp.float64(0.0):
            Ax = Ax - cxp * wp.float64(x[k, j, i + 1])
        if cxm > wp.float64(0.0):
            Ax = Ax - cxm * wp.float64(x[k, j, i - 1])
        if cyp > wp.float64(0.0):
            Ax = Ax - cyp * wp.float64(x[k, j + 1, i])
        if cym > wp.float64(0.0):
            Ax = Ax - cym * wp.float64(x[k, j - 1, i])
        if czp > wp.float64(0.0):
            Ax = Ax - czp * wp.float64(x[k + 1, j, i])
        if czm > wp.float64(0.0):
            Ax = Ax - czm * wp.float64(x[k - 1, j, i])

    rf64 = wp.float64(b[k, j, i]) - Ax
    r[k, j, i] = WP_FLOAT(rf64)
    wp.atomic_add(rTr_buf, 0, rf64 * rf64)


@wp.kernel
def copy_field_3d_kernel(
    src: wp.array(dtype=WP_FLOAT, ndim=3),
    dst: wp.array(dtype=WP_FLOAT, ndim=3),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return
    dst[k, j, i] = src[k, j, i]


@wp.kernel
def dh_change_reduce_3d_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=3),
    x_prev: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    dh2_buf: wp.array(dtype=wp.float64, ndim=1),
    dh_max_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return

    x_new = wp.float64(x[k, j, i])
    x_old = wp.float64(x_prev[k, j, i])
    x_prev[k, j, i] = x[k, j, i]

    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        return

    dh = x_new - x_old
    abs_dh = wp.abs(dh)
    wp.atomic_add(dh2_buf, 0, dh * dh)
    wp.atomic_max(dh_max_buf, 0, abs_dh)


@wp.kernel
def add_correction_3d_kernel(
    x_f: wp.array(dtype=WP_FLOAT, ndim=3),
    e_f: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    bc_values: wp.array(dtype=WP_FLOAT, ndim=3),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return
    if active[k, j, i] == 0:
        x_f[k, j, i] = WP_FLOAT(0.0)
        return
    if bc_mask[k, j, i] != 0:
        x_f[k, j, i] = bc_values[k, j, i]
        return
    x_f[k, j, i] = x_f[k, j, i] + e_f[k, j, i]


@wp.kernel
def restrict_blockavg_3d_kernel(
    r_f: wp.array(dtype=WP_FLOAT, ndim=3),
    active_f: wp.array(dtype=wp.int32, ndim=3),
    bc_mask_f: wp.array(dtype=wp.int32, ndim=3),
    b_c: wp.array(dtype=WP_FLOAT, ndim=3),
    nx_f: int,
    ny_f: int,
    nz_f: int,
    nx_c: int,
    ny_c: int,
    nz_c: int,
):
    kc, jc, ic = wp.tid()
    if kc >= nz_c or jc >= ny_c or ic >= nx_c:
        return

    k0 = 2 * kc
    j0 = 2 * jc
    i0 = 2 * ic

    s = WP_FLOAT(0.0)
    n = WP_FLOAT(0.0)

    for dk in range(2):
        kf = k0 + dk
        if kf >= nz_f:
            continue
        for dj in range(2):
            jf = j0 + dj
            if jf >= ny_f:
                continue
            for di in range(2):
                i_f = i0 + di
                if i_f >= nx_f:
                    continue
                if active_f[kf, jf, i_f] != 0 and bc_mask_f[kf, jf, i_f] == 0:
                    s = s + r_f[kf, jf, i_f]
                    n = n + WP_FLOAT(1.0)

    if n > WP_FLOAT(0.0):
        b_c[kc, jc, ic] = s / n
    else:
        b_c[kc, jc, ic] = WP_FLOAT(0.0)


@wp.kernel
def prolong_trilinear_any_3d_kernel(
    x_c: wp.array(dtype=WP_FLOAT, ndim=3),
    e_f: wp.array(dtype=WP_FLOAT, ndim=3),
    nx_f: int,
    ny_f: int,
    nz_f: int,
    nx_c: int,
    ny_c: int,
    nz_c: int,
):
    k, j, i = wp.tid()
    if k >= nz_f or j >= ny_f or i >= nx_f:
        return

    kc = k // 2
    jc = j // 2
    ic = i // 2

    fz = WP_FLOAT(0.0)
    fy = WP_FLOAT(0.0)
    fx = WP_FLOAT(0.0)
    if (k & 1) == 1:
        fz = WP_FLOAT(0.5)
    if (j & 1) == 1:
        fy = WP_FLOAT(0.5)
    if (i & 1) == 1:
        fx = WP_FLOAT(0.5)

    kc1 = kc + 1
    jc1 = jc + 1
    ic1 = ic + 1
    if kc1 >= nz_c:
        kc1 = nz_c - 1
    if jc1 >= ny_c:
        jc1 = ny_c - 1
    if ic1 >= nx_c:
        ic1 = nx_c - 1

    c000 = x_c[kc, jc, ic]
    c001 = x_c[kc, jc, ic1]
    c010 = x_c[kc, jc1, ic]
    c011 = x_c[kc, jc1, ic1]
    c100 = x_c[kc1, jc, ic]
    c101 = x_c[kc1, jc, ic1]
    c110 = x_c[kc1, jc1, ic]
    c111 = x_c[kc1, jc1, ic1]

    one = WP_FLOAT(1.0)
    wx0 = one - fx
    wx1 = fx
    wy0 = one - fy
    wy1 = fy
    wz0 = one - fz
    wz1 = fz

    e_f[k, j, i] = (
        c000 * wx0 * wy0 * wz0
        + c001 * wx1 * wy0 * wz0
        + c010 * wx0 * wy1 * wz0
        + c011 * wx1 * wy1 * wz0
        + c100 * wx0 * wy0 * wz1
        + c101 * wx1 * wy0 * wz1
        + c110 * wx0 * wy1 * wz1
        + c111 * wx1 * wy1 * wz1
    )


@wp.kernel
def restrict_blockavg_xy_3d_kernel(
    r_f: wp.array(dtype=WP_FLOAT, ndim=3),
    active_f: wp.array(dtype=wp.int32, ndim=3),
    bc_mask_f: wp.array(dtype=wp.int32, ndim=3),
    b_c: wp.array(dtype=WP_FLOAT, ndim=3),
    nx_f: int,
    ny_f: int,
    nz_f: int,
    nx_c: int,
    ny_c: int,
    nz_c: int,
):
    kc, jc, ic = wp.tid()
    if kc >= nz_c or jc >= ny_c or ic >= nx_c:
        return

    kf = kc
    j0 = 2 * jc
    i0 = 2 * ic

    s = WP_FLOAT(0.0)
    n = WP_FLOAT(0.0)

    for dj in range(2):
        jf = j0 + dj
        if jf >= ny_f:
            continue
        for di in range(2):
            i_f = i0 + di
            if i_f >= nx_f:
                continue
            if active_f[kf, jf, i_f] != 0 and bc_mask_f[kf, jf, i_f] == 0:
                s = s + r_f[kf, jf, i_f]
                n = n + WP_FLOAT(1.0)

    if n > WP_FLOAT(0.0):
        b_c[kc, jc, ic] = s / n
    else:
        b_c[kc, jc, ic] = WP_FLOAT(0.0)


@wp.kernel
def prolong_bilinear_xy_3d_kernel(
    x_c: wp.array(dtype=WP_FLOAT, ndim=3),
    e_f: wp.array(dtype=WP_FLOAT, ndim=3),
    nx_f: int,
    ny_f: int,
    nz_f: int,
    nx_c: int,
    ny_c: int,
    nz_c: int,
):
    k, j, i = wp.tid()
    if k >= nz_f or j >= ny_f or i >= nx_f:
        return

    kc = k
    jc = j // 2
    ic = i // 2

    fy = WP_FLOAT(0.0)
    fx = WP_FLOAT(0.0)
    if (j & 1) == 1:
        fy = WP_FLOAT(0.5)
    if (i & 1) == 1:
        fx = WP_FLOAT(0.5)

    jc1 = jc + 1
    ic1 = ic + 1
    if jc1 >= ny_c:
        jc1 = ny_c - 1
    if ic1 >= nx_c:
        ic1 = nx_c - 1

    c00 = x_c[kc, jc, ic]
    c01 = x_c[kc, jc, ic1]
    c10 = x_c[kc, jc1, ic]
    c11 = x_c[kc, jc1, ic1]

    one = WP_FLOAT(1.0)
    wx0 = one - fx
    wx1 = fx
    wy0 = one - fy
    wy1 = fy

    e_f[k, j, i] = (
        c00 * wx0 * wy0
        + c01 * wx1 * wy0
        + c10 * wx0 * wy1
        + c11 * wx1 * wy1
    )


@wp.kernel
def dot_active_3d_kernel(
    a: wp.array(dtype=WP_FLOAT, ndim=3),
    b: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    out_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return
    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        return
    wp.atomic_add(out_buf, 0, wp.float64(a[k, j, i]) * wp.float64(b[k, j, i]))


@wp.kernel
def axpy_active_scalar_3d_kernel(
    y: wp.array(dtype=WP_FLOAT, ndim=3),
    x: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    alpha: float,
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return
    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        return
    y[k, j, i] = y[k, j, i] + WP_FLOAT(alpha) * x[k, j, i]


@wp.kernel
def apply_A_and_pAp_7point_kernel(
    tx_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_m: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_p: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_m: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_m: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=3),
    p: wp.array(dtype=WP_FLOAT, ndim=3),
    Ap: wp.array(dtype=WP_FLOAT, ndim=3),
    pAp_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return

    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        Ap[k, j, i] = p[k, j, i]
        return

    tiny = wp.float64(1.0e-12)
    pC = wp.float64(p[k, j, i])

    cxp = wp.float64(tx_p[k, j, i])
    cxm = wp.float64(tx_m[k, j, i])
    cyp = wp.float64(ty_p[k, j, i])
    cym = wp.float64(ty_m[k, j, i])
    czp = wp.float64(tz_p[k, j, i])
    czm = wp.float64(tz_m[k, j, i])
    sdiag = wp.float64(storage_diag[k, j, i])

    if cxp < wp.float64(0.0):
        cxp = wp.float64(0.0)
    if cxm < wp.float64(0.0):
        cxm = wp.float64(0.0)
    if cyp < wp.float64(0.0):
        cyp = wp.float64(0.0)
    if cym < wp.float64(0.0):
        cym = wp.float64(0.0)
    if czp < wp.float64(0.0):
        czp = wp.float64(0.0)
    if czm < wp.float64(0.0):
        czm = wp.float64(0.0)
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)

    if i + 1 >= nx or active[k, j, i + 1] == 0:
        cxp = wp.float64(0.0)
    if i - 1 < 0 or active[k, j, i - 1] == 0:
        cxm = wp.float64(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0:
        cyp = wp.float64(0.0)
    if j - 1 < 0 or active[k, j - 1, i] == 0:
        cym = wp.float64(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0:
        czp = wp.float64(0.0)
    if k - 1 < 0 or active[k - 1, j, i] == 0:
        czm = wp.float64(0.0)

    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    val64 = wp.float64(0.0)
    if diag < tiny:
        val64 = pC
    else:
        val64 = diag * pC
        if cxp > wp.float64(0.0):
            val64 = val64 - cxp * wp.float64(p[k, j, i + 1])
        if cxm > wp.float64(0.0):
            val64 = val64 - cxm * wp.float64(p[k, j, i - 1])
        if cyp > wp.float64(0.0):
            val64 = val64 - cyp * wp.float64(p[k, j + 1, i])
        if cym > wp.float64(0.0):
            val64 = val64 - cym * wp.float64(p[k, j - 1, i])
        if czp > wp.float64(0.0):
            val64 = val64 - czp * wp.float64(p[k + 1, j, i])
        if czm > wp.float64(0.0):
            val64 = val64 - czm * wp.float64(p[k - 1, j, i])

    Ap[k, j, i] = WP_FLOAT(val64)
    wp.atomic_add(pAp_buf, 0, pC * val64)


@wp.kernel
def build_diag_preconditioner_7point_kernel(
    tx_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_m: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_p: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_m: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_m: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=3),
    M_inv_out: wp.array(dtype=WP_FLOAT, ndim=3),
    nx: int,
    ny: int,
    nz: int,
):
    """
    Device-side Jacobi (diagonal) preconditioner for the 7-point 3D operator.

    Mirrors the host ``build_diag_preconditioner_7point`` so the host/device
    backends produce an identical ``M_inv``: ``M_inv = 1 / (sum of face
    conductances + storage)`` on free active cells where the diagonal exceeds
    ``tiny`` (else 1.0), and 1.0 on inactive / Dirichlet cells.
    """
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return

    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        M_inv_out[k, j, i] = WP_FLOAT(1.0)
        return

    tiny = wp.float64(1.0e-12)
    cxp = wp.float64(tx_p[k, j, i])
    cxm = wp.float64(tx_m[k, j, i])
    cyp = wp.float64(ty_p[k, j, i])
    cym = wp.float64(ty_m[k, j, i])
    czp = wp.float64(tz_p[k, j, i])
    czm = wp.float64(tz_m[k, j, i])
    sdiag = wp.float64(storage_diag[k, j, i])
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)

    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    if diag > tiny:
        M_inv_out[k, j, i] = WP_FLOAT(wp.float64(1.0) / diag)
    else:
        M_inv_out[k, j, i] = WP_FLOAT(1.0)


@wp.kernel
def compute_head_residual_7point_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=3),
    b: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tx_m: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_p: wp.array(dtype=WP_FLOAT, ndim=3),
    ty_m: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_p: wp.array(dtype=WP_FLOAT, ndim=3),
    tz_m: wp.array(dtype=WP_FLOAT, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    storage_diag: wp.array(dtype=WP_FLOAT, ndim=3),
    M_inv: wp.array(dtype=WP_FLOAT, ndim=3),
    r: wp.array(dtype=WP_FLOAT, ndim=3),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
):
    """
    Head-equivalent (Jacobi-preconditioned) residual: ``r_h = M_inv * (b - A x)``.

    Identical to ``compute_residual_7point_kernel`` but multiplies the residual
    by ``M_inv`` before storing and squaring into ``rTr_buf``. The resulting RMS
    is the head-equivalent residual norm (``h_rms_end``) used to drive the
    inexact-Picard inner tolerance in the unconfined solver.
    """
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return

    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        r[k, j, i] = WP_FLOAT(0.0)
        return

    tiny = wp.float64(1.0e-12)
    hC = wp.float64(x[k, j, i])

    cxp = wp.float64(tx_p[k, j, i])
    cxm = wp.float64(tx_m[k, j, i])
    cyp = wp.float64(ty_p[k, j, i])
    cym = wp.float64(ty_m[k, j, i])
    czp = wp.float64(tz_p[k, j, i])
    czm = wp.float64(tz_m[k, j, i])
    sdiag = wp.float64(storage_diag[k, j, i])

    if cxp < wp.float64(0.0):
        cxp = wp.float64(0.0)
    if cxm < wp.float64(0.0):
        cxm = wp.float64(0.0)
    if cyp < wp.float64(0.0):
        cyp = wp.float64(0.0)
    if cym < wp.float64(0.0):
        cym = wp.float64(0.0)
    if czp < wp.float64(0.0):
        czp = wp.float64(0.0)
    if czm < wp.float64(0.0):
        czm = wp.float64(0.0)
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)

    if i + 1 >= nx or active[k, j, i + 1] == 0:
        cxp = wp.float64(0.0)
    if i - 1 < 0 or active[k, j, i - 1] == 0:
        cxm = wp.float64(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0:
        cyp = wp.float64(0.0)
    if j - 1 < 0 or active[k, j - 1, i] == 0:
        cym = wp.float64(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0:
        czp = wp.float64(0.0)
    if k - 1 < 0 or active[k - 1, j, i] == 0:
        czm = wp.float64(0.0)

    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    Ax = wp.float64(0.0)
    if diag < tiny:
        Ax = hC
    else:
        Ax = diag * hC
        if cxp > wp.float64(0.0):
            Ax = Ax - cxp * wp.float64(x[k, j, i + 1])
        if cxm > wp.float64(0.0):
            Ax = Ax - cxm * wp.float64(x[k, j, i - 1])
        if cyp > wp.float64(0.0):
            Ax = Ax - cyp * wp.float64(x[k, j + 1, i])
        if cym > wp.float64(0.0):
            Ax = Ax - cym * wp.float64(x[k, j - 1, i])
        if czp > wp.float64(0.0):
            Ax = Ax - czp * wp.float64(x[k + 1, j, i])
        if czm > wp.float64(0.0):
            Ax = Ax - czm * wp.float64(x[k - 1, j, i])

    rf64 = wp.float64(b[k, j, i]) - Ax
    rh = rf64 * wp.float64(M_inv[k, j, i])
    r[k, j, i] = WP_FLOAT(rh)
    wp.atomic_add(rTr_buf, 0, rh * rh)
