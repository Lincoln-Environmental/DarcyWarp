# SPDX-License-Identifier: AGPL-3.0-only
"""Small FP64 face-array kernels for a future 3D execution path.

These kernels keep the six directional conductance arrays explicit and use
per-block partial reductions. They are intentionally independent of the
classic solver kernels and are not selected by the public 3D solver yet.
"""

from __future__ import annotations

import warp as wp


BLOCK_SIZE_3D = 256


@wp.kernel
def apply_A_faces_3d_f64_kernel(
    x: wp.array(dtype=wp.float64, ndim=3),
    tx_p: wp.array(dtype=wp.float64, ndim=3),
    tx_m: wp.array(dtype=wp.float64, ndim=3),
    ty_p: wp.array(dtype=wp.float64, ndim=3),
    ty_m: wp.array(dtype=wp.float64, ndim=3),
    tz_p: wp.array(dtype=wp.float64, ndim=3),
    tz_m: wp.array(dtype=wp.float64, ndim=3),
    storage: wp.array(dtype=wp.float64, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    out: wp.array(dtype=wp.float64, ndim=3),
    nx: int,
    ny: int,
    nz: int,
):
    t = wp.tid()
    n = nx * ny * nz
    if t >= n:
        return
    i = t % nx
    j = (t // nx) % ny
    k = t // (nx * ny)
    if active[k, j, i] == 0:
        out[k, j, i] = wp.float64(0.0)
        return
    if bc_mask[k, j, i] != 0:
        out[k, j, i] = x[k, j, i]
        return
    cxp = tx_p[k, j, i]
    cxm = tx_m[k, j, i]
    cyp = ty_p[k, j, i]
    cym = ty_m[k, j, i]
    czp = tz_p[k, j, i]
    czm = tz_m[k, j, i]
    if i + 1 >= nx or active[k, j, i + 1] == 0 or cxp < wp.float64(0.0):
        cxp = wp.float64(0.0)
    if i == 0 or active[k, j, i - 1] == 0 or cxm < wp.float64(0.0):
        cxm = wp.float64(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0 or cyp < wp.float64(0.0):
        cyp = wp.float64(0.0)
    if j == 0 or active[k, j - 1, i] == 0 or cym < wp.float64(0.0):
        cym = wp.float64(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0 or czp < wp.float64(0.0):
        czp = wp.float64(0.0)
    if k == 0 or active[k - 1, j, i] == 0 or czm < wp.float64(0.0):
        czm = wp.float64(0.0)
    sdiag = storage[k, j, i]
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)
    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    if diag < wp.float64(1.0e-12):
        out[k, j, i] = x[k, j, i]
        return
    value = diag * x[k, j, i]
    if cxp > wp.float64(0.0):
        value = value - cxp * x[k, j, i + 1]
    if cxm > wp.float64(0.0):
        value = value - cxm * x[k, j, i - 1]
    if cyp > wp.float64(0.0):
        value = value - cyp * x[k, j + 1, i]
    if cym > wp.float64(0.0):
        value = value - cym * x[k, j - 1, i]
    if czp > wp.float64(0.0):
        value = value - czp * x[k + 1, j, i]
    if czm > wp.float64(0.0):
        value = value - czm * x[k - 1, j, i]
    out[k, j, i] = value


@wp.kernel
def residual_partials_3d_f64_kernel(
    x: wp.array(dtype=wp.float64, ndim=3),
    rhs: wp.array(dtype=wp.float64, ndim=3),
    tx_p: wp.array(dtype=wp.float64, ndim=3),
    tx_m: wp.array(dtype=wp.float64, ndim=3),
    ty_p: wp.array(dtype=wp.float64, ndim=3),
    ty_m: wp.array(dtype=wp.float64, ndim=3),
    tz_p: wp.array(dtype=wp.float64, ndim=3),
    tz_m: wp.array(dtype=wp.float64, ndim=3),
    storage: wp.array(dtype=wp.float64, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    residual: wp.array(dtype=wp.float64, ndim=3),
    partials: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
    block_span: int,
):
    t = wp.tid()
    n = nx * ny * nz
    if t >= n:
        return
    i = t % nx
    j = (t // nx) % ny
    k = t // (nx * ny)
    block = t // block_span
    residual[k, j, i] = wp.float64(0.0)
    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        return
    cxp = tx_p[k, j, i]
    cxm = tx_m[k, j, i]
    cyp = ty_p[k, j, i]
    cym = ty_m[k, j, i]
    czp = tz_p[k, j, i]
    czm = tz_m[k, j, i]
    if i + 1 >= nx or active[k, j, i + 1] == 0 or cxp < wp.float64(0.0):
        cxp = wp.float64(0.0)
    if i == 0 or active[k, j, i - 1] == 0 or cxm < wp.float64(0.0):
        cxm = wp.float64(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0 or cyp < wp.float64(0.0):
        cyp = wp.float64(0.0)
    if j == 0 or active[k, j - 1, i] == 0 or cym < wp.float64(0.0):
        cym = wp.float64(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0 or czp < wp.float64(0.0):
        czp = wp.float64(0.0)
    if k == 0 or active[k - 1, j, i] == 0 or czm < wp.float64(0.0):
        czm = wp.float64(0.0)
    sdiag = storage[k, j, i]
    if sdiag < wp.float64(0.0):
        sdiag = wp.float64(0.0)
    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    value = x[k, j, i]
    if diag >= wp.float64(1.0e-12):
        value = diag * x[k, j, i]
        if cxp > wp.float64(0.0):
            value = value - cxp * x[k, j, i + 1]
        if cxm > wp.float64(0.0):
            value = value - cxm * x[k, j, i - 1]
        if cyp > wp.float64(0.0):
            value = value - cyp * x[k, j + 1, i]
        if cym > wp.float64(0.0):
            value = value - cym * x[k, j - 1, i]
        if czp > wp.float64(0.0):
            value = value - czp * x[k + 1, j, i]
        if czm > wp.float64(0.0):
            value = value - czm * x[k - 1, j, i]
    res = rhs[k, j, i] - value
    residual[k, j, i] = res
    wp.atomic_add(partials, block, res * res)


@wp.kernel
def combine_partials_3d_f64_kernel(
    partials: wp.array(dtype=wp.float64, ndim=1),
    out: wp.array(dtype=wp.float64, ndim=1),
    count: int,
):
    if wp.tid() == 0:
        total = wp.float64(0.0)
        for index in range(count):
            total = total + partials[index]
        out[0] = total


@wp.kernel
def face_jacobi_3d_f64_kernel(
    rhs: wp.array(dtype=wp.float64, ndim=3),
    x_in: wp.array(dtype=wp.float64, ndim=3),
    tx_p: wp.array(dtype=wp.float64, ndim=3),
    tx_m: wp.array(dtype=wp.float64, ndim=3),
    ty_p: wp.array(dtype=wp.float64, ndim=3),
    ty_m: wp.array(dtype=wp.float64, ndim=3),
    tz_p: wp.array(dtype=wp.float64, ndim=3),
    tz_m: wp.array(dtype=wp.float64, ndim=3),
    diag: wp.array(dtype=wp.float64, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    bc_values: wp.array(dtype=wp.float64, ndim=3),
    omega: float,
    nx: int,
    ny: int,
    nz: int,
    x_out: wp.array(dtype=wp.float64, ndim=3),
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return
    if active[k, j, i] == 0:
        x_out[k, j, i] = wp.float64(0.0)
        return
    if bc_mask[k, j, i] != 0:
        x_out[k, j, i] = bc_values[k, j, i]
        return
    value = diag[k, j, i] * x_in[k, j, i]
    if tx_p[k, j, i] > wp.float64(0.0):
        value = value - tx_p[k, j, i] * x_in[k, j, i + 1]
    if tx_m[k, j, i] > wp.float64(0.0):
        value = value - tx_m[k, j, i] * x_in[k, j, i - 1]
    if ty_p[k, j, i] > wp.float64(0.0):
        value = value - ty_p[k, j, i] * x_in[k, j + 1, i]
    if ty_m[k, j, i] > wp.float64(0.0):
        value = value - ty_m[k, j, i] * x_in[k, j - 1, i]
    if tz_p[k, j, i] > wp.float64(0.0):
        value = value - tz_p[k, j, i] * x_in[k + 1, j, i]
    if tz_m[k, j, i] > wp.float64(0.0):
        value = value - tz_m[k, j, i] * x_in[k - 1, j, i]
    x_out[k, j, i] = x_in[k, j, i] + wp.float64(omega) * (rhs[k, j, i] - value) / diag[k, j, i]


@wp.kernel
def dot_partials_3d_f64_kernel(
    a: wp.array(dtype=wp.float64, ndim=3),
    b: wp.array(dtype=wp.float64, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    partials: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
    block_span: int,
):
    t = wp.tid()
    n = nx * ny * nz
    if t >= n:
        return
    i = t % nx
    j = (t // nx) % ny
    k = t // (nx * ny)
    if active[k, j, i] != 0 and bc_mask[k, j, i] == 0:
        wp.atomic_add(partials, t // block_span, a[k, j, i] * b[k, j, i])


@wp.kernel
def apply_a_dot_partials_3d_f64_kernel(
    x: wp.array(dtype=wp.float64, ndim=3),
    tx_p: wp.array(dtype=wp.float64, ndim=3),
    tx_m: wp.array(dtype=wp.float64, ndim=3),
    ty_p: wp.array(dtype=wp.float64, ndim=3),
    ty_m: wp.array(dtype=wp.float64, ndim=3),
    tz_p: wp.array(dtype=wp.float64, ndim=3),
    tz_m: wp.array(dtype=wp.float64, ndim=3),
    diag: wp.array(dtype=wp.float64, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    partials: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
    block_span: int,
):
    t = wp.tid()
    n = nx * ny * nz
    if t >= n:
        return
    i = t % nx
    j = (t // nx) % ny
    k = t // (nx * ny)
    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        return
    value = diag[k, j, i] * x[k, j, i]
    if tx_p[k, j, i] > wp.float64(0.0):
        value = value - tx_p[k, j, i] * x[k, j, i + 1]
    if tx_m[k, j, i] > wp.float64(0.0):
        value = value - tx_m[k, j, i] * x[k, j, i - 1]
    if ty_p[k, j, i] > wp.float64(0.0):
        value = value - ty_p[k, j, i] * x[k, j + 1, i]
    if ty_m[k, j, i] > wp.float64(0.0):
        value = value - ty_m[k, j, i] * x[k, j - 1, i]
    if tz_p[k, j, i] > wp.float64(0.0):
        value = value - tz_p[k, j, i] * x[k + 1, j, i]
    if tz_m[k, j, i] > wp.float64(0.0):
        value = value - tz_m[k, j, i] * x[k - 1, j, i]
    wp.atomic_add(partials, t // block_span, x[k, j, i] * value)


@wp.kernel
def compute_safe_alpha_3d_kernel(
    numerator: wp.array(dtype=wp.float64, ndim=1),
    denominator: wp.array(dtype=wp.float64, ndim=1),
    alpha: wp.array(dtype=wp.float64, ndim=1),
):
    if wp.tid() == 0:
        value = wp.float64(0.0)
        if wp.abs(denominator[0]) > wp.float64(1.0e-30):
            value = numerator[0] / denominator[0]
        alpha[0] = value


@wp.kernel
def axpy_active_device_scalar_3d_kernel(
    destination: wp.array(dtype=wp.float64, ndim=3),
    source: wp.array(dtype=wp.float64, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    alpha: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k < nz and j < ny and i < nx and active[k, j, i] != 0 and bc_mask[k, j, i] == 0:
        destination[k, j, i] = destination[k, j, i] + alpha[0] * source[k, j, i]


__all__ = [
    "BLOCK_SIZE_3D",
    "apply_A_faces_3d_f64_kernel",
    "residual_partials_3d_f64_kernel",
    "combine_partials_3d_f64_kernel",
    "face_jacobi_3d_f64_kernel",
    "dot_partials_3d_f64_kernel",
    "apply_a_dot_partials_3d_f64_kernel",
    "compute_safe_alpha_3d_kernel",
    "axpy_active_device_scalar_3d_kernel",
]
