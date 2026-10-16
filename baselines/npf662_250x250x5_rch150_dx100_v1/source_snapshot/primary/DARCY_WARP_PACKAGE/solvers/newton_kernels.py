# SPDX-License-Identifier: AGPL-3.0-only
"""Small allocation-free vector kernels for 2D Newton--FGMRES."""

from __future__ import annotations

import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT


@wp.kernel
def masked_copy_kernel(
    src: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    dst: wp.array(dtype=WP_FLOAT, ndim=2),
    scale: wp.float64,
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or prescribed[j, i] != 0:
        dst[j, i] = WP_FLOAT(0.0)
    else:
        dst[j, i] = WP_FLOAT(wp.float64(src[j, i]) * wp.float64(scale))


@wp.kernel
def masked_axpy_kernel(
    y: wp.array(dtype=WP_FLOAT, ndim=2),
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    alpha: wp.float64,
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or prescribed[j, i] != 0:
        y[j, i] = WP_FLOAT(0.0)
    else:
        y[j, i] = WP_FLOAT(wp.float64(y[j, i]) + wp.float64(alpha) * wp.float64(x[j, i]))


@wp.kernel
def masked_candidate_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    delta: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    prescribed_values: wp.array(dtype=WP_FLOAT, ndim=2),
    alpha: wp.float64,
    candidate: wp.array(dtype=WP_FLOAT, ndim=2),
    change_sq: wp.array(dtype=wp.float64, ndim=1),
    change_max: wp.array(dtype=wp.float64, ndim=1),
    finite_flag: wp.array(dtype=wp.int32, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0:
        candidate[j, i] = WP_FLOAT(0.0)
        return
    if prescribed[j, i] != 0:
        candidate[j, i] = prescribed_values[j, i]
        return
    change = wp.float64(alpha) * wp.float64(delta[j, i])
    value = wp.float64(head[j, i]) + change
    candidate[j, i] = WP_FLOAT(value)
    if wp.isnan(value) or wp.isinf(value):
        wp.atomic_max(finite_flag, 0, wp.int32(1))
    wp.atomic_add(change_sq, 0, change * change)
    wp.atomic_max(change_max, 0, wp.abs(change))


@wp.kernel
def masked_dot_kernel(
    x: wp.array(dtype=WP_FLOAT, ndim=2),
    y: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    out: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j < ny and i < nx and active[j, i] != 0 and prescribed[j, i] == 0:
        wp.atomic_add(out, 0, wp.float64(x[j, i]) * wp.float64(y[j, i]))


@wp.kernel
def head_equivalent_norm_kernel(
    residual: wp.array(dtype=WP_FLOAT, ndim=2),
    diagonal_inverse: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    out: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j < ny and i < nx and active[j, i] != 0 and prescribed[j, i] == 0:
        value = wp.float64(residual[j, i]) * wp.float64(diagonal_inverse[j, i])
        wp.atomic_add(out, 0, value * value)

