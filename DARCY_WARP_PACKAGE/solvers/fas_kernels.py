# SPDX-License-Identifier: AGPL-3.0-only
"""Quantity-aware transfers and nonlinear FAS work kernels."""

from __future__ import annotations

import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT


@wp.kernel
def fas_defect_kernel(
    physical_residual: wp.array(dtype=WP_FLOAT, ndim=2),
    physical_forcing: wp.array(dtype=WP_FLOAT, ndim=2),
    fas_forcing: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    defect: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or prescribed[j, i] != 0:
        defect[j, i] = WP_FLOAT(0.0)
    else:
        # Fphysical = N(h) - physical_forcing.
        defect[j, i] = WP_FLOAT(
            wp.float64(fas_forcing[j, i])
            - wp.float64(physical_forcing[j, i])
            - wp.float64(physical_residual[j, i])
        )


@wp.kernel
def fas_frozen_diagonal_kernel(
    transmissivity: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diagonal: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    ghb_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    diagonal: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or prescribed[j, i] != 0:
        diagonal[j, i] = WP_FLOAT(1.0)
        return
    tc = wp.float64(transmissivity[j, i])
    value = wp.float64(storage_diagonal[j, i])
    if i + 1 < nx and active[j, i + 1] != 0:
        tn = wp.float64(transmissivity[j, i + 1])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if i - 1 >= 0 and active[j, i - 1] != 0:
        tn = wp.float64(transmissivity[j, i - 1])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if j + 1 < ny and active[j + 1, i] != 0:
        tn = wp.float64(transmissivity[j + 1, i])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if j - 1 >= 0 and active[j - 1, i] != 0:
        tn = wp.float64(transmissivity[j - 1, i])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if ghb_mask[j, i] != 0:
        factor = wp.float64(ghb_factor[j, i])
        if factor > wp.float64(0.0) and not wp.isnan(factor):
            value = value + tc * factor
    if value < wp.float64(1.0e-12):
        value = wp.float64(1.0)
    diagonal[j, i] = WP_FLOAT(value)


@wp.kernel
def fas_jacobi_update_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    defect: wp.array(dtype=WP_FLOAT, ndim=2),
    diagonal: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    prescribed_values: wp.array(dtype=WP_FLOAT, ndim=2),
    damping: wp.float64,
    correction_limit: wp.float64,
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0:
        head[j, i] = WP_FLOAT(0.0)
    elif prescribed[j, i] != 0:
        head[j, i] = prescribed_values[j, i]
    else:
        correction = wp.float64(damping) * wp.float64(defect[j, i]) / wp.float64(diagonal[j, i])
        limit = wp.float64(correction_limit)
        if correction > limit:
            correction = limit
        if correction < -limit:
            correction = -limit
        head[j, i] = WP_FLOAT(wp.float64(head[j, i]) + correction)


@wp.kernel
def fas_restrict_head_kernel(
    fine_head: wp.array(dtype=WP_FLOAT, ndim=2),
    fine_active: wp.array(dtype=wp.int32, ndim=2),
    coarse_active: wp.array(dtype=wp.int32, ndim=2),
    coarse_prescribed: wp.array(dtype=wp.int32, ndim=2),
    coarse_values: wp.array(dtype=WP_FLOAT, ndim=2),
    coarse_head: wp.array(dtype=WP_FLOAT, ndim=2),
    nx_f: int,
    ny_f: int,
    nx_c: int,
    ny_c: int,
):
    j, i = wp.tid()
    if j >= ny_c or i >= nx_c:
        return
    if coarse_active[j, i] == 0:
        coarse_head[j, i] = WP_FLOAT(0.0)
        return
    if coarse_prescribed[j, i] != 0:
        coarse_head[j, i] = coarse_values[j, i]
        return
    total = wp.float64(0.0)
    count = wp.int32(0)
    for dj in range(2):
        fj = 2 * j + dj
        if fj < ny_f:
            for di in range(2):
                fi = 2 * i + di
                if fi < nx_f and fine_active[fj, fi] != 0:
                    total = total + wp.float64(fine_head[fj, fi])
                    count = count + wp.int32(1)
    coarse_head[j, i] = WP_FLOAT(total / wp.float64(count)) if count > wp.int32(0) else WP_FLOAT(0.0)


@wp.kernel
def fas_restrict_integrated_kernel(
    fine_quantity: wp.array(dtype=WP_FLOAT, ndim=2),
    fine_active: wp.array(dtype=wp.int32, ndim=2),
    fine_prescribed: wp.array(dtype=wp.int32, ndim=2),
    coarse_active: wp.array(dtype=wp.int32, ndim=2),
    coarse_prescribed: wp.array(dtype=wp.int32, ndim=2),
    coarse_quantity: wp.array(dtype=WP_FLOAT, ndim=2),
    nx_f: int,
    ny_f: int,
    nx_c: int,
    ny_c: int,
):
    j, i = wp.tid()
    if j >= ny_c or i >= nx_c:
        return
    if coarse_active[j, i] == 0 or coarse_prescribed[j, i] != 0:
        coarse_quantity[j, i] = WP_FLOAT(0.0)
        return
    total = wp.float64(0.0)
    for dj in range(2):
        fj = 2 * j + dj
        if fj < ny_f:
            for di in range(2):
                fi = 2 * i + di
                if fi < nx_f and fine_active[fj, fi] != 0 and fine_prescribed[fj, fi] == 0:
                    total = total + wp.float64(fine_quantity[fj, fi])
    coarse_quantity[j, i] = WP_FLOAT(total)


@wp.kernel
def fas_build_coarse_forcing_kernel(
    coarse_physical_residual: wp.array(dtype=WP_FLOAT, ndim=2),
    coarse_physical_forcing: wp.array(dtype=WP_FLOAT, ndim=2),
    restricted_defect: wp.array(dtype=WP_FLOAT, ndim=2),
    restricted_fine_forcing: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    coarse_forcing: wp.array(dtype=WP_FLOAT, ndim=2),
    tau: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or prescribed[j, i] != 0:
        coarse_forcing[j, i] = WP_FLOAT(0.0)
        tau[j, i] = WP_FLOAT(0.0)
    else:
        nonlinear_value = wp.float64(coarse_physical_residual[j, i]) + wp.float64(coarse_physical_forcing[j, i])
        forcing = nonlinear_value + wp.float64(restricted_defect[j, i])
        coarse_forcing[j, i] = WP_FLOAT(forcing)
        tau[j, i] = WP_FLOAT(forcing - wp.float64(restricted_fine_forcing[j, i]))


@wp.kernel
def fas_difference_kernel(
    value: wp.array(dtype=WP_FLOAT, ndim=2),
    reference: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    difference: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j < ny and i < nx:
        difference[j, i] = (
            WP_FLOAT(wp.float64(value[j, i]) - wp.float64(reference[j, i]))
            if active[j, i] != 0 and prescribed[j, i] == 0 else WP_FLOAT(0.0)
        )


@wp.kernel
def fas_prolong_correction_kernel(
    coarse: wp.array(dtype=WP_FLOAT, ndim=2),
    fine_active: wp.array(dtype=wp.int32, ndim=2),
    fine_prescribed: wp.array(dtype=wp.int32, ndim=2),
    fine: wp.array(dtype=WP_FLOAT, ndim=2),
    nx_f: int,
    ny_f: int,
    nx_c: int,
    ny_c: int,
):
    j, i = wp.tid()
    if j >= ny_f or i >= nx_f:
        return
    if fine_active[j, i] == 0 or fine_prescribed[j, i] != 0:
        fine[j, i] = WP_FLOAT(0.0)
        return
    x = (wp.float64(i) + wp.float64(0.5)) * wp.float64(0.5) - wp.float64(0.5)
    y = (wp.float64(j) + wp.float64(0.5)) * wp.float64(0.5) - wp.float64(0.5)
    i0 = wp.int32(wp.floor(x))
    j0 = wp.int32(wp.floor(y))
    tx = x - wp.float64(i0)
    ty = y - wp.float64(j0)
    if i0 < 0:
        i0 = 0
        tx = wp.float64(0.0)
    if j0 < 0:
        j0 = 0
        ty = wp.float64(0.0)
    i1 = i0 + 1
    j1 = j0 + 1
    if i1 >= nx_c:
        i1 = nx_c - 1
        tx = wp.float64(0.0)
    if j1 >= ny_c:
        j1 = ny_c - 1
        ty = wp.float64(0.0)
    c00 = wp.float64(coarse[j0, i0])
    c10 = wp.float64(coarse[j0, i1])
    c01 = wp.float64(coarse[j1, i0])
    c11 = wp.float64(coarse[j1, i1])
    fine[j, i] = WP_FLOAT(
        (wp.float64(1.0) - ty) * ((wp.float64(1.0) - tx) * c00 + tx * c10)
        + ty * ((wp.float64(1.0) - tx) * c01 + tx * c11)
    )


@wp.kernel
def fas_candidate_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    correction: wp.array(dtype=WP_FLOAT, ndim=2),
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
    elif prescribed[j, i] != 0:
        candidate[j, i] = prescribed_values[j, i]
    else:
        change = wp.float64(alpha) * wp.float64(correction[j, i])
        value = wp.float64(head[j, i]) + change
        candidate[j, i] = WP_FLOAT(value)
        if wp.isnan(value) or wp.isinf(value):
            wp.atomic_max(finite_flag, 0, wp.int32(1))
        wp.atomic_add(change_sq, 0, change * change)
        wp.atomic_max(change_max, 0, wp.abs(change))


@wp.kernel
def fas_norm_kernel(
    value: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    diagonal: wp.array(dtype=WP_FLOAT, ndim=2),
    use_diagonal: int,
    sum_sq: wp.array(dtype=wp.float64, ndim=1),
    max_abs: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j < ny and i < nx and active[j, i] != 0 and prescribed[j, i] == 0:
        v = wp.float64(value[j, i])
        if use_diagonal != 0:
            v = v / wp.float64(diagonal[j, i])
        wp.atomic_add(sum_sq, 0, v * v)
        wp.atomic_max(max_abs, 0, wp.abs(v))


@wp.kernel
def fas_copy_kernel(
    source: wp.array(dtype=WP_FLOAT, ndim=2),
    target: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    j, i = wp.tid()
    if j < ny and i < nx:
        target[j, i] = source[j, i]



@wp.kernel
def fas_defect_diagonal_kernel(
    physical_residual: wp.array(dtype=WP_FLOAT, ndim=2),
    physical_forcing: wp.array(dtype=WP_FLOAT, ndim=2),
    fas_forcing: wp.array(dtype=WP_FLOAT, ndim=2),
    transmissivity: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diagonal: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    ghb_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    defect: wp.array(dtype=WP_FLOAT, ndim=2),
    diagonal: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    # Fused fas_defect_kernel + fas_frozen_diagonal_kernel (verbatim bodies,
    # one grid pass): defect = forcing - physical_forcing - N(h), and the
    # frozen Picard diagonal used by the Jacobi update and head-equivalent
    # norms.
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or prescribed[j, i] != 0:
        defect[j, i] = WP_FLOAT(0.0)
        diagonal[j, i] = WP_FLOAT(1.0)
        return
    defect[j, i] = WP_FLOAT(
        wp.float64(fas_forcing[j, i])
        - wp.float64(physical_forcing[j, i])
        - wp.float64(physical_residual[j, i])
    )
    tc = wp.float64(transmissivity[j, i])
    value = wp.float64(storage_diagonal[j, i])
    if i + 1 < nx and active[j, i + 1] != 0:
        tn = wp.float64(transmissivity[j, i + 1])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if i - 1 >= 0 and active[j, i - 1] != 0:
        tn = wp.float64(transmissivity[j, i - 1])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if j + 1 < ny and active[j + 1, i] != 0:
        tn = wp.float64(transmissivity[j + 1, i])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if j - 1 >= 0 and active[j - 1, i] != 0:
        tn = wp.float64(transmissivity[j - 1, i])
        if tc > wp.float64(0.0) and tn > wp.float64(0.0):
            value = value + wp.float64(2.0) * tc * tn / (tc + tn + wp.float64(1.0e-12))
    if ghb_mask[j, i] != 0:
        factor = wp.float64(ghb_factor[j, i])
        if factor > wp.float64(0.0) and not wp.isnan(factor):
            value = value + tc * factor
    if value < wp.float64(1.0e-12):
        value = wp.float64(1.0)
    diagonal[j, i] = WP_FLOAT(value)


@wp.kernel
def fas_norm_pair_kernel(
    value_a: wp.array(dtype=WP_FLOAT, ndim=2),
    value_b: wp.array(dtype=WP_FLOAT, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    prescribed: wp.array(dtype=wp.int32, ndim=2),
    norms: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    # Paired plain (non head-equivalent) norms in one launch:
    # norms = [sum_sq_a, max_abs_a, sum_sq_b, max_abs_b].
    j, i = wp.tid()
    if j < ny and i < nx and active[j, i] != 0 and prescribed[j, i] == 0:
        va = wp.float64(value_a[j, i])
        vb = wp.float64(value_b[j, i])
        wp.atomic_add(norms, 0, va * va)
        wp.atomic_max(norms, 1, wp.abs(va))
        wp.atomic_add(norms, 2, vb * vb)
        wp.atomic_max(norms, 3, wp.abs(vb))
