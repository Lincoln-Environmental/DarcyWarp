"""FP32 NPF preconditioner kernels; FP64 physical residual and Krylov stay authoritative.

Mirrors the Cartesian restriction/prolongation and face K-cycle kernels.
Energy reductions accumulate FP64, as in the production 2D mixed solver.
"""


import warp as wp


@wp.kernel
def cast_coefficients(source: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float32),
    invalid: wp.array(dtype=wp.int32)):
    i = wp.tid()
    value = wp.float32(source[i])
    out[i] = value
    if not wp.isfinite(value) or (value == wp.float32(0.) and source[i] != wp.float64(0.)):
        wp.atomic_max(invalid, 0, 1)


@wp.kernel
def scaled_rhs(vector: wp.array(dtype=wp.float64), diagonal: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.float32(0.)
    if free[i] != 0:
        out[i] = wp.float32(vector[i]*diagonal[i])


@wp.kernel
def cast_correction(source: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = wp.float64(source[i])


@wp.kernel
def smooth_factored_lines(xp: wp.array3d(dtype=wp.float32), xm: wp.array3d(dtype=wp.float32),
    yp: wp.array3d(dtype=wp.float32), ym: wp.array3d(dtype=wp.float32),
    diagonal: wp.array3d(dtype=wp.float32), active: wp.array3d(dtype=wp.int32),
    fixed: wp.array3d(dtype=wp.int32), rhs: wp.array3d(dtype=wp.float32),
    old: wp.array3d(dtype=wp.float32), inverse: wp.array3d(dtype=wp.float32),
    upper: wp.array3d(dtype=wp.float32), lower: wp.array3d(dtype=wp.float32),
    forward: wp.array3d(dtype=wp.float32), out: wp.array3d(dtype=wp.float32), omega: wp.float32):
    j, i = wp.tid()
    previous = wp.float32(0.)
    for k in range(old.shape[0]):
        value = wp.float32(0.)
        if active[k,j,i] != 0 and fixed[k,j,i] == 0:
            value = rhs[k,j,i]
            if i+1 < old.shape[2]: value += xp[k,j,i]*old[k,j,i+1]
            if i > 0: value += xm[k,j,i]*old[k,j,i-1]
            if j+1 < old.shape[1]: value += yp[k,j,i]*old[k,j+1,i]
            if j > 0: value += ym[k,j,i]*old[k,j-1,i]
            value = value*inverse[k,j,i]+lower[k,j,i]*previous
            if diagonal[k,j,i] == wp.float32(0.): value = old[k,j,i]
        forward[k,j,i] = value
        previous = value
    previous = wp.float32(0.)
    for reverse in range(old.shape[0]):
        k = old.shape[0]-1-reverse
        value = forward[k,j,i]-upper[k,j,i]*previous
        previous = value
        out[k,j,i] = wp.float32(0.)
        if active[k,j,i] != 0 and fixed[k,j,i] == 0:
            out[k,j,i] = old[k,j,i]+omega*(value-old[k,j,i])


@wp.kernel
def residual_partials_3d_f64_kernel(
    x: wp.array(dtype=wp.float32, ndim=3),
    rhs: wp.array(dtype=wp.float32, ndim=3),
    tx_p: wp.array(dtype=wp.float32, ndim=3),
    tx_m: wp.array(dtype=wp.float32, ndim=3),
    ty_p: wp.array(dtype=wp.float32, ndim=3),
    ty_m: wp.array(dtype=wp.float32, ndim=3),
    tz_p: wp.array(dtype=wp.float32, ndim=3),
    tz_m: wp.array(dtype=wp.float32, ndim=3),
    storage: wp.array(dtype=wp.float32, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    residual: wp.array(dtype=wp.float32, ndim=3),
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
    residual[k, j, i] = wp.float32(0.0)
    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        return
    cxp = tx_p[k, j, i]
    cxm = tx_m[k, j, i]
    cyp = ty_p[k, j, i]
    cym = ty_m[k, j, i]
    czp = tz_p[k, j, i]
    czm = tz_m[k, j, i]
    if i + 1 >= nx or active[k, j, i + 1] == 0 or cxp < wp.float32(0.0):
        cxp = wp.float32(0.0)
    if i == 0 or active[k, j, i - 1] == 0 or cxm < wp.float32(0.0):
        cxm = wp.float32(0.0)
    if j + 1 >= ny or active[k, j + 1, i] == 0 or cyp < wp.float32(0.0):
        cyp = wp.float32(0.0)
    if j == 0 or active[k, j - 1, i] == 0 or cym < wp.float32(0.0):
        cym = wp.float32(0.0)
    if k + 1 >= nz or active[k + 1, j, i] == 0 or czp < wp.float32(0.0):
        czp = wp.float32(0.0)
    if k == 0 or active[k - 1, j, i] == 0 or czm < wp.float32(0.0):
        czm = wp.float32(0.0)
    sdiag = storage[k, j, i]
    if sdiag < wp.float32(0.0):
        sdiag = wp.float32(0.0)
    diag = cxp + cxm + cyp + cym + czp + czm + sdiag
    value = wp.float32(0.0)
    if diag > wp.float32(0.0):
        value = sdiag * x[k, j, i]
        if cxp > wp.float32(0.0):
            value = value + cxp * (x[k, j, i] - x[k, j, i + 1])
        if cxm > wp.float32(0.0):
            value = value + cxm * (x[k, j, i] - x[k, j, i - 1])
        if cyp > wp.float32(0.0):
            value = value + cyp * (x[k, j, i] - x[k, j + 1, i])
        if cym > wp.float32(0.0):
            value = value + cym * (x[k, j, i] - x[k, j - 1, i])
        if czp > wp.float32(0.0):
            value = value + czp * (x[k, j, i] - x[k + 1, j, i])
        if czm > wp.float32(0.0):
            value = value + czm * (x[k, j, i] - x[k - 1, j, i])
    res = rhs[k, j, i] - value
    residual[k, j, i] = res
    wp.atomic_add(partials, block, wp.float64(res) * wp.float64(res))


@wp.kernel
def dot_partials_3d_f64_kernel(
    a: wp.array(dtype=wp.float32, ndim=3),
    b: wp.array(dtype=wp.float32, ndim=3),
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
        wp.atomic_add(partials, t // block_span, wp.float64(a[k, j, i]) * wp.float64(b[k, j, i]))


@wp.kernel
def apply_a_dot_partials_3d_f64_kernel(
    x: wp.array(dtype=wp.float32, ndim=3),
    tx_p: wp.array(dtype=wp.float32, ndim=3),
    tx_m: wp.array(dtype=wp.float32, ndim=3),
    ty_p: wp.array(dtype=wp.float32, ndim=3),
    ty_m: wp.array(dtype=wp.float32, ndim=3),
    tz_p: wp.array(dtype=wp.float32, ndim=3),
    tz_m: wp.array(dtype=wp.float32, ndim=3),
    diag: wp.array(dtype=wp.float32, ndim=3),
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
    if tx_p[k, j, i] > wp.float32(0.0):
        value = value - tx_p[k, j, i] * x[k, j, i + 1]
    if tx_m[k, j, i] > wp.float32(0.0):
        value = value - tx_m[k, j, i] * x[k, j, i - 1]
    if ty_p[k, j, i] > wp.float32(0.0):
        value = value - ty_p[k, j, i] * x[k, j + 1, i]
    if ty_m[k, j, i] > wp.float32(0.0):
        value = value - ty_m[k, j, i] * x[k, j - 1, i]
    if tz_p[k, j, i] > wp.float32(0.0):
        value = value - tz_p[k, j, i] * x[k + 1, j, i]
    if tz_m[k, j, i] > wp.float32(0.0):
        value = value - tz_m[k, j, i] * x[k - 1, j, i]
    wp.atomic_add(partials, t // block_span, wp.float64(x[k, j, i]) * wp.float64(value))


@wp.kernel
def axpy_active_device_scalar_3d_kernel(
    destination: wp.array(dtype=wp.float32, ndim=3),
    source: wp.array(dtype=wp.float32, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    alpha: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k < nz and j < ny and i < nx and active[k, j, i] != 0 and bc_mask[k, j, i] == 0:
        destination[k, j, i] = destination[k, j, i] + wp.float32(alpha[0]) * source[k, j, i]


@wp.kernel
def mask_correction_3d_kernel(
    correction: wp.array(dtype=wp.float32, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
):
    k, j, i = wp.tid()
    if active[k, j, i] == 0 or bc_mask[k, j, i] != 0:
        correction[k, j, i] = wp.float32(0.0)


@wp.kernel
def restrict_blockavg_axes_3d_kernel(
    r_f: wp.array(dtype=wp.float32, ndim=3),
    active_f: wp.array(dtype=wp.int32, ndim=3),
    bc_mask_f: wp.array(dtype=wp.int32, ndim=3),
    b_c: wp.array(dtype=wp.float32, ndim=3),
    nx_f: int,
    ny_f: int,
    nz_f: int,
    nx_c: int,
    ny_c: int,
    nz_c: int,
    coarsen_y: int,
    coarsen_x: int,
):
    """Residual restriction with independent horizontal coarsening factors."""
    kc, jc, ic = wp.tid()
    if kc >= nz_c or jc >= ny_c or ic >= nx_c:
        return

    j0 = coarsen_y * jc
    i0 = coarsen_x * ic
    s = wp.float32(0.0)
    n = wp.float32(0.0)
    for dj in range(2):
        if coarsen_y == 1 and dj != 0:
            continue
        jf = j0 + dj
        if jf >= ny_f:
            continue
        for di in range(2):
            if coarsen_x == 1 and di != 0:
                continue
            i_f = i0 + di
            if i_f >= nx_f:
                continue
            if active_f[kc, jf, i_f] != 0 and bc_mask_f[kc, jf, i_f] == 0:
                s = s + r_f[kc, jf, i_f]
                n = n + wp.float32(1.0)
    if n > wp.float32(0.0):
        b_c[kc, jc, ic] = s / n
    else:
        b_c[kc, jc, ic] = wp.float32(0.0)


@wp.kernel
def prolong_bilinear_axes_3d_kernel(
    x_c: wp.array(dtype=wp.float32, ndim=3),
    e_f: wp.array(dtype=wp.float32, ndim=3),
    nx_f: int,
    ny_f: int,
    nz_f: int,
    nx_c: int,
    ny_c: int,
    nz_c: int,
    coarsen_y: int,
    coarsen_x: int,
):
    """Bilinear horizontal prolongation with independent axis factors."""
    k, j, i = wp.tid()
    if k >= nz_f or j >= ny_f or i >= nx_f:
        return

    jc = j // coarsen_y
    ic = i // coarsen_x
    fy = wp.float32(0.0)
    fx = wp.float32(0.0)
    if coarsen_y == 2 and (j & 1) == 1:
        fy = wp.float32(0.5)
    if coarsen_x == 2 and (i & 1) == 1:
        fx = wp.float32(0.5)
    jc1 = jc + 1
    ic1 = ic + 1
    if jc1 >= ny_c:
        jc1 = ny_c - 1
    if ic1 >= nx_c:
        ic1 = nx_c - 1

    c00 = x_c[k, jc, ic]
    c01 = x_c[k, jc, ic1]
    c10 = x_c[k, jc1, ic]
    c11 = x_c[k, jc1, ic1]
    e_f[k, j, i] = (
        c00 * (wp.float32(1.0) - fx) * (wp.float32(1.0) - fy)
        + c01 * fx * (wp.float32(1.0) - fy)
        + c10 * (wp.float32(1.0) - fx) * fy
        + c11 * fx * fy
    )


@wp.kernel
def add_correction_3d_kernel(
    x_f: wp.array(dtype=wp.float32, ndim=3),
    e_f: wp.array(dtype=wp.float32, ndim=3),
    active: wp.array(dtype=wp.int32, ndim=3),
    bc_mask: wp.array(dtype=wp.int32, ndim=3),
    bc_values: wp.array(dtype=wp.float32, ndim=3),
    nx: int,
    ny: int,
    nz: int,
):
    k, j, i = wp.tid()
    if k >= nz or j >= ny or i >= nx:
        return
    if active[k, j, i] == 0:
        x_f[k, j, i] = wp.float32(0.0)
        return
    if bc_mask[k, j, i] != 0:
        x_f[k, j, i] = bc_values[k, j, i]
        return
    x_f[k, j, i] = x_f[k, j, i] + e_f[k, j, i]
