"""FP64 device-side terrain nonlinear-update kernels.

Mirrors the production 2D fast unconfined contract for the 3D variable-
geometry terrain operator: the Picard outer loop evaluates saturation,
transformed face coefficients, secant storage and the linearised RHS on
the device from persistent buffers, so no full-head download and no host
coefficient packing is needed inside the nonlinear iteration. The host
CPU reference operator remains the independent scientific authority;
these kernels implement the SAME flux equations and are validated
coefficient-by-coefficient against pack_terrain_coefficients.

Static geometry (interface/centre/face slopes, thickness, masks) is
uploaded once per geometry revision; head-dependent constitutive fields
are refreshed in place every outer iteration. Contact coefficients
(conductance, metric couplings, prescribed-head offset) are packed
host-side once per contact set: they change only through
update_terrain_contacts, and the discharge-only gate is evaluated
on-device inside the divergence kernel from the current head.
"""
import warp as wp


BLOCK_SIZE = 256


@wp.kernel
def terrain_saturation_kernel(head: wp.array(dtype=wp.float64),
    bottoms: wp.array(dtype=wp.float64), thickness: wp.array(dtype=wp.float64),
    min_sat: wp.float64, true_sat: wp.array(dtype=wp.float64),
    fraction: wp.array(dtype=wp.float64)):
    c = wp.tid()
    local = wp.min(min_sat, thickness[c])
    true_sat[c] = wp.clamp(head[c]-bottoms[c], wp.float64(0.0), thickness[c])
    fraction[c] = wp.clamp(head[c]-bottoms[c], local, thickness[c])/thickness[c]


@wp.kernel
def terrain_face_coefficients_kernel(fraction: wp.array(dtype=wp.float64),
    kx: wp.array(dtype=wp.float64), ky: wp.array(dtype=wp.float64),
    kz: wp.array(dtype=wp.float64), thickness: wp.array(dtype=wp.float64),
    zx_interfaces: wp.array(dtype=wp.float64), zy_interfaces: wp.array(dtype=wp.float64),
    zx_centre: wp.array(dtype=wp.float64), zy_centre: wp.array(dtype=wp.float64),
    zxf: wp.array(dtype=wp.float64), zyf: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32), coefficients: wp.array2d(dtype=wp.float64),
    dx: wp.float64, dy: wp.float64, nx: int, ny: int, nz: int):
    """Per-cell packed face coefficients: the same equations as the host
    reference operator (harmonic half-cell transmissivities on unit-s map
    areas; interface slopes with harmonic resistance weights on vertical
    faces; metric-defect terms in columns 11/12)."""
    c = wp.tid()
    ncol = nx*ny
    i = c % nx
    j = (c // nx) % ny
    k = c // ncol
    zero = wp.float64(0.0)
    for entry in range(13):
        coefficients[c, entry] = zero
    if active[c] == 0:
        return
    half = wp.float64(0.5)
    one = wp.float64(1.0)
    fx = kx[c]*fraction[c]
    fy = ky[c]*fraction[c]
    tz = thickness[c]
    tx = fx*tz
    ty = fy*tz
    if i+1 < nx and active[c+1] != 0:
        cn = c+1
        txr = kx[cn]*fraction[cn]*thickness[cn]
        inv = one/((half*dx)/tx + (half*dx)/txr)
        coefficients[c, 0] = inv*dy
        wl = (one/tx)/(one/tx + one/txr)
        slope = zxf[c]
        coefficients[c, 1] = wl*tx*slope*dy
        coefficients[c, 2] = (one-wl)*txr*slope*dy
    if j+1 < ny and active[c+nx] != 0:
        cn = c+nx
        tyr = ky[cn]*fraction[cn]*thickness[cn]
        inv = one/((half*dy)/ty + (half*dy)/tyr)
        coefficients[c, 3] = inv*dx
        wl = (one/ty)/(one/ty + one/tyr)
        slope = zyf[c]
        coefficients[c, 4] = wl*ty*slope*dx
        coefficients[c, 5] = (one-wl)*tyr*slope*dx
    if k+1 < nz and active[c+ncol] != 0:
        cn = c+ncol
        ifx = (k+1)*ncol + (c % ncol)
        zxif = zx_interfaces[ifx]
        zyif = zy_interfaces[ifx]
        fxn = kx[cn]*fraction[cn]
        fyn = ky[cn]*fraction[cn]
        azz_t = fx*zxif*zxif + fy*zyif*zyif + kz[c]
        azz_b = fxn*zxif*zxif + fyn*zyif*zyif + kz[cn]
        tb = thickness[cn]
        rt = half*tz/azz_t
        rb = half*tb/azz_b
        area = dx*dy
        coefficients[c, 6] = area/(rt+rb)
        wl = rt/(rt+rb)
        wr = one-wl
        coefficients[c, 7] = wl*fx*zxif*area
        coefficients[c, 8] = wr*fxn*zxif*area
        coefficients[c, 9] = wl*fy*zyif*area
        coefficients[c, 10] = wr*fyn*zyif*area
        coefficients[c, 11] = wl*(fx*zxif*(zx_centre[c]-zxif)
            + fy*zyif*(zy_centre[c]-zyif))*area
        coefficients[c, 12] = wr*(fxn*zxif*(zx_centre[cn]-zxif)
            + fyn*zyif*(zy_centre[cn]-zyif))*area


@wp.kernel
def terrain_secant_storage_kernel(head: wp.array(dtype=wp.float64),
    previous: wp.array(dtype=wp.float64), bottoms: wp.array(dtype=wp.float64),
    thickness: wp.array(dtype=wp.float64), sy: wp.array(dtype=wp.float64),
    ss: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
    dx: wp.float64, dy: wp.float64, dt: wp.float64,
    storage_diag: wp.array(dtype=wp.float64), storage_change: wp.array(dtype=wp.float64)):
    """Volume-conservative secant Sy/Ss storage diagonal and change."""
    c = wp.tid()
    zero = wp.float64(0.0)
    storage_diag[c] = zero
    storage_change[c] = zero
    if free[c] == 0:
        return
    wet = wp.clamp(head[c]-bottoms[c], wp.float64(0.0), thickness[c])
    old_wet = wp.clamp(previous[c]-bottoms[c], wp.float64(0.0), thickness[c])
    difference = wet-old_wet
    excess = wp.max(head[c]-(bottoms[c]+thickness[c]), wp.float64(0.0))
    old_excess = wp.max(previous[c]-(bottoms[c]+thickness[c]), wp.float64(0.0))
    half = wp.float64(0.5)
    increment = sy[c]*difference + ss[c]*(half*difference*(wet+old_wet)
        + thickness[c]*(excess-old_excess))
    above = head[c] > bottoms[c]
    below_top = head[c] < bottoms[c]+thickness[c]
    phreatic = wp.where(above and below_top, wp.float64(1.0), wp.float64(0.0))
    tangent = sy[c]*phreatic + ss[c]*wet
    scaled = increment*dx*dy/dt
    diag = wp.where(head[c] != previous[c], increment/(head[c]-previous[c]), tangent)*dx*dy/dt
    storage_diag[c] = diag
    storage_change[c] = scaled


@wp.kernel
def terrain_rhs_kernel(rhs: wp.array(dtype=wp.float64), source: wp.array(dtype=wp.float64),
    storage_diag: wp.array(dtype=wp.float64), previous: wp.array(dtype=wp.float64),
    datum: wp.float64, fixed: wp.array(dtype=wp.int32), bc_values: wp.array(dtype=wp.float64),
    contact_rhs: wp.array(dtype=wp.float64), active: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float64)):
    """Row-scaled rhs = (source + storage_diag*(previous-datum)
    + contact c*(hb-datum))/diagonal, with prescribed-head rows carrying
    their datum-relative value and inactive rows zero."""
    c = wp.tid()
    zero = wp.float64(0.0)
    value = source[c]
    if active[c] != 0:
        value = value + storage_diag[c]*(previous[c]-datum) + contact_rhs[c]
    if fixed[c] != 0:
        value = bc_values[c]-datum
    if active[c] == 0:
        value = zero
    rhs[c] = value/scale[c]


@wp.kernel
def terrain_relative_head_kernel(head: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64), datum: wp.float64,
    active: wp.array(dtype=wp.int32)):
    c = wp.tid()
    if active[c] != 0:
        out[c] = head[c]-datum
    else:
        out[c] = wp.float64(0.0)


@wp.kernel
def terrain_extract_principal_kernel(coefficients: wp.array2d(dtype=wp.float64),
    active: wp.array(dtype=wp.int32), storage_diag: wp.array(dtype=wp.float64),
    boundary_col0: wp.array(dtype=wp.float64),
    tx_p: wp.array(dtype=wp.float64), tx_m: wp.array(dtype=wp.float64),
    ty_p: wp.array(dtype=wp.float64), ty_m: wp.array(dtype=wp.float64),
    tz_p: wp.array(dtype=wp.float64), tz_m: wp.array(dtype=wp.float64),
    diag: wp.array(dtype=wp.float64), principal_storage: wp.array(dtype=wp.float64),
    nx: int, ny: int, nz: int):
    """Extract the seven-point principal part (level-0 faces, diagonal and
    principal storage) from the packed conservative coefficients."""
    c = wp.tid()
    ncol = nx*ny
    i = c % nx
    j = (c // nx) % ny
    k = c // ncol
    txp = coefficients[c, 0]
    typ = coefficients[c, 3]
    tzp = coefficients[c, 6]
    txm = wp.float64(0.0)
    tym = wp.float64(0.0)
    tzm = wp.float64(0.0)
    if i > 0:
        txm = coefficients[c-1, 0]
    if j > 0:
        tym = coefficients[c-nx, 3]
    if k > 0:
        tzm = coefficients[c-ncol, 6]
    tx_p[c] = txp
    tx_m[c] = txm
    ty_p[c] = typ
    ty_m[c] = tym
    tz_p[c] = tzp
    tz_m[c] = tzm
    principal = storage_diag[c] + boundary_col0[c]
    principal_storage[c] = principal
    if active[c] == 0:
        diag[c] = wp.float64(0.0)
        return
    diag[c] = txp + txm + typ + tym + tzp + tzm + principal


@wp.kernel
def terrain_relaxed_update_kernel(head: wp.array(dtype=wp.float64),
    delta: wp.array(dtype=wp.float64), omega: wp.float64, max_change: wp.float64,
    fixed: wp.array(dtype=wp.int32), bc_values: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32)):
    """head <- head + omega*clip(delta), prescribed heads restored exactly
    and inactive cells zero."""
    c = wp.tid()
    zero = wp.float64(0.0)
    if active[c] == 0:
        head[c] = zero
        return
    if fixed[c] != 0:
        head[c] = bc_values[c]
        return
    head[c] = head[c] + omega*wp.clamp(delta[c], -max_change, max_change)


@wp.kernel
def terrain_copy_kernel(source: wp.array(dtype=wp.float64),
    destination: wp.array(dtype=wp.float64)):
    c = wp.tid()
    destination[c] = source[c]


@wp.kernel
def terrain_dual_check_kernel(head: wp.array(dtype=wp.float64),
    candidate: wp.array(dtype=wp.float64), residual: wp.array(dtype=wp.float64),
    rhs: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.int32), partials: wp.array2d(dtype=wp.float64),
    block_span: int):
    """Block partials over free cells: [0] dh^2, [1] dh_max, [2] flow r^2,
    [3] head-equivalent r^2, [4] rhs^2, [5] nonfinite flag. residual must
    already hold rhs - A*head for the current operator."""
    t = wp.tid()
    block = t//block_span
    dh = candidate[t]-head[t]
    r = residual[t]
    s = scale[t]
    is_free = free[t] != 0
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    dh2 = wp.where(is_free, dh*dh, zero)
    dhmax = wp.where(is_free and wp.abs(dh) > partials[block, 1], wp.abs(dh), partials[block, 1])
    r2 = wp.where(is_free, r*r, zero)
    heq = wp.where(is_free and s > 0.0, (r/s)*(r/s), zero)
    rhs2 = wp.where(is_free, rhs[t]*rhs[t], zero)
    bad = wp.where(is_free and (wp.abs(candidate[t]) > 1e300 or r != r), one, zero)
    wp.atomic_add(partials, block, 0, dh2)
    wp.atomic_max(partials, block, 1, dhmax)
    wp.atomic_add(partials, block, 2, r2)
    wp.atomic_add(partials, block, 3, heq)
    wp.atomic_add(partials, block, 4, rhs2)
    wp.atomic_add(partials, block, 5, bad)


@wp.kernel
def terrain_combine_partials_kernel(partials: wp.array2d(dtype=wp.float64),
    out: wp.array(dtype=wp.float64), blocks: int):
    if wp.tid() == 0:
        sums = wp.float64(0.0)
        maxima = wp.float64(0.0)
        for index in range(blocks):
            sums = sums + partials[index, 0] + partials[index, 2] \
                + partials[index, 3] + partials[index, 4] + partials[index, 5]
            maxima = wp.max(maxima, partials[index, 1])
        out[0] = sums
        out[1] = maxima
