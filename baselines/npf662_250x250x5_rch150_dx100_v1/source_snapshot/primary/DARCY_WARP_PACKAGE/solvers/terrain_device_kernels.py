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
are refreshed in place every outer iteration. Contact geometry is packed host-side; head-dependent conductance and
metric couplings are refreshed on device at each nonlinear state before
independent per-subface discharge-only gating and aggregation.
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


@wp.func
def terrain_harmonic_resistance(left: wp.float64, right: wp.float64,
    dl: wp.float64, dr: wp.float64):
    """Scaled series conductance and weights, matching the CPU reference."""
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    tl = left/dl
    tr = right/dr
    small = wp.min(tl, tr)
    large = wp.max(tl, tr)
    if large > zero:
        conductance = small/(one + small/large)
        scaled_l = tl/large
        scaled_r = tr/large
        denom = scaled_l + scaled_r
        return conductance, scaled_r/denom, scaled_l/denom
    return zero, zero, zero


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
    fx = kx[c]*fraction[c]
    fy = ky[c]*fraction[c]
    tz = thickness[c]
    tx = fx*tz
    ty = fy*tz
    if i+1 < nx and active[c+1] != 0:
        cn = c+1
        txr = kx[cn]*fraction[cn]*thickness[cn]
        inv, wl, wr = terrain_harmonic_resistance(left=tx, right=txr, dl=half*dx, dr=half*dx)
        coefficients[c, 0] = inv*dy
        slope = zxf[c]
        coefficients[c, 1] = wl*tx*slope*dy
        coefficients[c, 2] = wr*txr*slope*dy
    if j+1 < ny and active[c+nx] != 0:
        cn = c+nx
        tyr = ky[cn]*fraction[cn]*thickness[cn]
        inv, wl, wr = terrain_harmonic_resistance(left=ty, right=tyr, dl=half*dy, dr=half*dy)
        coefficients[c, 3] = inv*dx
        slope = zyf[c]
        coefficients[c, 4] = wl*ty*slope*dx
        coefficients[c, 5] = wr*tyr*slope*dx
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
        inv, wl, wr = terrain_harmonic_resistance(left=azz_t, right=azz_b,
            dl=half*tz, dr=half*tb)
        area = dx*dy
        coefficients[c, 6] = inv*area
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
    contact_rhs: wp.array(dtype=wp.float64), boundary: wp.array2d(dtype=wp.float64),
    active: wp.array(dtype=wp.int32), scale: wp.array(dtype=wp.float64)):
    """Row-scaled rhs = (source + storage_diag*(previous-datum)
    + open contact c*(hb-datum))/diagonal, with prescribed-head rows
    carrying their datum-relative value and inactive rows zero. The
    contact offset buffer is aggregated per outer iteration over the
    gate-open subfaces only (boundary arg kept for layout compatibility)."""
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
    active: wp.array(dtype=wp.int32), fixed: wp.array(dtype=wp.int32),
    storage_diag: wp.array(dtype=wp.float64),
    boundary_col0: wp.array(dtype=wp.float64),
    tx_p: wp.array(dtype=wp.float64, ndim=3), tx_m: wp.array(dtype=wp.float64, ndim=3),
    ty_p: wp.array(dtype=wp.float64, ndim=3), ty_m: wp.array(dtype=wp.float64, ndim=3),
    tz_p: wp.array(dtype=wp.float64, ndim=3), tz_m: wp.array(dtype=wp.float64, ndim=3),
    diag: wp.array(dtype=wp.float64, ndim=3),
    principal_storage: wp.array(dtype=wp.float64, ndim=3),
    scale_flat: wp.array(dtype=wp.float64),
    nx: int, ny: int, nz: int):
    """Extract the seven-point principal part (level-0 faces, 3-D diagonal,
    principal storage) plus the flat row scale from the packed coefficients.
    Fixed and inactive rows carry unit scale, matching the host assembly."""
    c = wp.tid()
    ncol = nx*ny
    i = c % nx
    j = (c // nx) % ny
    k = c // ncol
    one = wp.float64(1.0)
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
    tx_p[k,j,i] = txp
    tx_m[k,j,i] = txm
    ty_p[k,j,i] = typ
    ty_m[k,j,i] = tym
    tz_p[k,j,i] = tzp
    tz_m[k,j,i] = tzm
    principal = storage_diag[c] + boundary_col0[c]
    principal_storage[k,j,i] = principal
    if active[c] == 0 or fixed[c] != 0:
        diag[k,j,i] = one
        scale_flat[c] = one
        return
    diagonal = txp + txm + typ + tym + tzp + tzm + principal
    diag[k,j,i] = diagonal
    scale_flat[c] = diagonal


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
    free: wp.array(dtype=wp.int32), delta: wp.array(dtype=wp.float64),
    partials: wp.array2d(dtype=wp.float64), block_span: int):
    """Block partials over free cells: [0] dh^2, [1] dh_max, [2] flow r^2,
    [3] head-equivalent r^2, [4] gross physical throughput, [5] nonfinite flag, [6] residual L1,
    [7] unrelaxed, unclipped correction maximum.
    residual holds the independently evaluated physical balance. The rhs
    argument carries per-cell gross throughput, not a Krylov-system RHS."""
    t = wp.tid()
    block = t//block_span
    dh = candidate[t]-head[t]
    r = residual[t]
    sc = scale[t]
    is_free = free[t] != 0
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    dh2 = wp.where(is_free, dh*dh, zero)
    dhmax = wp.where(is_free, wp.abs(dh), zero)
    r2 = wp.where(is_free, r*r, zero)
    heq = wp.where(is_free and sc > 0.0, (r/sc)*(r/sc), zero)
    gross = wp.where(is_free, rhs[t], zero)
    bad = wp.where(is_free and (not wp.isfinite(candidate[t]) or not wp.isfinite(r)
        or not wp.isfinite(sc) or sc <= zero), one, zero)
    wp.atomic_add(partials, block, 0, dh2)
    wp.atomic_max(partials, block, 1, dhmax)
    wp.atomic_add(partials, block, 2, r2)
    wp.atomic_add(partials, block, 3, heq)
    wp.atomic_add(partials, block, 4, gross)
    wp.atomic_add(partials, block, 5, bad)
    wp.atomic_add(partials, block, 6, wp.where(is_free, wp.abs(r), zero))
    wp.atomic_max(partials, block, 7, wp.where(is_free, wp.abs(delta[t]), zero))


@wp.kernel
def terrain_combine_partials_kernel(partials: wp.array2d(dtype=wp.float64),
    out: wp.array(dtype=wp.float64), blocks: int):
    if wp.tid() == 0:
        sum_dh2 = wp.float64(0.0)
        max_dh = wp.float64(0.0)
        sum_r2 = wp.float64(0.0)
        sum_heq = wp.float64(0.0)
        sum_throughput = wp.float64(0.0)
        sum_bad = wp.float64(0.0)
        sum_abs_residual = wp.float64(0.0)
        max_delta = wp.float64(0.0)
        for index in range(blocks):
            sum_dh2 = sum_dh2 + partials[index, 0]
            max_dh = wp.max(max_dh, partials[index, 1])
            sum_r2 = sum_r2 + partials[index, 2]
            sum_heq = sum_heq + partials[index, 3]
            sum_throughput = sum_throughput + partials[index, 4]
            sum_bad = sum_bad + partials[index, 5]
            sum_abs_residual = sum_abs_residual + partials[index, 6]
            max_delta = wp.max(max_delta, partials[index, 7])
        out[0] = sum_dh2
        out[1] = max_dh
        out[2] = sum_r2
        out[3] = sum_heq
        out[4] = sum_throughput
        out[5] = sum_bad
        out[6] = sum_abs_residual
        out[7] = max_delta


@wp.kernel
def terrain_scale_divide_kernel(residual: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    """Row-scale a raw residual into the Krylov correction RHS."""
    c = wp.tid()
    out[c] = residual[c]/scale[c]


@wp.kernel
def terrain_scale_multiply_kernel(scaled: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    c = wp.tid()
    out[c] = scaled[c]*scale[c]


@wp.kernel
def terrain_coarse_faces_kernel(
    txp_f: wp.array(dtype=wp.float64, ndim=3), txm_f: wp.array(dtype=wp.float64, ndim=3),
    typ_f: wp.array(dtype=wp.float64, ndim=3), tym_f: wp.array(dtype=wp.float64, ndim=3),
    tzp_f: wp.array(dtype=wp.float64, ndim=3), tzm_f: wp.array(dtype=wp.float64, ndim=3),
    storage_f: wp.array(dtype=wp.float64, ndim=3),
    active_f: wp.array(dtype=wp.int32, ndim=3), bc_f: wp.array(dtype=wp.int32, ndim=3),
    txp_c: wp.array(dtype=wp.float64, ndim=3), txm_c: wp.array(dtype=wp.float64, ndim=3),
    typ_c: wp.array(dtype=wp.float64, ndim=3), tym_c: wp.array(dtype=wp.float64, ndim=3),
    tzp_c: wp.array(dtype=wp.float64, ndim=3), tzm_c: wp.array(dtype=wp.float64, ndim=3),
    storage_c: wp.array(dtype=wp.float64, ndim=3),
    active_c: wp.array(dtype=wp.int32, ndim=3), bc_c: wp.array(dtype=wp.int32, ndim=3),
    fy: int, fx: int, nx_f: int, ny_f: int, nz_f: int):
    """Semi-coarsened 7-point operator refresh replicating the host
    _coarsen_faces_consistent_axes: crossing fine faces on coarse cell
    boundaries over both-free fine pairs, 1/(fy*fx) block scaling,
    Dirichlet springs riding on the coarse storage slot. Launch over the
    FINE grid; each fine cell contributes to its coarse cell."""
    k, j, i = wp.tid()
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    inv = one/(wp.float64(fy)*wp.float64(fx))
    free_f = (active_f[k, j, i] != 0) and (bc_f[k, j, i] == 0)
    jco = j//fy
    ico = i//fx
    # x-face plane between coarse ico and ico+1 sits at fine i = (ico+1)*fx-1
    if i == (ico+1)*fx-1 and ico+1 < (nx_f+fx-1)//fx and i+1 < nx_f:
        both = free_f and (active_f[k, j, i+1] != 0) and (bc_f[k, j, i+1] == 0)
        if both:
            wp.atomic_add(txp_c, k, jco, ico, txp_f[k, j, i]*inv)
    # y-face plane between coarse jco and jco+1 at fine j = (jco+1)*fy-1
    if j == (jco+1)*fy-1 and jco+1 < (ny_f+fy-1)//fy and j+1 < ny_f:
        both = free_f and (active_f[k, j+1, i] != 0) and (bc_f[k, j+1, i] == 0)
        if both:
            wp.atomic_add(typ_c, k, jco, ico, typ_f[k, j, i]*inv)
    # z faces: block-summed over the fy x fx fine patch, k -> k+1
    if k+1 < nz_f:
        both = free_f and (active_f[k+1, j, i] != 0) and (bc_f[k+1, j, i] == 0)
        if both:
            wp.atomic_add(tzp_c, k, jco, ico, tzp_f[k, j, i]*inv)
    # Dirichlet springs: this fine cell free, neighbour fixed
    spr = zero
    if free_f:
        if i+1 < nx_f and bc_f[k, j, i+1] != 0:
            spr += txp_f[k, j, i]
        if i > 0 and bc_f[k, j, i-1] != 0:
            spr += txm_f[k, j, i]
        if j+1 < ny_f and bc_f[k, j+1, i] != 0:
            spr += typ_f[k, j, i]
        if j > 0 and bc_f[k, j-1, i] != 0:
            spr += tym_f[k, j, i]
        if k+1 < nz_f and bc_f[k+1, j, i] != 0:
            spr += tzp_f[k, j, i]
        if k > 0 and bc_f[k-1, j, i] != 0:
            spr += tzm_f[k, j, i]
        wp.atomic_add(storage_c, k, jco, ico, storage_f[k, j, i]*inv + spr*inv)


@wp.kernel
def terrain_coarse_shifts_kernel(
    txp_c: wp.array(dtype=wp.float64, ndim=3), txm_c: wp.array(dtype=wp.float64, ndim=3),
    typ_c: wp.array(dtype=wp.float64, ndim=3), tym_c: wp.array(dtype=wp.float64, ndim=3),
    tzp_c: wp.array(dtype=wp.float64, ndim=3), tzm_c: wp.array(dtype=wp.float64, ndim=3),
    storage_c: wp.array(dtype=wp.float64, ndim=3),
    active_c: wp.array(dtype=wp.int32, ndim=3), bc_c: wp.array(dtype=wp.int32, ndim=3),
    faces_sum: wp.array(dtype=wp.float64, ndim=3)):
    """Reciprocity shifts (txm[jc,ic] = txp[jc,ic-1] etc.), the inactive/
    fixed zeroing of the coarse storage slot, and the coarse diagonal."""
    k, j, i = wp.tid()
    zero = wp.float64(0.0)
    # Single-writer reciprocity: each txm/ty/tzm slot is written exactly
    # once (edge threads write the zero), so no ordering hazard exists.
    if i == 0:
        txm_c[k, j, i] = zero
    else:
        txm_c[k, j, i] = txp_c[k, j, i-1]
    if j == 0:
        tym_c[k, j, i] = zero
    else:
        tym_c[k, j, i] = typ_c[k, j-1, i]
    if k == 0:
        tzm_c[k, j, i] = zero
    else:
        tzm_c[k, j, i] = tzp_c[k-1, j, i]
    if active_c[k, j, i] == 0 or bc_c[k, j, i] != 0:
        storage_c[k, j, i] = zero
    diag = txp_c[k, j, i] + txm_c[k, j, i] + typ_c[k, j, i] + tym_c[k, j, i] \
        + tzp_c[k, j, i] + tzm_c[k, j, i] + storage_c[k, j, i]
    faces_sum[k, j, i] = diag


@wp.kernel
def terrain_subface_gate_kernel(sub_cell: wp.array(dtype=wp.int32),
    sub_c: wp.array(dtype=wp.float64), sub_bz: wp.array(dtype=wp.float64),
    sub_bx: wp.array(dtype=wp.float64), sub_by: wp.array(dtype=wp.float64),
    sub_hb: wp.array(dtype=wp.float64), sub_discharge: wp.array(dtype=wp.int32),
    gx: wp.array(dtype=wp.float64), gy: wp.array(dtype=wp.float64),
    gz: wp.array(dtype=wp.float64), head_relative: wp.array(dtype=wp.float64),
    sub_open: wp.array(dtype=wp.int32)):
    """Evaluate every exterior subface INDEPENDENTLY at the accepted head:
    a discharge-only subface is open exactly when its own physical flux
    (relative units; hb carries the datum shift) is outgoing."""
    s = wp.tid()
    if sub_discharge[s] == 0:
        sub_open[s] = 1
        return
    cell = sub_cell[s]
    q = sub_c[s]*(head_relative[cell]-sub_hb[s]) \
        - sub_bz[s]*gz[cell] - sub_bx[s]*gx[cell] - sub_by[s]*gy[cell]
    sub_open[s] = wp.where(q > wp.float64(0.0), 1, 0)


@wp.kernel
def terrain_subface_scatter_kernel(sub_cell: wp.array(dtype=wp.int32),
    sub_c: wp.array(dtype=wp.float64), sub_bz: wp.array(dtype=wp.float64),
    sub_bx: wp.array(dtype=wp.float64), sub_by: wp.array(dtype=wp.float64),
    sub_hb: wp.array(dtype=wp.float64), sub_open: wp.array(dtype=wp.int32),
    boundary: wp.array2d(dtype=wp.float64), contact_rhs: wp.array(dtype=wp.float64),
    open_col0: wp.array(dtype=wp.float64)):
    """Aggregate only the gate-open subfaces into the per-cell boundary
    rows (columns 0-3), the contact RHS offset c*(hb-datum), and the open
    conductance column used by the preconditioner extraction."""
    s = wp.tid()
    if sub_open[s] == 0:
        return
    cell = sub_cell[s]
    wp.atomic_add(boundary, cell, 0, sub_c[s])
    wp.atomic_add(boundary, cell, 1, sub_bz[s])
    wp.atomic_add(boundary, cell, 2, sub_bx[s])
    wp.atomic_add(boundary, cell, 3, sub_by[s])
    wp.atomic_add(contact_rhs, cell, sub_c[s]*sub_hb[s])
    wp.atomic_add(open_col0, cell, sub_c[s])


@wp.kernel
def terrain_contact_refresh_kernel(cell: wp.array(dtype=wp.int32),
    fraction: wp.array(dtype=wp.float64), dry: wp.array2d(dtype=wp.float64),
    wet: wp.array2d(dtype=wp.float64), normals: wp.array2d(dtype=wp.float64),
    conductance: wp.array(dtype=wp.float64), bz: wp.array(dtype=wp.float64),
    bx: wp.array(dtype=wp.float64), by: wp.array(dtype=wp.float64)):
    """Contact rows are affine in wet fraction except overridden top/bottom
    metric couplings (bz, bx and by), whose affine numerators divide by normal K."""
    t = wp.tid()
    f = fraction[cell[t]]
    weight = f
    normal = (wp.float64(1.0)-f)*normals[t, 0]+f*normals[t, 1]
    if normals[t, 2] != wp.float64(0.0) and normal > wp.float64(0.0):
        weight = f*normals[t, 1]/normal
    conductance[t] = dry[t, 1]+f*(wet[t, 1]-dry[t, 1])
    bz[t] = dry[t, 2]+weight*(wet[t, 2]-dry[t, 2])
    bx[t] = dry[t, 3]+weight*(wet[t, 3]-dry[t, 3])
    by[t] = dry[t, 4]+weight*(wet[t, 4]-dry[t, 4])


@wp.kernel
def terrain_contact_flux_kernel(cell: wp.array(dtype=wp.int32),
    conductance: wp.array(dtype=wp.float64), bz: wp.array(dtype=wp.float64),
    bx: wp.array(dtype=wp.float64), by: wp.array(dtype=wp.float64),
    hb: wp.array(dtype=wp.float64), opened: wp.array(dtype=wp.int32),
    head: wp.array(dtype=wp.float64), gx: wp.array(dtype=wp.float64),
    gy: wp.array(dtype=wp.float64), gz: wp.array(dtype=wp.float64),
    net: wp.array(dtype=wp.float64), gross: wp.array(dtype=wp.float64)):
    t = wp.tid()
    c = cell[t]
    if opened[t] != 0:
        q = conductance[t]*(head[c]-hb[t])-bz[t]*gz[c]-bx[t]*gx[c]-by[t]*gy[c]
        wp.atomic_add(net, c, q)
        wp.atomic_add(gross, c, wp.abs(q))


@wp.kernel
def terrain_physical_residual_kernel(source: wp.array(dtype=wp.float64),
    storage_change: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32), qx: wp.array(dtype=wp.float64),
    qy: wp.array(dtype=wp.float64), qz: wp.array(dtype=wp.float64),
    contact: wp.array(dtype=wp.float64), contact_gross: wp.array(dtype=wp.float64),
    residual: wp.array(dtype=wp.float64), throughput: wp.array(dtype=wp.float64),
    nx: int, ny: int, nz: int):
    """Independent flux balance, preserving source and exact volume change.
    Count each free/fixed exchange and each contact before cancellation."""
    c = wp.tid()
    residual[c] = wp.float64(0.0)
    throughput[c] = wp.float64(0.0)
    if free[c] == 0:
        return
    i = c % nx
    j = (c//nx) % ny
    k = c//(nx*ny)
    divergence = qx[c]+qy[c]+qz[c]+contact[c]
    gross = wp.abs(source[c])+wp.abs(storage_change[c])+contact_gross[c]
    if i+1 < nx and fixed[c+1] != 0:
        gross += wp.abs(qx[c])
    if j+1 < ny and fixed[c+nx] != 0:
        gross += wp.abs(qy[c])
    if k+1 < nz and fixed[c+nx*ny] != 0:
        gross += wp.abs(qz[c])
    if i > 0:
        divergence -= qx[c-1]
        if fixed[c-1] != 0:
            gross += wp.abs(qx[c-1])
    if j > 0:
        divergence -= qy[c-nx]
        if fixed[c-nx] != 0:
            gross += wp.abs(qy[c-nx])
    if k > 0:
        divergence -= qz[c-nx*ny]
        if fixed[c-nx*ny] != 0:
            gross += wp.abs(qz[c-nx*ny])
    residual[c] = source[c]-divergence-storage_change[c]
    throughput[c] = gross
