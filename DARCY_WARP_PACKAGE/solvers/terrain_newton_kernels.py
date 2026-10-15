"""Analytic forward directional derivatives of the discrete terrain fluxes.

All geometry is fixed. At clamp endpoints choose the constant branch (df=0);
at a zero discharge flux choose the closed branch. The vector is projected to
zero on prescribed/isolated/inactive cells, without removing any dry free row.
"""
import warp as wp

from DARCY_WARP_PACKAGE.solvers.terrain_device_kernels import terrain_harmonic_resistance


@wp.func
def harmonic_direction(left: wp.float64, right: wp.float64, dl: wp.float64,
                       dr: wp.float64, dleft: wp.float64, dright: wp.float64):
    conductance, wl, wr = terrain_harmonic_resistance(left, right, dl, dr)
    # C = tl*tr/(tl+tr), wl=tr/(tl+tr), wr=tl/(tl+tr).
    # Scaled arithmetic avoids squaring large half-cell conductances.
    tl = left/dl
    tr = right/dr
    large = wp.max(tl, tr)
    dc = wp.float64(0.0)
    dwl = wp.float64(0.0)
    if large > wp.float64(0.0):
        dc = wl*wl*dleft/dl + wr*wr*dright/dr
        dwl = (wr*(dright/dr/large)-wl*(dleft/dl/large))/(tl/large+tr/large)
    return dc, dwl, -dwl


@wp.kernel
def project_direction(v: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
                      projected: wp.array(dtype=wp.float64)):
    c = wp.tid()
    projected[c] = wp.where(free[c] != 0, v[c], wp.float64(0.0))


@wp.func
def fraction_direction(c: int, head: wp.array(dtype=wp.float64),
                       bottoms: wp.array(dtype=wp.float64), thickness: wp.array(dtype=wp.float64),
                       v: wp.array(dtype=wp.float64), floor: wp.float64):
    wet = head[c]-bottoms[c]
    result = wp.float64(0.0)
    if wet > wp.min(floor, thickness[c]) and wet < thickness[c]:
        result = v[c]/thickness[c]
    return result


@wp.kernel
def coefficient_direction(head: wp.array(dtype=wp.float64), v: wp.array(dtype=wp.float64),
    bottoms: wp.array(dtype=wp.float64), fraction: wp.array(dtype=wp.float64),
    kx: wp.array(dtype=wp.float64), ky: wp.array(dtype=wp.float64), kz: wp.array(dtype=wp.float64),
    thickness: wp.array(dtype=wp.float64), zx_interfaces: wp.array(dtype=wp.float64),
    zy_interfaces: wp.array(dtype=wp.float64), zx_centre: wp.array(dtype=wp.float64),
    zy_centre: wp.array(dtype=wp.float64), zxf: wp.array(dtype=wp.float64), zyf: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32), derivative: wp.array2d(dtype=wp.float64),
    floor: wp.float64, dx: wp.float64, dy: wp.float64, nx: int, ny: int, nz: int):
    c = wp.tid()
    ncol = nx*ny
    i = c % nx
    j = (c//nx) % ny
    k = c//ncol
    for entry in range(13):
        derivative[c, entry] = wp.float64(0.0)
    if active[c] == 0:
        return
    half = wp.float64(0.5)
    df = fraction_direction(c, head, bottoms, thickness, v, floor)
    fx = kx[c]*fraction[c]
    fy = ky[c]*fraction[c]
    dfx = kx[c]*df
    dfy = ky[c]*df
    tz = thickness[c]
    tx = fx*tz
    ty = fy*tz
    dtx = dfx*tz
    dty = dfy*tz
    if i+1 < nx and active[c+1] != 0:
        cn = c+1
        txr = kx[cn]*fraction[cn]*thickness[cn]
        dtxr = kx[cn]*fraction_direction(cn, head, bottoms, thickness, v, floor)*thickness[cn]
        inv, wl, wr = terrain_harmonic_resistance(tx, txr, half*dx, half*dx)
        dc, dwl, dwr = harmonic_direction(tx, txr, half*dx, half*dx, dtx, dtxr)
        derivative[c, 0] = dc*dy
        derivative[c, 1] = (dwl*tx+wl*dtx)*zxf[c]*dy
        derivative[c, 2] = (dwr*txr+wr*dtxr)*zxf[c]*dy
    if j+1 < ny and active[c+nx] != 0:
        cn = c+nx
        tyr = ky[cn]*fraction[cn]*thickness[cn]
        dtyr = ky[cn]*fraction_direction(cn, head, bottoms, thickness, v, floor)*thickness[cn]
        inv, wl, wr = terrain_harmonic_resistance(ty, tyr, half*dy, half*dy)
        dc, dwl, dwr = harmonic_direction(ty, tyr, half*dy, half*dy, dty, dtyr)
        derivative[c, 3] = dc*dx
        derivative[c, 4] = (dwl*ty+wl*dty)*zyf[c]*dx
        derivative[c, 5] = (dwr*tyr+wr*dtyr)*zyf[c]*dx
    if k+1 < nz and active[c+ncol] != 0:
        cn = c+ncol
        interface = (k+1)*ncol+c % ncol
        zx = zx_interfaces[interface]
        zy = zy_interfaces[interface]
        fxn = kx[cn]*fraction[cn]
        fyn = ky[cn]*fraction[cn]
        dfn = fraction_direction(cn, head, bottoms, thickness, v, floor)
        dfxn = kx[cn]*dfn
        dfyn = ky[cn]*dfn
        left = fx*zx*zx+fy*zy*zy+kz[c]
        right = fxn*zx*zx+fyn*zy*zy+kz[cn]
        dleft = dfx*zx*zx+dfy*zy*zy
        dright = dfxn*zx*zx+dfyn*zy*zy
        inv, wl, wr = terrain_harmonic_resistance(left, right, half*tz, half*thickness[cn])
        dc, dwl, dwr = harmonic_direction(left, right, half*tz, half*thickness[cn], dleft, dright)
        area = dx*dy
        derivative[c, 6] = dc*area
        derivative[c, 7] = (dwl*fx+wl*dfx)*zx*area
        derivative[c, 8] = (dwr*fxn+wr*dfxn)*zx*area
        derivative[c, 9] = (dwl*fy+wl*dfy)*zy*area
        derivative[c, 10] = (dwr*fyn+wr*dfyn)*zy*area
        derivative[c, 11] = ((dwl*fx+wl*dfx)*zx*(zx_centre[c]-zx)
            +(dwl*fy+wl*dfy)*zy*(zy_centre[c]-zy))*area
        derivative[c, 12] = ((dwr*fxn+wr*dfxn)*zx*(zx_centre[cn]-zx)
            +(dwr*fyn+wr*dfyn)*zy*(zy_centre[cn]-zy))*area


@wp.kernel
def add_face_direction(head: wp.array(dtype=wp.float64), gx: wp.array(dtype=wp.float64),
    gy: wp.array(dtype=wp.float64), gz: wp.array(dtype=wp.float64), dc: wp.array2d(dtype=wp.float64),
    qx: wp.array(dtype=wp.float64), qy: wp.array(dtype=wp.float64), qz: wp.array(dtype=wp.float64),
    nx: int, ny: int, nz: int):
    # q already contains C Gv; add (DC[v]) Gh, including all metric columns.
    c = wp.tid()
    if c % nx+1 < nx:
        qx[c] += dc[c, 0]*(head[c]-head[c+1])-dc[c, 1]*gz[c]-dc[c, 2]*gz[c+1]
    if (c//nx) % ny+1 < ny:
        qy[c] += dc[c, 3]*(head[c]-head[c+nx])-dc[c, 4]*gz[c]-dc[c, 5]*gz[c+nx]
    ncol = nx*ny
    if c//ncol+1 < nz:
        cn = c+ncol
        qz[c] += dc[c, 6]*(head[c]-head[cn])-dc[c, 7]*gx[c]-dc[c, 8]*gx[cn] \
            -dc[c, 9]*gy[c]-dc[c, 10]*gy[cn]-dc[c, 11]*gz[c]-dc[c, 12]*gz[cn]


@wp.kernel
def contact_direction(cell: wp.array(dtype=wp.int32), opened: wp.array(dtype=wp.int32),
    conductance: wp.array(dtype=wp.float64), bz: wp.array(dtype=wp.float64),
    bx: wp.array(dtype=wp.float64), by: wp.array(dtype=wp.float64), hb: wp.array(dtype=wp.float64),
    dry: wp.array2d(dtype=wp.float64), wet: wp.array2d(dtype=wp.float64), normals: wp.array2d(dtype=wp.float64),
    head: wp.array(dtype=wp.float64), relative: wp.array(dtype=wp.float64),
    bottoms: wp.array(dtype=wp.float64), thickness: wp.array(dtype=wp.float64), fraction: wp.array(dtype=wp.float64),
    v: wp.array(dtype=wp.float64), gx: wp.array(dtype=wp.float64), gy: wp.array(dtype=wp.float64), gz: wp.array(dtype=wp.float64),
    vx: wp.array(dtype=wp.float64), vy: wp.array(dtype=wp.float64), vz: wp.array(dtype=wp.float64),
    net: wp.array(dtype=wp.float64), floor: wp.float64):
    t = wp.tid()
    if opened[t] == 0:
        return
    c = cell[t]
    df = fraction_direction(c, head, bottoms, thickness, v, floor)
    f = fraction[c]
    dw = df
    normal = (wp.float64(1.0)-f)*normals[t, 0]+f*normals[t, 1]
    if normals[t, 2] != wp.float64(0.0) and normal > wp.float64(0.0):
        dw = df*(normals[t, 1]/normal)*(normals[t, 0]/normal)
    dconductance = df*(wet[t, 1]-dry[t, 1])
    dbz = dw*(wet[t, 2]-dry[t, 2])
    dbx = dw*(wet[t, 3]-dry[t, 3])
    dby = dw*(wet[t, 4]-dry[t, 4])
    value = conductance[t]*v[c]-bz[t]*vz[c]-bx[t]*vx[c]-by[t]*vy[c] \
        +dconductance*(relative[c]-hb[t])-dbz*gz[c]-dbx*gx[c]-dby*gy[c]
    wp.atomic_add(net, c, value)


@wp.kernel
def jacobian_divergence(v: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64),
    qx: wp.array(dtype=wp.float64), qy: wp.array(dtype=wp.float64), qz: wp.array(dtype=wp.float64),
    contact: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float64), alpha: wp.float64, beta: wp.float64,
    counter: wp.array(dtype=wp.int64), nx: int, ny: int):
    c = wp.tid()
    value = v[c]
    if free[c] != 0:
        value = qx[c]+qy[c]+qz[c]+contact[c]
        if c % nx > 0:
            value -= qx[c-1]
        if (c//nx) % ny > 0:
            value -= qy[c-nx]
        if c//(nx*ny) > 0:
            value -= qz[c-nx*ny]
        value /= scale[c]
    result = alpha*value
    if beta != wp.float64(0.0):
        result += beta*y[c]
    out[c] = result
    if c == 0:
        counter[0] += wp.int64(1)
