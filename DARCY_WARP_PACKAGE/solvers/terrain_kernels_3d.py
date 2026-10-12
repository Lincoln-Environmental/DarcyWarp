"""FP64 conservative terrain flux operator for total hydraulic head.

One authoritative positive-orientation face array per axis, then a gather
divergence: opposite cell contributions are identical, without atomics.
"""
import warp as wp


@wp.kernel
def terrain_gradients_kernel(h: wp.array(dtype=wp.float64),
    horizontal: wp.array2d(dtype=wp.float64), depths: wp.array(dtype=wp.float64),
    gx: wp.array(dtype=wp.float64), gy: wp.array(dtype=wp.float64), gz: wp.array(dtype=wp.float64),
    nx: int, ny: int, nz: int):
    c = wp.tid()
    ncol = nx*ny
    col = c % ncol
    i = col % nx
    j = col // nx
    k = c // ncol
    x = wp.float64(0)
    y = wp.float64(0)
    z = wp.float64(0)
    if i > 0:
        x += horizontal[col,1]*(h[c-1]-h[c])
    if i+1 < nx:
        x += horizontal[col,2]*(h[c+1]-h[c])
    if j > 0:
        y += horizontal[col,4]*(h[c-nx]-h[c])
    if j+1 < ny:
        y += horizontal[col,5]*(h[c+nx]-h[c])
    if nz > 1:
        if k+1 < nz:
            z = (h[c+ncol]-h[c])/depths[k]
        else:
            z = (h[c]-h[c-ncol])/depths[k-1]
    gx[c] = x
    gy[c] = y
    gz[c] = z


@wp.kernel
def terrain_faces_kernel(h: wp.array(dtype=wp.float64), gx: wp.array(dtype=wp.float64),
    gy: wp.array(dtype=wp.float64), gz: wp.array(dtype=wp.float64),
    coefficients: wp.array2d(dtype=wp.float64),
    qx: wp.array(dtype=wp.float64), qy: wp.array(dtype=wp.float64), qz: wp.array(dtype=wp.float64),
    nx: int, ny: int, nz: int):
    c = wp.tid()
    i = c % nx
    j = (c // nx) % ny
    k = c // (nx*ny)
    ncol = nx*ny
    x = wp.float64(0)
    y = wp.float64(0)
    z = wp.float64(0)
    if i+1 < nx:
        x = coefficients[c,0]*(h[c]-h[c+1])-coefficients[c,1]*gz[c]-coefficients[c,2]*gz[c+1]
    if j+1 < ny:
        y = coefficients[c,3]*(h[c]-h[c+nx])-coefficients[c,4]*gz[c]-coefficients[c,5]*gz[c+nx]
    if k+1 < nz:
        z = coefficients[c,6]*(h[c]-h[c+ncol])-coefficients[c,7]*gx[c]-coefficients[c,8]*gx[c+ncol]-coefficients[c,9]*gy[c]-coefficients[c,10]*gy[c+ncol]
    qx[c] = x
    qy[c] = y
    qz[c] = z


@wp.kernel
def terrain_divergence_kernel(h: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64),
    gx: wp.array(dtype=wp.float64), gy: wp.array(dtype=wp.float64), gz: wp.array(dtype=wp.float64),
    qx: wp.array(dtype=wp.float64), qy: wp.array(dtype=wp.float64), qz: wp.array(dtype=wp.float64),
    storage: wp.array(dtype=wp.float64), boundary: wp.array2d(dtype=wp.float64),
    active: wp.array(dtype=wp.int32), fixed: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float64), row_scaled: int,
    out: wp.array(dtype=wp.float64), alpha: wp.float64, beta: wp.float64,
    nx: int, ny: int, nz: int):
    c = wp.tid()
    i = c % nx
    j = (c // nx) % ny
    k = c // (nx*ny)
    value = h[c]
    if active[c] != 0 and fixed[c] == 0:
        value = qx[c]+qy[c]+qz[c]+(storage[c]+boundary[c,0])*h[c]-boundary[c,1]*gz[c]-boundary[c,2]*gx[c]-boundary[c,3]*gy[c]
        if i > 0:
            value -= qx[c-1]
        if j > 0:
            value -= qy[c-nx]
        if k > 0:
            value -= qz[c-nx*ny]
    result = alpha*value
    if row_scaled != 0:
        result = result/scale[c]
    if beta != wp.float64(0):
        result += beta*y[c]
    out[c] = result


@wp.kernel
def terrain_combine_kernel(value: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64), alpha: wp.float64, beta: wp.float64):
    c = wp.tid()
    result = alpha*value[c]
    if beta != wp.float64(0):
        result += beta*y[c]
    out[c] = result


@wp.kernel
def terrain_preconditioner_rhs_kernel(source: wp.array(dtype=wp.float64),
    scale: wp.array(dtype=wp.float64),
    rhs: wp.array3d(dtype=wp.float64), x: wp.array3d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32), fixed: wp.array3d(dtype=wp.int32)):
    k,j,i = wp.tid()
    c = (k*rhs.shape[1]+j)*rhs.shape[2]+i
    value = wp.float64(0)
    if active[k,j,i] != 0 and fixed[k,j,i] == 0:
        value = source[c]*scale[c]
    rhs[k,j,i] = value
    x[k,j,i] = wp.float64(0)


@wp.kernel
def terrain_preconditioner_output_kernel(source: wp.array(dtype=wp.float64),
    correction: wp.array3d(dtype=wp.float64), active: wp.array3d(dtype=wp.int32),
    fixed: wp.array3d(dtype=wp.int32), value: wp.array(dtype=wp.float64)):
    k,j,i = wp.tid()
    c = (k*correction.shape[1]+j)*correction.shape[2]+i
    result = source[c]
    if active[k,j,i] != 0 and fixed[k,j,i] == 0:
        result = correction[k,j,i]
    value[c] = result
