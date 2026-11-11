"""Reusable vertical factors for homogeneous NPF preconditioner corrections."""
import warp as wp


@wp.kernel
def factor_lines(diagonal: wp.array3d(dtype=wp.float64),
    zp: wp.array3d(dtype=wp.float64), zm: wp.array3d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32), fixed: wp.array3d(dtype=wp.int32),
    inverse: wp.array3d(dtype=wp.float64), upper: wp.array3d(dtype=wp.float64),
    lower: wp.array3d(dtype=wp.float64)):
    j, i = wp.tid()
    previous = wp.float64(0.)
    for k in range(diagonal.shape[0]):
        inv = wp.float64(0.)
        cp = wp.float64(0.)
        lp = wp.float64(0.)
        if active[k,j,i] != 0 and fixed[k,j,i] == 0:
            a = wp.float64(0.)
            c = wp.float64(0.)
            if k > 0:
                if active[k-1,j,i] != 0 and fixed[k-1,j,i] == 0:
                    a = zm[k,j,i]
            if k+1 < diagonal.shape[0]:
                if active[k+1,j,i] != 0 and fixed[k+1,j,i] == 0:
                    c = zp[k,j,i]
            pivot = diagonal[k,j,i]+a*previous
            if pivot != wp.float64(0.):
                inv = wp.float64(1.)/pivot
                cp = -c/pivot
                lp = a/pivot
        inverse[k,j,i] = inv
        upper[k,j,i] = cp
        lower[k,j,i] = lp
        previous = cp


@wp.kernel
def smooth_factored_lines(xp: wp.array3d(dtype=wp.float64), xm: wp.array3d(dtype=wp.float64),
    yp: wp.array3d(dtype=wp.float64), ym: wp.array3d(dtype=wp.float64),
    diagonal: wp.array3d(dtype=wp.float64), active: wp.array3d(dtype=wp.int32),
    fixed: wp.array3d(dtype=wp.int32), rhs: wp.array3d(dtype=wp.float64),
    old: wp.array3d(dtype=wp.float64), inverse: wp.array3d(dtype=wp.float64),
    upper: wp.array3d(dtype=wp.float64), lower: wp.array3d(dtype=wp.float64),
    forward: wp.array3d(dtype=wp.float64), out: wp.array3d(dtype=wp.float64), omega: wp.float64):
    j, i = wp.tid()
    previous = wp.float64(0.)
    for k in range(old.shape[0]):
        value = wp.float64(0.)
        if active[k,j,i] != 0 and fixed[k,j,i] == 0:
            value = rhs[k,j,i]
            if i+1 < old.shape[2]: value += xp[k,j,i]*old[k,j,i+1]
            if i > 0: value += xm[k,j,i]*old[k,j,i-1]
            if j+1 < old.shape[1]: value += yp[k,j,i]*old[k,j+1,i]
            if j > 0: value += ym[k,j,i]*old[k,j-1,i]
            value = value*inverse[k,j,i]+lower[k,j,i]*previous
            if diagonal[k,j,i] == wp.float64(0.): value = old[k,j,i]
        forward[k,j,i] = value
        previous = value
    previous = wp.float64(0.)
    for reverse in range(old.shape[0]):
        k = old.shape[0]-1-reverse
        value = forward[k,j,i]-upper[k,j,i]*previous
        previous = value
        out[k,j,i] = wp.float64(0.)
        if active[k,j,i] != 0 and fixed[k,j,i] == 0:
            out[k,j,i] = old[k,j,i]+omega*(value-old[k,j,i])
