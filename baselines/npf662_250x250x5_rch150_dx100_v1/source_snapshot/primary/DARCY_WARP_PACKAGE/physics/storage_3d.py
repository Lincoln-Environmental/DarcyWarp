"""Integrated Sy/Ss storage: stable differences without subtracting potentials."""
import numpy as np


def storage_increment_3d(*, head, previous, bottoms, thickness, sy, ss):
    """Exact difference of Sy*b+Ss*(b²/2+dz*excess), per plan area [m].

    Factoring the square difference retains tiny elastic changes alongside
    much larger, unchanged specific-yield storage. There is no dimensional
    head-change cutoff and no numerical conductance-floor storage.
    """
    wet = np.clip(np.asarray(head)-bottoms,0,thickness)
    old_wet = np.clip(np.asarray(previous)-bottoms,0,thickness)
    difference = wet-old_wet
    excess = np.maximum(np.asarray(head)-(bottoms+thickness),0)
    old_excess = np.maximum(np.asarray(previous)-(bottoms+thickness),0)
    return sy*difference+ss*(.5*difference*(wet+old_wet)+thickness*(excess-old_excess))


def seed_zero_capacity_state_3d(*, head, previous, bottoms, thickness, sy, ss,
                                source, free, area, dt):
    """Seed a nonlinear guess where the current tangent cannot start a step.

    Dry injection and saturated Sy-only drainage are physically valid even
    with zero current tangent. Invert the local integrated law for a guess;
    retain the supplied old state and require the full final flux/storage
    balance. No storage is attributed to numerical conductance saturation.
    """
    rewet = (head <= bottoms)&(source > 0)&((sy > 0)|(ss > 0))&free
    drain = (head >= bottoms+thickness)&(source < 0)&(sy > 0)&(ss == 0)&free
    seed = rewet|drain
    if not np.any(seed):
        return 0
    old_wet = np.clip(previous-bottoms,0,thickness)
    old_excess = np.maximum(previous-(bottoms+thickness),0)
    old_potential = sy*old_wet+ss*(.5*old_wet**2+thickness*old_excess)
    target = np.maximum(old_potential+source*dt/area,0)
    denominator = sy+np.sqrt(sy**2+2*ss*target)
    wet = np.divide(2*target,denominator,out=np.zeros_like(target),where=denominator > 0)
    maximum = sy*thickness+.5*ss*thickness**2
    excess = np.divide(np.maximum(target-maximum,0),ss*thickness,
        out=np.zeros_like(target),where=np.broadcast_to(ss > 0,target.shape))
    guess = bottoms+np.minimum(wet,thickness)+excess
    head[seed] = guess[seed]
    return int(np.count_nonzero(seed))
