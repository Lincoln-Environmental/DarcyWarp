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
