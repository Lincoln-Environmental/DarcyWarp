"""Validation and canonical representation of reciprocal Cartesian faces."""
from __future__ import annotations

import numpy as np


def validate_reciprocal_faces(*, faces, active, canonicalize=False, copy_arrays=True):
    """Validate nonnegative finite internal faces, with no absolute cutoff.

    Reciprocal values may differ by at most 32 machine epsilons relative to
    their magnitude (using the least precise input dtype). Accepted roundoff
    is canonicalized to the positive-direction value when requested. Exterior
    slots are unused by the stencil; connections touching inactive cells must
    be exactly zero. Substantial asymmetry is never averaged.
    """
    act = np.asarray(active) != 0
    arrays = tuple(np.asarray(face) for face in faces)
    if len(arrays) != 6 or any(a.shape != act.shape for a in arrays):
        raise ValueError("Six face arrays must match active shape (nz, ny, nx).")
    for index, a in enumerate(arrays):
        if np.any(~np.isfinite(a)) or np.any(a < 0.0):
            bad = ~np.isfinite(a) | (a < 0.0)
            location = tuple(np.argwhere(bad)[0])
            raise ValueError(f"Face {index} at {location} must be nonnegative finite; got {a[location]}.")
    # Diagnostic callers may borrow exact inputs; canonicalization copies a
    # negative-direction array only if accepted roundoff actually differs.
    result = [a.copy() for a in arrays] if canonicalize and copy_arrays else list(arrays)
    for axis, p, m in ((2, 0, 1), (1, 2, 3), (0, 4, 5)):
        here = [slice(None)] * 3
        there = [slice(None)] * 3
        here[axis], there[axis] = slice(None, -1), slice(1, None)
        here, there = tuple(here), tuple(there)
        left, right = arrays[p][here], arrays[m][there]
        eps = max(np.finfo(a.dtype).eps if a.dtype.kind == "f" else np.finfo(np.float64).eps
                  for a in (arrays[p], arrays[m]))
        connected = act[here] & act[there]
        invalid = (~connected & ((left != 0.0) | (right != 0.0)))
        invalid |= np.abs(left - right) > 32.0 * eps * np.maximum(left, right)
        # A roundoff tolerance must never turn a positive connection into zero.
        invalid |= (left == 0.0) != (right == 0.0)
        if np.any(invalid):
            location = tuple(np.argwhere(invalid)[0])
            raise ValueError(f"Nonreciprocal or inactive internal face axis={axis} at {location}: "
                             f"positive={left[location]}, neighbour negative={right[location]}, "
                             f"difference={left[location] - right[location]}.")
        if canonicalize and (copy_arrays or np.any(left != right)):
            if not copy_arrays:
                result[m] = arrays[m].copy()
            result[m][there] = left
    return tuple(result)
