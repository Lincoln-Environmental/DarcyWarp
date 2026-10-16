"""Absolute prescribed-head and active-domain checks for saved results."""
import numpy as np


def inspect_heads(*, head, active, fixed, values, boundary_tolerance_m=1e-10):
    active = np.asarray(active, dtype=bool)
    head = np.asarray(head)
    if head.shape != active.shape:
        raise ValueError('returned head shape differs from canonical domain')
    invalid = ~np.isfinite(head) | (np.abs(head) >= 1e20)
    prescribed = active & np.asarray(fixed, dtype=bool)
    error = np.abs(head[prescribed] - np.asarray(values)[prescribed])
    maximum = float(np.max(error)) if error.size and np.all(np.isfinite(error)) else None
    valid_boundary = bool(not np.any(invalid[prescribed]) and
        (not error.size or maximum <= boundary_tolerance_m))
    return dict(prescribed_cells_checked=int(prescribed.sum()),
        prescribed_head_error_max_m=maximum, prescribed_head_tolerance_m=boundary_tolerance_m,
        boundary_condition_valid=valid_boundary, invalid_active_heads=int(np.sum(invalid & active)),
        inactive_marker_cells=int(np.sum(invalid & ~active)), active_cells_checked=int(active.sum()),
        input_valid=bool(not np.any(invalid & active)))


def masked_extreme(*, values, mask, minimum=False):
    selected = np.asarray(values)[mask]
    if not selected.size:
        return None
    return float(np.min(selected) if minimum else np.max(selected))


def masked_rms(*, values, mask):
    selected = np.asarray(values)[mask]
    return float(np.sqrt(np.mean(selected ** 2))) if selected.size else None
