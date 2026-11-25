"""Absolute prescribed-head and active-domain checks for saved results."""
import numpy as np


def invalid_head_values(*, head):
    """Identify nonfinite heads and reserved dry/inactive marker magnitudes.

    :param head: Scalar or array of returned hydraulic heads.
    :returns: Boolean invalid-head mask, including magnitudes at least 1e20.
    """
    head = np.asarray(head)
    return ~np.isfinite(head) | (np.abs(head) >= 1e20)


def inspect_heads(*, head, active, fixed, values, boundary_tolerance_m=1e-10):
    """Check active head validity and prescribed-head enforcement.

    :param head: Returned hydraulic head array, in metres.
    :param active: Active-cell mask.
    :param fixed: Prescribed-head mask.
    :param values: Prescribed hydraulic head values, in metres.
    :param boundary_tolerance_m: Absolute prescribed-head error tolerance in metres.
    :returns: Counts, maximum prescribed-head error and acceptance flags.
    :raises ValueError: If the returned head shape differs from the active mask.
    """
    active = np.asarray(active, dtype=bool)
    head = np.asarray(head)
    if head.shape != active.shape:
        raise ValueError('returned head shape differs from canonical domain')
    invalid = invalid_head_values(head=head)
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
    """Compute an extreme over the selected entries.

    :param values: Numerical array.
    :param mask: Boolean selector with compatible shape.
    :param minimum: Return minimum rather than maximum when true.
    :returns: Selected extreme, or None if no entries are selected.
    """
    selected = np.asarray(values)[mask]
    if not selected.size:
        return None
    return float(np.min(selected) if minimum else np.max(selected))


def masked_rms(*, values, mask):
    """Compute the root-mean-square value over the selected entries.

    :param values: Numerical array.
    :param mask: Boolean selector with compatible shape.
    :returns: Selected RMS, or None if no entries are selected.
    """
    selected = np.asarray(values)[mask]
    return float(np.sqrt(np.mean(selected ** 2))) if selected.size else None


def finite_diagnostics(*, value):
    """Convert nonfinite diagnostic scalars to None in nested report containers.

    :param value: Diagnostic scalar, dictionary, list or other object.
    :returns: Copied dictionary/list with nonfinite floats replaced by None;
        other values are returned unchanged.
    """
    if isinstance(value, dict):
        return {key: finite_diagnostics(value=item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_diagnostics(value=item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value
