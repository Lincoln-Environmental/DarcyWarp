"""Finite physical acceptance and stable FP64 diagnostic reductions."""
import math
import numpy as np


def componentwise_roundoff(*, head, source, residual, free, left, right,
                          conductance, ghb_conductance, ghb_head,
                          storage=None, tangent=None, previous=None,
                          previous_tangent=None, storage_coordinate_scale=None):
    """Independent backward-error indicator for the frozen endpoint equations.

    The allowance is 32 FP64 eps times the absolute row terms, before head
    subtraction. This is a backward residual test, not a forward head-error
    guarantee. Zero rows require an exactly zero residual.
    """
    h = np.asarray(head).ravel()
    edge_scale = conductance*(np.abs(h[left])+np.abs(h[right]))
    scale = np.abs(np.asarray(source).ravel()).copy()
    scale += np.bincount(left, weights=edge_scale, minlength=h.size)
    scale += np.bincount(right, weights=edge_scale, minlength=h.size)
    selected = np.asarray(ghb_conductance).ravel() > 0.
    scale[selected] += np.asarray(ghb_conductance).ravel()[selected]*(
        np.abs(h[selected])+np.abs(np.asarray(ghb_head).ravel()[selected]))
    if storage is not None:
        scale += np.abs(np.asarray(storage).ravel())
        coordinates = np.abs(h)+np.abs(np.asarray(previous).ravel())
        if storage_coordinate_scale is not None:
            coordinates += 2.*np.asarray(storage_coordinate_scale).ravel()
        scale += (np.abs(np.asarray(tangent).ravel())+
                  np.abs(np.asarray(previous_tangent).ravel()))*coordinates
    selected = np.asarray(free).ravel()
    allowance = 32.*np.finfo(float).eps*scale[selected]
    error = np.abs(np.asarray(residual).ravel()[selected])
    valid = np.all(np.isfinite(allowance)) and np.all(allowance >= 0.) and np.all(np.isfinite(error))
    ratio = np.divide(error, allowance, out=np.zeros_like(error), where=allowance > 0.)
    ratio[(allowance == 0.) & (error != 0.)] = np.inf
    maximum = float(np.max(ratio, initial=0.)) if valid else float('inf')
    return dict(roundoff_bound_ratio_max=maximum, roundoff_rows_valid=bool(valid and maximum <= 1.))


def apply_roundoff_acceptance(*, checks, certificate):
    """Explicit precision mode retains the original physical conservation gate."""
    checks.update(certificate)
    checks['requested_thresholds_met'] = bool(checks['independent_accepted'])
    conservation = checks['physical_rms'] <= 1e-12 or checks['relative_cell_imbalance'] <= 1e-8
    accepted = bool(checks['input_valid'] and checks['boundary_condition_valid']
        and checks['physical_metrics_valid'] and checks['ungrounded_current_components'] == 0
        and checks['head_residual_rms'] <= min(1e-12, checks['final_acceptance_thresholds']['head_equivalent_rms_m'])
        and conservation and certificate['roundoff_rows_valid'])
    checks.update(roundoff_accepted=accepted, independent_accepted=accepted,
        precision_limited=bool(accepted and not checks['requested_thresholds_met']))


def final_tolerances(*, physical_tolerance=1e-12, relative_tolerance=1e-8,
                     head_tolerance_m=1e-6, boundary_tolerance_m=1e-10):
    values = dict(physical_rms_m3_day=physical_tolerance,
                  relative_cell_imbalance=relative_tolerance,
                  head_equivalent_rms_m=head_tolerance_m,
                  prescribed_head_error_m=boundary_tolerance_m)
    for name, value in values.items():
        if np.ndim(value) != 0 or not np.isfinite(value) or value < 0.:
            raise ValueError(f'{name} tolerance must be a finite nonnegative scalar')
    return {name: float(value) for name, value in values.items()}


def stable_sum(*, values):
    selected = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(selected)):
        return float('nan')
    scale = float(np.max(np.abs(selected))) if selected.size else 0.
    if scale == 0.:
        return 0.
    return scale * math.fsum(selected / scale)


def stable_rms(*, values):
    selected = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(selected)):
        return float('inf')
    scale = float(np.max(np.abs(selected))) if selected.size else 0.
    if scale == 0.:
        return 0.
    return scale * float(np.sqrt(np.mean((selected / scale)**2)))
