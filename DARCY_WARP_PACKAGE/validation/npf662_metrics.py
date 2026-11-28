"""Finite physical acceptance and stable FP64 diagnostic reductions."""
import math
import numpy as np


def exact_zero_equation_rows(*, case, head, free, diagonal, residual,
                             boundary_throughput, storage=None):
    """Identify hydraulically isolated, unforced, exact zero equations.

    :param case: Canonical NPF physical inputs.
    :param head: Evaluated total heads, not a numerical trial surrogate.
    :param free: Free-cell mask.
    :param diagonal: Physical conductance plus boundary and storage tangents.
    :param residual: Unmodified physical balance residual.
    :param boundary_throughput: Gross physical package exchange by cell.
    :param storage: Optional actual accumulation rate; a dry transient cell
        that is still releasing stored water is not a zero equation.
    :return: Boolean mask of exact zero rows; no tolerances or physical floors.
    """
    zero_rows = (free & (diagonal == 0.) & (residual == 0.)
                 & (case.source == 0.) & (boundary_throughput == 0.))
    if case.source_gross is not None:
        zero_rows &= np.asarray(case.source_gross) == 0.
    if storage is not None:
        zero_rows &= storage == 0.
    return zero_rows


# Compatibility name retained for existing internal callers.
exact_dry_zero_rows = exact_zero_equation_rows


def balanced_nullspace_components(*, labels, active, grounded, source,
                                  left, right, face_left_derivative,
                                  face_right_derivative,
                                  boundary_tangent, boundary_throughput,
                                  storage=None, storage_tangent=None):
    """Find balanced ungrounded components with an exact constant-head mode.

    A qualifying component has no fixed or head-dependent restraint or
    external exchange, and its physical Jacobian annihilates a uniform head
    increment. Its total source plus storage rate is exactly balanced. Final
    residual acceptance is evaluated separately, and a shifted candidate is
    checked against the full nonlinear equations before it can be accepted.

    :param labels: Connected-component labels from positive physical face
        conductances.
    :param active: Active-cell mask.
    :param grounded: Cells with a fixed head or positive external tangent.
    :param source: Signed source rate by cell.
    :param left: First cell index for each conducting face.
    :param right: Second cell index for each conducting face.
    :param face_left_derivative: Face-flow derivative with respect to its
        first cell head.
    :param face_right_derivative: Face-flow derivative with respect to its
        second cell head.
    :param boundary_tangent: Head-dependent boundary tangent by cell.
    :param boundary_throughput: Absolute external exchange by cell.
    :param storage: Optional actual storage rate by cell.
    :param storage_tangent: Optional storage derivative by cell.
    :return: Mapping from component label to its active flat cell indices.
    """
    labels = np.asarray(labels).ravel()
    active = np.asarray(active).ravel() != 0
    grounded = np.asarray(grounded).ravel() != 0
    source = np.asarray(source, dtype=np.float64).ravel()
    boundary_tangent = np.asarray(boundary_tangent, dtype=np.float64).ravel()
    boundary_throughput = np.asarray(boundary_throughput, dtype=np.float64).ravel()
    storage = None if storage is None else np.asarray(storage).ravel()
    storage_tangent = (None if storage_tangent is None
                       else np.asarray(storage_tangent).ravel())
    left = np.asarray(left, dtype=np.int64).ravel()
    right = np.asarray(right, dtype=np.int64).ravel()
    face_left_derivative = np.asarray(face_left_derivative, dtype=np.float64).ravel()
    face_right_derivative = np.asarray(face_right_derivative, dtype=np.float64).ravel()

    active_labels = set(np.unique(labels[active]))
    grounded_labels = set(np.unique(labels[active & grounded]))
    candidates = active_labels - grounded_labels
    # Residual size belongs to final acceptance. A balanced isolated system
    # can have a nonzero local residual while Newton is solving it; that does
    # not remove its constant-head nullspace.
    bad_cells = grounded | (boundary_tangent != 0.) | (boundary_throughput != 0.)
    if storage is not None:
        bad_cells |= storage != 0.
    if storage_tangent is not None:
        bad_cells |= storage_tangent != 0.
    bad_labels = set(np.unique(labels[active & bad_cells]))
    left_labels = labels[left]
    right_labels = labels[right]
    same_component = left_labels == right_labels
    bad_left = ((same_component &
                 ((face_left_derivative + face_right_derivative) != 0.))
                | (~same_component & (face_left_derivative != 0.)))
    bad_right = ((same_component &
                  ((face_left_derivative + face_right_derivative) != 0.))
                 | (~same_component & (face_right_derivative != 0.)))
    bad_labels.update(np.unique(left_labels[bad_left]))
    bad_labels.update(np.unique(right_labels[bad_right]))
    eligible_labels = candidates - bad_labels
    if not eligible_labels:
        return {}
    eligible_cells = np.flatnonzero(active & np.isin(labels, list(eligible_labels)))
    eligible_cell_labels = labels[eligible_cells]
    eligible = {}
    for label in np.unique(eligible_cell_labels):
        cells = eligible_cells[eligible_cell_labels == label]
        balanced_rate = stable_sum(values=source[cells] +
            (0. if storage is None else storage[cells]))
        if balanced_rate == 0.:
            eligible[int(label)] = cells
    return eligible


def unbalanced_ungrounded_components(*, labels, active, grounded, source):
    """Find unrestrained components whose signed source cannot balance.

    :param labels: Connected-component labels for potentially conducting faces.
    :param active: Active-cell mask.
    :param grounded: Cells with a fixed head, potential boundary, or storage
        restraint.
    :param source: Signed source rate by cell.
    :return: Component labels and signed net rates for unbalanced components.
    """
    labels = np.asarray(labels).ravel()
    active = np.asarray(active).ravel() != 0
    grounded = np.asarray(grounded).ravel() != 0
    source = np.asarray(source, dtype=np.float64).ravel()
    candidates = set(np.unique(labels[active])) - set(np.unique(labels[active & grounded]))
    unbalanced = []
    for label in candidates:
        cells = active & (labels == label)
        net_source = stable_sum(values=source[cells])
        if net_source != 0.:
            unbalanced.append((int(label), int(cells.sum()), float(net_source)))
    return unbalanced


def local_balance_valid(*, case, free, left, right, head, residual, flux, conductance, ghb_conductance,
                        ghb_exchange, component_anchor, thresholds, storage=None, tangent=None,
                        boundary_throughput=None):
    """Check each free control volume at its own physical flow and head scales.

    Internal conductances do not restrain a common head shift. Only an
    external boundary or a storage tangent can supply that head scale.
    Relative tolerances use actual local fluxes. A representability allowance
    applies to differences of distinct stored heads across flowing faces.
    """
    head = np.asarray(head, dtype=np.float64).ravel()
    fixed = case.fixed.ravel() != 0
    # Package entry magnitudes are a reporting quantity. The numerical row
    # sees only their signed cell sum, including exact cancellation.
    source_throughput = np.abs(case.source.ravel())
    boundary_gross = np.abs(ghb_exchange.ravel()) if boundary_throughput is None else np.asarray(boundary_throughput).ravel()
    throughput = source_throughput + boundary_gross
    anchor = ghb_conductance.ravel().copy()
    if storage is not None:
        throughput += np.abs(storage.ravel())
        anchor += tangent.ravel()
    if component_anchor is not None:
        shared = component_anchor.ravel()
        anchor = np.where(shared > 0.0, shared, anchor)
    roundoff = np.zeros_like(head)
    eps = np.finfo(np.float64).eps
    for owner, neighbour in ((left, right), (right, left)):
        throughput += np.bincount(owner, weights=np.abs(flux), minlength=head.size)
        anchor += np.bincount(owner, weights=conductance * fixed[neighbour], minlength=head.size)
        allowance = (eps * np.abs(head[owner]) + eps * np.abs(head[neighbour])) * conductance
        roundoff += np.bincount(owner, weights=np.where(flux != 0.0, allowance, 0.0), minlength=head.size)
    flow_allowance = np.maximum(thresholds['physical_rms_m3_day'],
                                throughput * thresholds['relative_cell_imbalance'])
    anchored_head_tolerance = thresholds.get('anchored_cell_head_error_m', thresholds['head_equivalent_rms_m'])
    allowance = np.where(anchor > 0.0, anchor * anchored_head_tolerance, flow_allowance) + roundoff
    free = free.ravel()
    return bool(np.isfinite(allowance[free]).all() and
                np.all(np.abs(residual.ravel()[free]) <= allowance[free]))

def componentwise_roundoff(*, head, source, residual, free, left, right,
                          conductance, ghb_conductance, ghb_head,
                          storage=None, tangent=None, previous=None,
                          previous_tangent=None, storage_coordinate_scale=None,
                          boundary_row_scale=None):
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
    if boundary_row_scale is not None:
        scale += np.asarray(boundary_row_scale, dtype=np.float64).ravel()
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
        and conservation and certificate['roundoff_rows_valid'] and checks.get('local_balance_valid', True))
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
