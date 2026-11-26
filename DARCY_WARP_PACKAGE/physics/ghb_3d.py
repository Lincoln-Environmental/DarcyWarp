# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell-based general-head exchange: outward flow C * (h - boundary head).

Conductance is an integrated volume/time/head coefficient, independent of
cell conductivity and saturation. One equivalent boundary is stored per cell.
"""
import numpy as np

from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask, floating_field
from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values


def validate_ghb_fields(*, conductance, head, active, fixed, mask=None):
    """Own finite full-grid inputs; validate before conversion or assignment."""
    shape = np.asarray(active).shape
    if conductance is None and head is None and mask is None:
        return None, None
    if conductance is None or head is None:
        raise ValueError("GHB requires both ghb_conductance and ghb_head")
    c = floating_field(value=conductance, name="ghb_conductance")
    hb = floating_field(value=head, name="ghb_head")
    if c.ndim not in (0, len(shape)) or hb.ndim not in (0, len(shape)):
        raise ValueError("GHB fields must be scalars or full-grid arrays")
    if (c.ndim and c.shape != shape) or (hb.ndim and hb.shape != shape):
        raise ValueError(f"GHB fields must have shape {shape}")
    if np.any(c < 0.0):
        raise ValueError("ghb_conductance must be nonnegative")
    c = np.broadcast_to(array=c, shape=shape).copy()
    hb = np.broadcast_to(array=hb, shape=shape).copy()
    selected = c > 0.0 if mask is None else binary_mask(value=mask, name="ghb_mask", shape=shape) != 0
    if np.any(selected & ((np.asarray(active) == 0) | (np.asarray(fixed) != 0))):
        raise ValueError("GHB cells must be active and cannot also have prescribed heads")
    if mask is not None and np.any((c != 0.0) & ~selected):
        # A scalar conductance with an explicit mask is a convenient uniform C.
        if np.asarray(conductance).ndim != 0:
            raise ValueError("Nonzero ghb_conductance outside ghb_mask would be lost")
    c[~selected] = 0.0
    hb[~selected] = 0.0
    if np.any(invalid_head_values(head=hb[c > 0.0])):
        raise ValueError("Invalid active GHB head: reserved dry/inactive marker on a conducting boundary")
    with np.errstate(over="ignore", invalid="ignore"):
        forcing = c * hb
    if not np.all(np.isfinite(forcing)):
        raise ValueError("GHB conductance times boundary head must be representable")
    return c, hb


def add_ghb_linear_terms(*, rhs, diagonal, conductance, head):
    """Add the GHB diagonal/forcing while retaining separate physical storage."""
    if conductance is None:
        return rhs, diagonal
    with np.errstate(over="ignore", invalid="ignore"):
        result_rhs = np.asarray(rhs) + conductance * head
        result_diagonal = np.asarray(diagonal) + conductance
    if not np.all(np.isfinite(result_rhs)) or not np.all(np.isfinite(result_diagonal)):
        raise ValueError("Assembled GHB forcing or diagonal is not representable")
    return result_rhs, result_diagonal


def ghb_flux(*, head, conductance, boundary_head):
    """Evaluate physical exchange directly, avoiding C*h minus C*h_boundary."""
    if conductance is None:
        return np.zeros_like(a=head, dtype=np.float64)
    result = np.zeros_like(a=conductance, dtype=np.float64)
    np.subtract(head, boundary_head, out=result, where=conductance > 0.0)
    result *= conductance
    return result


def npf_ghb_component_anchor(*, conductance, active, fixed, left=None, right=None,
                             face_conductance=None, return_layout=False):
    """Share the GHB head restraint over each connected aquifer without a CHD.

    For a uniform head shift, internal flows cancel and the derivative of the
    component's external discharge is sum(C_ghb). Dividing that conductance by
    the number of cells assigns a local flow tolerance whose sum retains the
    same head tolerance. Current positive face conductances define connectivity
    when edge arrays are supplied, so dry faces cannot share restraint between
    wet components. Zero marks CHD-grounded components.
    """
    active_grid = np.asarray(active) != 0
    fixed_grid = np.asarray(fixed) != 0
    if active_grid.shape != fixed_grid.shape:
        raise ValueError("Active and fixed grids must have matching shapes")
    if face_conductance is None:
        if left is not None or right is not None:
            raise ValueError("Face connectivity requires left, right and conductance arrays")
        from scipy.ndimage import label
        labels, count = label(active_grid)
        component_count = count + 1
    else:
        if left is None or right is None:
            raise ValueError("Face connectivity requires both endpoint arrays")
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components
        edge_left = np.asarray(left)
        edge_right = np.asarray(right)
        faces = np.asarray(face_conductance, dtype=np.float64)
        if edge_left.ndim != 1 or edge_right.ndim != 1 or faces.ndim != 1:
            raise ValueError("Face connectivity arrays must be one-dimensional")
        if not (edge_left.shape == edge_right.shape == faces.shape):
            raise ValueError("Face connectivity arrays must have matching lengths")
        cell_count = active_grid.size
        valid = ((edge_left >= 0) & (edge_left < cell_count) &
                 (edge_right >= 0) & (edge_right < cell_count))
        if np.any(~np.isfinite(faces[valid])) or np.any(faces[valid] < 0.):
            raise ValueError("Current face conductances must be finite and nonnegative")
        active_flat = active_grid.ravel()
        safe_left = np.clip(edge_left, 0, cell_count - 1)
        safe_right = np.clip(edge_right, 0, cell_count - 1)
        connected = (valid & (faces > 0.) & active_flat[safe_left] & active_flat[safe_right])
        graph = sparse.coo_matrix((np.ones(int(np.sum(connected))),
            (edge_left[connected], edge_right[connected])),
            shape=(cell_count, cell_count)).tocsr()
        component_count, flat_labels = connected_components(csgraph=graph, directed=False)
        labels = flat_labels.reshape(active_grid.shape)
    flat_labels = labels.ravel()
    active_flat = active_grid.ravel()
    sizes = np.bincount(flat_labels[active_flat], minlength=component_count)
    eligible = np.ones(component_count, dtype=np.float64)
    if face_conductance is None:
        eligible[0] = 0.
    eligible[flat_labels[fixed_grid.ravel()]] = 0.
    weights = np.zeros(active_grid.size, dtype=np.float64)
    selected = active_flat & (sizes[flat_labels] > 0)
    weights[selected] = eligible[flat_labels[selected]] / sizes[flat_labels[selected]]
    anchor = None
    if conductance is not None and np.any(conductance):
        c = np.broadcast_to(conductance, active_grid.shape).ravel()
        if np.any(~np.isfinite(c[active_flat])) or np.any(c[active_flat] < 0.):
            raise ValueError("Component restraint must be finite and nonnegative")
        # Scale each summand first so a representable average cannot overflow.
        average = np.bincount(flat_labels[active_flat],
            weights=c[active_flat] / sizes[flat_labels[active_flat]],
            minlength=component_count)
        average *= eligible
        if np.any(average):
            anchor = average[flat_labels].reshape(active_grid.shape)
            anchor[~active_grid] = 0.
    if return_layout:
        return anchor, labels.astype(np.int32, copy=False), weights.reshape(active_grid.shape)
    return anchor
