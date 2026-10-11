# SPDX-License-Identifier: AGPL-3.0-only
"""Sparse coastal subfaces for a depth-integrated hydrostatic hot-start model.

Submerged faces impose total sea head; exposed faces discharge only. Records
are supplied by the caller's authoritative geometry, not inferred from a mask.
This is a saturated-conductivity Dupuit approximation, not a Richards operator.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import warp as wp

from .kernels import WP_FLOAT


@dataclass(frozen=True)
class CoastalSubfaces2D:
    rows: np.ndarray
    columns: np.ndarray
    head_m: np.ndarray
    conductance_m2_per_time: np.ndarray
    submerged: np.ndarray

    def validate(self, *, active, prescribed):
        arrays = [np.asarray(getattr(self, name)) for name in self.__dataclass_fields__]
        if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
            raise ValueError('coastal subface arrays must be aligned vectors')
        rows, cols, heads, cond, submerged = arrays
        if (not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(cols.dtype, np.integer)
                or np.any(rows < 0) or np.any(rows >= active.shape[0])
                or np.any(cols < 0) or np.any(cols >= active.shape[1])
                or not np.all(np.isfinite(heads)) or not np.all(np.isfinite(cond))
                or np.any(cond <= 0) or np.any(~np.isin(submerged, [0, 1]))):
            raise ValueError('invalid coastal owner, head, conductance or type')
        if np.any(~active[rows, cols]) or np.any(prescribed[rows, cols]):
            raise ValueError('coastal owners must be free active cells')

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in self.__dataclass_fields__:
            array = np.ascontiguousarray(getattr(self, name))
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def fluxes(self, *, head):
        delta = np.asarray(head)[self.rows, self.columns] - self.head_m
        return self.conductance_m2_per_time * np.where(self.submerged, delta, np.maximum(delta, 0.))


@wp.kernel
def coastal_add_kernel(head: wp.array(dtype=WP_FLOAT, ndim=2),
                       vector: wp.array(dtype=WP_FLOAT, ndim=2),
                       rows: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
                       levels: wp.array(dtype=WP_FLOAT), cond: wp.array(dtype=WP_FLOAT),
                       submerged: wp.array(dtype=wp.int32), mode: int,
                       output: wp.array(dtype=WP_FLOAT, ndim=2)):
    n = wp.tid()
    j, i = rows[n], cols[n]
    difference = head[j, i] - levels[n]
    enabled = submerged[n] != 0 or difference > WP_FLOAT(0.)
    if enabled:
        term = cond[n] * difference
        if mode == 1:
            term = cond[n] * vector[j, i]
        elif mode == 2:
            term = cond[n]
        elif mode == 3:
            # Source side of a frozen-T affine boundary row. The diagonal
            # C*H is already present in the frozen matrix action.
            term = -cond[n] * levels[n]
        wp.atomic_add(output, j, i, term)


@wp.kernel
def coastal_norm_kernel(residual: wp.array(dtype=WP_FLOAT, ndim=2),
                        free: wp.array(dtype=wp.int32, ndim=2),
                        square_sum: wp.array(dtype=wp.float64), maximum: wp.array(dtype=wp.float64)):
    j, i = wp.tid()
    if free[j, i] != 0:
        value = wp.float64(residual[j, i])
        wp.atomic_add(square_sum, 0, value * value)
        wp.atomic_max(maximum, 0, wp.abs(value))


class CoastalDeviceTerms:
    def __init__(self, *, boundary, device):
        self.boundary = boundary
        self.device = device
        self.fields = [wp.array(np.asarray(getattr(boundary, name)),
            dtype=wp.int32 if name in ('rows', 'columns', 'submerged') else WP_FLOAT,
            device=device) for name in boundary.__dataclass_fields__]

    def add(self, *, head, vector, output, mode):
        if self.fields[0].size:
            wp.launch(kernel=coastal_add_kernel, dim=self.fields[0].size,
                inputs=[head, vector, *self.fields, mode, output], device=self.device)

    def model_rhs(self, *, head, branch_head, out):
        if self.fields[0].size:
            wp.launch(kernel=coastal_model_rhs_kernel, dim=self.fields[0].size,
                inputs=[head, branch_head, *self.fields, out], device=self.device)

    def activation_diagnostics(self, *, first, second, counts):
        if self.fields[0].size:
            wp.launch(kernel=coastal_activation_diagnostics_kernel, dim=self.fields[0].size,
                inputs=[first, second, *self.fields, counts], device=self.device)

    def branch_changes(self, *, first, second, count):
        if self.fields[0].size:
            wp.launch(kernel=coastal_branch_changes_kernel, dim=self.fields[0].size,
                inputs=[first, second, *self.fields, count], device=self.device)


@wp.kernel
def coastal_model_rhs_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2), branch: wp.array(dtype=WP_FLOAT, ndim=2),
    rows: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
    levels: wp.array(dtype=WP_FLOAT), cond: wp.array(dtype=WP_FLOAT),
    submerged: wp.array(dtype=wp.int32), output: wp.array(dtype=WP_FLOAT, ndim=2),
):
    n = wp.tid()
    if submerged[n] == 0:
        j, i = rows[n], cols[n]
        gap = head[j, i]-levels[n]
        affine = WP_FLOAT(0.)
        if branch[j, i] > levels[n]:
            affine = gap
        wp.atomic_add(output, j, i, cond[n]*(wp.max(gap, WP_FLOAT(0.))-affine))


@wp.kernel
def coastal_branch_changes_kernel(
    first: wp.array(dtype=WP_FLOAT, ndim=2), second: wp.array(dtype=WP_FLOAT, ndim=2),
    rows: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
    levels: wp.array(dtype=WP_FLOAT), cond: wp.array(dtype=WP_FLOAT),
    submerged: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
):
    n = wp.tid()
    j, i = rows[n], cols[n]
    if submerged[n] == 0 and ((first[j, i] > levels[n]) != (second[j, i] > levels[n])):
        wp.atomic_add(count, 0, 1)


@wp.kernel
def gated_diagonal_kernel(head: wp.array(dtype=WP_FLOAT, ndim=2),
                          free: wp.array(dtype=wp.int32, ndim=2),
                          drn_mask: wp.array(dtype=wp.int32, ndim=2),
                          drn_elev: wp.array(dtype=WP_FLOAT, ndim=2),
                          drn_cond: wp.array(dtype=WP_FLOAT, ndim=2),
                          riv_mask: wp.array(dtype=wp.int32, ndim=2),
                          riv_bottom: wp.array(dtype=WP_FLOAT, ndim=2),
                          riv_cond: wp.array(dtype=WP_FLOAT, ndim=2),
                          drain_equality_active: int,
                          diagonal: wp.array(dtype=WP_FLOAT, ndim=2)):
    j, i = wp.tid()
    if free[j, i] != 0:
        # Match the analytical Jacobian's active-side selection at equality;
        # this is a linearisation, not a nonzero physical drain discharge.
        if drn_mask[j, i] != 0 and (head[j, i] > drn_elev[j, i]
                or (drain_equality_active != 0 and head[j, i] == drn_elev[j, i])):
            diagonal[j, i] += drn_cond[j, i]
        if riv_mask[j, i] != 0 and head[j, i] > riv_bottom[j, i]:
            diagonal[j, i] += riv_cond[j, i]


@wp.kernel
def coastal_activation_diagnostics_kernel(
    first: wp.array(dtype=WP_FLOAT, ndim=2), second: wp.array(dtype=WP_FLOAT, ndim=2),
    rows: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
    levels: wp.array(dtype=WP_FLOAT), cond: wp.array(dtype=WP_FLOAT),
    submerged: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
):
    n = wp.tid()
    j, i = rows[n], cols[n]
    if submerged[n] == 0:
        old = first[j, i] > levels[n]
        new = second[j, i] > levels[n]
        if old != new:
            index = 5
            if new:
                index = 4
            wp.atomic_add(counts, index, 1)
