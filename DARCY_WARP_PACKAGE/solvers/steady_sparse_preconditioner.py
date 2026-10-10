# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit steady-only sparse analytical-Jacobian ILU preconditioner.

The native Warp residual/Jv and FGMRES true-residual checks remain authoritative.
Sparse setup and full-vector transfers are permitted only during standalone
steady preparation; this is not a production transient/coupling backend.
"""
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spilu, splu
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT


def steady_jacobian_matrix(*, context, head, boundary_diagonal):
    active = np.asarray(context.boundaries.active, bool)
    free = active & ~np.asarray(context.boundaries.dirichlet_mask, bool)
    K = np.asarray(context.flow.K, float)
    bottom = np.asarray(context.flow.zbot, float)
    cap = np.full(head.shape, np.inf) if context.flow.ztop is None else np.maximum(
        np.asarray(context.flow.ztop)-bottom, context.flow.min_sat)
    raw = head-bottom
    T = K*np.minimum(np.maximum(raw, context.flow.min_sat), cap)
    derivative = K*((raw > context.flow.min_sat) & (raw < cap))
    diagonal = np.asarray(boundary_diagonal, float).copy()
    gh_factor = np.where(context.boundaries.ghb_mask, context.boundaries.ghb_factor, 0.)
    diagonal += gh_factor*(T+derivative*(head-context.boundaries.ghb_external_head))
    indices = np.arange(head.size).reshape(head.shape)
    rows, cols, values = [], [], []
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        connection = active[first] & active[second]
        a, b = T[first], T[second]
        denominator = a+b+1e-12
        g = np.where(connection, 2*a*b/denominator, 0.)
        da = np.where(connection, 2*b*(b+1e-12)*derivative[first]/denominator**2, 0.)
        db = np.where(connection, 2*a*(a+1e-12)*derivative[second]/denominator**2, 0.)
        dh = head[first]-head[second]
        diagonal[first] += g+da*dh
        diagonal[second] += g-db*dh
        connected_free = connection & free[first] & free[second]
        rows.extend((indices[first][connected_free], indices[second][connected_free]))
        cols.extend((indices[second][connected_free], indices[first][connected_free]))
        values.extend(((-g+db*dh)[connected_free], (-g-da*dh)[connected_free]))
    diagonal[~free] = 1.
    rows.append(indices.ravel())
    cols.append(indices.ravel())
    values.append(diagonal.ravel())
    matrix = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(head.size, head.size)).tocsc()
    return matrix


class SteadySparsePreconditioner2D:
    def __init__(self, *, operator, model, exact=False):
        self.exact = bool(exact)
        self.operator, self.model = operator, model
        self.applications = self.cycles = self.full_vector_transfer_bytes = 0
        self.setup_wall_s = self.apply_wall_s = 0.
        self.fine_diagonal_inverse = wp.zeros(operator.ctx.shape, dtype=WP_FLOAT, device=model.device_str)
        self.factor = None

    def freeze(self, *, transmissivity, storage_diagonal):
        started = time.perf_counter()
        head = self.operator.head_device.numpy().copy()
        matrix = steady_jacobian_matrix(context=self.operator.ctx, head=head,
            boundary_diagonal=storage_diagonal.numpy())
        # This is a preconditioner only. Sparse fill never changes the actual
        # equation, candidate head or native acceptance tolerance.
        self.factor = (splu(A=matrix, permc_spec="MMD_AT_PLUS_A") if self.exact else
            spilu(A=matrix, drop_tol=1e-4, fill_factor=10.))
        diagonal = np.maximum(np.abs(matrix.diagonal()), np.finfo(float).tiny)
        self.fine_diagonal_inverse.assign((1./diagonal).reshape(head.shape))
        self.setup_wall_s += time.perf_counter()-started

    def apply(self, rhs, out):
        started = time.perf_counter()
        vector = rhs.numpy().ravel().copy()
        answer = self.factor.solve(vector)
        out.assign(np.ascontiguousarray(answer.reshape(rhs.shape)))
        self.applications += 1
        self.cycles += 1
        self.full_vector_transfer_bytes += 2*vector.nbytes
        self.apply_wall_s += time.perf_counter()-started
