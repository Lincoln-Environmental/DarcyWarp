# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in sparse algebraic preconditioner for steady preparation only.

Fine residual/Jacobian and FGMRES acceptance remain native Warp. A fixed
CPU AMG V-cycle supplies a candidate correction. Full-vector PCIe transfers
are intentional and reported: this backend is unsuitable for storm timesteps.
PyAMG is an optional dependency, imported only when explicitly selected.
"""
import time

import numpy as np
from scipy import sparse
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT


class SteadyAMGPreconditioner2D:
    def __init__(self, *, operator, model, exact_jacobian=False):
        self.operator = operator
        self.model = model
        self.exact_jacobian = bool(exact_jacobian)
        self.applications = 0
        self.cycles = 0
        self.setup_wall_s = 0.
        self.apply_wall_s = 0.
        self.full_vector_transfer_bytes = 0
        self.fine_diagonal_inverse = wp.zeros(operator.ctx.shape, dtype=WP_FLOAT, device=model.device_str)
        self.hierarchy = None

    def freeze(self, *, transmissivity, storage_diagonal):
        import pyamg
        started = time.perf_counter()
        ctx = self.operator.ctx
        shape = ctx.shape
        active, fixed = np.asarray(ctx.boundaries.active, bool), np.asarray(ctx.boundaries.dirichlet_mask, bool)
        free = active & ~fixed
        T = np.asarray(transmissivity.numpy(), float)
        diagonal = np.asarray(storage_diagonal.numpy(), float).copy()
        diagonal += T*np.where(ctx.boundaries.ghb_mask, ctx.boundaries.ghb_factor, 0.)
        index = np.arange(T.size).reshape(shape)
        rows, columns, values = [], [], []
        for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
            connected = active[first] & active[second]
            conductance = np.where(connected, 2*T[first]*T[second]/(T[first]+T[second]+1e-12), 0.)
            diagonal[first] += conductance
            diagonal[second] += conductance
            valid = connected & free[first] & free[second]
            a, b, g = index[first][valid], index[second][valid], conductance[valid]
            rows.extend((a, b))
            columns.extend((b, a))
            values.extend((-g, -g))
        diagonal[~free] = 1.
        if np.any(diagonal[free] <= 0):
            raise ValueError('isolated groundwater cell lacks a physical derivative; refuse identity substitution')
        rows.append(index.ravel())
        columns.append(index.ravel())
        values.append(diagonal.ravel())
        matrix = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(T.size, T.size)).tocsr()
        if self.exact_jacobian:
            from .steady_sparse_preconditioner import steady_jacobian_matrix
            matrix = steady_jacobian_matrix(context=ctx, head=self.operator.head_device.numpy(),
                boundary_diagonal=storage_diagonal.numpy()).tocsr()
            diagonal = np.maximum(np.abs(matrix.diagonal()).reshape(shape), np.finfo(float).tiny)
        self.fine_diagonal_inverse.assign(np.ascontiguousarray(1./diagonal))
        self.hierarchy = pyamg.ruge_stuben_solver(A=matrix, max_coarse=64,
            max_levels=len(self.model.mg_levels),
            presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            postsmoother=('gauss_seidel', {'sweep': 'symmetric'}))
        self.setup_wall_s += time.perf_counter()-started

    def apply(self, rhs, out):
        started = time.perf_counter()
        values = rhs.numpy().ravel().copy()
        answer = self.hierarchy.solve(b=values, x0=np.zeros_like(values), tol=0., maxiter=1, cycle='V')
        out.assign(np.ascontiguousarray(answer.reshape(rhs.shape)))
        self.applications += 1
        self.cycles += 1
        self.full_vector_transfer_bytes += 2*values.nbytes
        self.apply_wall_s += time.perf_counter()-started
