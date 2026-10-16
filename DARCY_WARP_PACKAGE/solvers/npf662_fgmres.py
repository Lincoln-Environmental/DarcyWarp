"""Persistent FP64 flexible GMRES on masked flat 3D NPF equations.

Stored Z vectors reconstruct the correction for a variable K-cycle. Scalar
Arnoldi products cross the host; full grid vectors remain device resident.
"""
from time import perf_counter
import numpy as np
import warp as wp

from .face_kernels_3d_f64 import dot_partials_3d_f64_kernel, combine_partials_3d_f64_kernel, BLOCK_SIZE_3D


@wp.kernel
def masked_copy(src: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
                dst: wp.array(dtype=wp.float64), scale: wp.float64):
    i = wp.tid()
    dst[i] = wp.float64(0.)
    if free[i] != 0:
        dst[i] = scale * src[i]


@wp.kernel
def masked_axpy(y: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
                free: wp.array(dtype=wp.int32), alpha: wp.float64):
    i = wp.tid()
    if free[i] != 0:
        y[i] += alpha * x[i]
    else:
        y[i] = wp.float64(0.)


class RestartedFGMRES3D:
    def __init__(self, *, workspace, restart):
        self.owner = workspace
        self.device = workspace.device
        self.restart = restart
        self.shape = workspace.shape
        self.n = workspace.n
        self.V = [wp.zeros(shape=self.n, dtype=wp.float64, device=self.device) for unused in range(restart + 1)]
        self.Z = [wp.zeros(shape=self.n, dtype=wp.float64, device=self.device) for unused in range(restart)]
        self.r = wp.zeros_like(src=workspace.rhs)
        self.w = wp.zeros_like(src=workspace.rhs)
        self.Ax = wp.zeros_like(src=workspace.rhs)
        self.partials = wp.zeros(shape=(self.n + BLOCK_SIZE_3D - 1) // BLOCK_SIZE_3D, dtype=wp.float64, device=self.device)
        self.scalar = wp.zeros(shape=1, dtype=wp.float64, device=self.device)
        self.active = workspace.free.reshape(shape=self.shape)
        self.fixed = wp.zeros(shape=self.shape, dtype=wp.int32, device=self.device)
        self.closed = False
        self.readbacks = 0
        self.synchronization_seconds = 0.
        self.preconditioner_seconds = 0.

    def _copy(self, *, src, dst, scale=1.):
        wp.launch(kernel=masked_copy, dim=self.n, inputs=[src, self.owner.free, dst, scale], device=self.device)

    def _axpy(self, *, y, x, alpha):
        wp.launch(kernel=masked_axpy, dim=self.n, inputs=[y, x, self.owner.free, alpha], device=self.device)

    def _dot(self, *, x, y):
        nz, ny, nx = self.shape
        self.partials.zero_()
        wp.launch(kernel=dot_partials_3d_f64_kernel, dim=self.n,
            inputs=[x.reshape(shape=self.shape), y.reshape(shape=self.shape), self.active, self.fixed,
                    self.partials, nx, ny, nz, BLOCK_SIZE_3D], device=self.device)
        wp.launch(kernel=combine_partials_3d_f64_kernel, dim=1,
                  inputs=[self.partials, self.scalar, self.partials.size], device=self.device)
        start = perf_counter()
        value = float(self.scalar.numpy()[0])
        self.synchronization_seconds += perf_counter() - start
        self.readbacks += 1
        return value

    def _norm(self, *, x):
        value = self._dot(x=x, y=x)
        return np.sqrt(value) if np.isfinite(value) and value >= 0. else np.nan

    def _matvec(self, *, x, out):
        self.owner.matvec(x=x, y=out, z=out, alpha=1., beta=0.)

    def solve(self, *, preconditioner, max_iterations, absolute_tolerance):
        if self.closed:
            raise RuntimeError('FGMRES workspace is closed')
        self.readbacks = 0
        self.synchronization_seconds = self.preconditioner_seconds = 0.
        self.owner.delta.zero_()
        self._copy(src=self.owner.rhs, dst=self.r)
        beta = self._norm(x=self.r)
        history = [dict(iterations=0, true_residual=beta)]
        used = blocks = 0
        mg_device_start = preconditioner.device_seconds
        reason = 'budget'
        while np.isfinite(beta) and beta > absolute_tolerance and used < max_iterations:
            count = min(self.restart, max_iterations - used)
            self._copy(src=self.r, dst=self.V[0], scale=1. / beta)
            H = np.zeros(shape=(count + 1, count), dtype=np.float64)
            g = np.zeros(shape=count + 1, dtype=np.float64)
            cs = np.zeros(shape=count)
            sn = np.zeros(shape=count)
            g[0] = beta
            columns = 0
            breakdown = False
            for column in range(count):
                start = perf_counter()
                preconditioner.apply(vector=self.V[column], out=self.Z[column],
                                     diagonal=self.owner.scale, free=self.owner.free)
                self.preconditioner_seconds += perf_counter() - start
                self._matvec(x=self.Z[column], out=self.w)
                # Reorthogonalized MGS is robust for nonsymmetric scaled J.
                for unused in range(2):
                    for row in range(column + 1):
                        value = self._dot(x=self.w, y=self.V[row])
                        H[row, column] += value
                        self._axpy(y=self.w, x=self.V[row], alpha=-value)
                length = self._norm(x=self.w)
                preconditioner.collect_timing()
                used += 1
                if not np.isfinite(length) or not np.all(np.isfinite(H[:, column])):
                    reason = 'nonfinite_arnoldi'
                    breakdown = True
                    break
                H[column + 1, column] = length
                happy = length <= 1e-14 * max(float(np.linalg.norm(H[:column + 1, column])), np.finfo(float).tiny)
                if not happy:
                    self._copy(src=self.w, dst=self.V[column + 1], scale=1. / length)
                for row in range(column):
                    a, b = H[row, column], H[row + 1, column]
                    H[row, column] = cs[row] * a + sn[row] * b
                    H[row + 1, column] = -sn[row] * a + cs[row] * b
                denominator = float(np.hypot(H[column, column], H[column + 1, column]))
                if denominator == 0. or not np.isfinite(denominator):
                    reason = 'arnoldi_breakdown'
                    breakdown = True
                    break
                cs[column] = H[column, column] / denominator
                sn[column] = H[column + 1, column] / denominator
                H[column, column] = denominator
                H[column + 1, column] = 0.
                g[column + 1] = -sn[column] * g[column]
                g[column] *= cs[column]
                columns = column + 1
                if abs(g[column + 1]) <= absolute_tolerance or happy:
                    reason = 'happy_breakdown' if happy else 'estimated_target'
                    break
            if columns:
                try:
                    weights = np.linalg.solve(a=H[:columns, :columns], b=g[:columns])
                except np.linalg.LinAlgError:
                    reason = 'singular_hessenberg'
                    break
                for index in range(columns):
                    self._axpy(y=self.owner.delta, x=self.Z[index], alpha=float(weights[index]))
            self._matvec(x=self.owner.delta, out=self.Ax)
            self._copy(src=self.owner.rhs, dst=self.r)
            self._axpy(y=self.r, x=self.Ax, alpha=-1.)
            beta = self._norm(x=self.r)
            blocks += 1
            history.append(dict(iterations=used, true_residual=beta))
            if breakdown or not columns:
                break
        if not np.isfinite(beta):
            reason = 'nonfinite_true_residual'
        elif beta <= absolute_tolerance:
            reason = 'target'
        self.last_info = dict(iterations=used, restarts=max(blocks - 1, 0), blocks=blocks,
            true_residual=beta, target=absolute_tolerance, converged=bool(np.isfinite(beta) and beta <= absolute_tolerance),
            stopping=reason, history=history, scalar_readbacks=self.readbacks,
            synchronization_seconds=self.synchronization_seconds,
            multigrid_device_seconds=preconditioner.device_seconds-mg_device_start,
            preconditioner_enqueue_seconds=self.preconditioner_seconds)
        return used, beta, absolute_tolerance

    def close(self):
        if not self.closed:
            self.V = self.Z = []
            self.r = self.w = self.Ax = self.partials = self.scalar = None
            self.active = self.fixed = self.owner = None
            self.closed = True
