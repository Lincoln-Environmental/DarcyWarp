# SPDX-License-Identifier: Apache-2.0
# GMRES._run adapted from NVIDIA Warp (Copyright NVIDIA CORPORATION 2024-2026).
# The adaptation retains persistent restart graphs across solves as well as
# persistent Krylov arrays. Only scalar norms cross the device at restarts.
import math
import warp as wp
from warp.optim.linear import GMRES
from warp._src.optim.linear import (
    _initialize_absolute_tolerance, _gmres_update_x_kernel,
    _gmres_add_active_entries_kernel,
)


class TerrainGMRES3D(GMRES):
    def _run(self, A, b, x, M):
        device = self._device
        scalar_dtype = self._scalar_type
        restart = self._restart
        is_left_preconditioner = self._is_left_preconditioner
        batch_offsets = A.batch_offsets
        dofs_per_entry = self._dofs_per_entry

        tiled_dot = self._tiled_dot
        r = self._r
        w = self._w
        H = self._H
        y = self._y
        V = self._V
        beta = self._beta
        atol_sq = self._atol_sq
        w_repeated = self._w_repeated
        r_norm_sq = tiled_dot.col(0)

        least_squares_solve = self._least_squares_solve
        normalize_arnoldi_vec = self._normalize_arnoldi_vec
        arnoldi_axpy = self._arnoldi_axpy
        copy_hessenberg_col = self._copy_hessenberg_col

        # Initialize tolerance from right-hand-side norm
        _initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        # Initialize residual
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)
        tiled_dot.compute(r, r, col_offset=0)

        # Not strictly necessary, but makes it more robust to user-provided LinearOperators
        w.zero_()

        def do_arnoldi_iteration(j: int):
            # w = A * v[j];
            if M is not None:
                tmp = V[j + 1]

                if is_left_preconditioner:
                    A.matvec(V[j], tmp, tmp, alpha=1, beta=0)
                    M.matvec(tmp, w, w, alpha=1, beta=0)
                else:
                    M.matvec(V[j], tmp, tmp, alpha=1, beta=0)
                    A.matvec(tmp, w, w, alpha=1, beta=0)
            else:
                A.matvec(V[j], w, w, alpha=1, beta=0)

            # compute and apply dot products in parallel,
            # since the Hj column entries are independent per batch
            # Keep both operands at j + 1 rows: TiledDot uses the first operand
            # to choose the launch width, so an unsliced repeated view would read
            # past the active V slice in debug bounds-checked builds.
            tiled_dot.compute(w_repeated[: j + 1], V[: j + 1])
            copy_hessenberg_col.set_param_at_index(0, j)
            copy_hessenberg_col.launch()

            # w -= sum_k H[:, k, j] * v[k]
            arnoldi_axpy.set_param_at_index(0, j)
            arnoldi_axpy.launch()

            # H[:, j+1, j] = ||w||; normalize w into v[j+1]
            tiled_dot.compute(w, w)
            normalize_arnoldi_vec.set_params(
                [w, V[j + 1], tiled_dot.col(0), H[:, j + 1, j], batch_offsets, dofs_per_entry]
            )
            normalize_arnoldi_vec.launch()

        def do_restart_cycle(iterations):
            # A shortened final block leaves stale Hessenberg columns that
            # would pollute the least-squares solve; zeroing restores the
            # exact solution of the shorter Krylov system (their y entries
            # become zero, so the full-width update below stays correct).
            if iterations < restart:
                H.zero_()
            if M is not None and is_left_preconditioner:
                M.matvec(r, w, w, alpha=1, beta=0)
                rh = w
            else:
                rh = r

            # beta^2 = rh.rh
            tiled_dot.compute(rh, rh)

            # v[0] = rh / beta
            normalize_arnoldi_vec.set_params([rh, V[0], tiled_dot.col(0), beta, batch_offsets, dofs_per_entry])
            normalize_arnoldi_vec.launch()

            for j in range(iterations):
                do_arnoldi_iteration(j)

            least_squares_solve.launch()

            # update x
            if M is None or is_left_preconditioner:
                wp.launch(
                    _gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(1.0), y, V, x, batch_offsets, dofs_per_entry],
                )
            else:
                wp.launch(
                    _gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(0.0), y, V, w, batch_offsets, dofs_per_entry],
                )
                M.matvec(w, r, r, alpha=1, beta=0)
                wp.launch(
                    _gmres_add_active_entries_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[r, x, batch_offsets, dofs_per_entry],
                )

            # update r and residual
            wp.copy(src=b, dest=r)
            A.matvec(x, b, r, alpha=-1.0, beta=1.0)
            tiled_dot.compute(r, r)

        return self._run_cached_cycle(
            do_cycle=do_restart_cycle,
            cur_iter_and_condition=self._cur_iter_and_condition,
            cycle_size=restart,
            r_norm_sq=r_norm_sq,
            maxiter=self._maxiter,
            atol_sq=atol_sq,
            callback=self._callback,
            check_every=self._check_every,
            use_cuda_graph=self._use_cuda_graph,
        )

    def _run_cached_cycle(self, *, do_cycle, cur_iter_and_condition, cycle_size,
                          r_norm_sq, maxiter, atol_sq, callback,
                          check_every, use_cuda_graph):
        tolerance = math.sqrt(float(atol_sq.numpy().max()))
        error = math.sqrt(float(r_norm_sq.numpy().max()))
        used = 0
        if not hasattr(self, "graph_captures"):
            self.graph_captures = 0
            self.graph_replays = 0
            self._cycle_graph = None
        while error > tolerance and used < maxiter:
            # maxiter is the exact permitted Krylov-iteration budget, not a
            # number of whole restart blocks: the final block is shortened.
            iterations = min(cycle_size, maxiter - used)
            if use_cuda_graph and iterations == cycle_size and self._device.is_cuda:
                if self._cycle_graph is None:
                    with wp.ScopedCapture(device=self._device) as capture:
                        do_cycle(cycle_size)
                    self._cycle_graph = capture.graph
                    self.graph_captures += 1
                wp.capture_launch(self._cycle_graph)
                self.graph_replays += 1
            else:
                do_cycle(iterations)
            used += iterations
            error = math.sqrt(max(float(r_norm_sq.numpy().max()), 0.0))
            if callback is not None:
                callback(used, error, tolerance)
        return used, error, tolerance
